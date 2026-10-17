"""Paired CE/Steer one-step diagnostics from existing trained checkpoints.

Each trial restores identical weights and RNG, with a fresh AdamW8bit
optimizer. This is not resuming or reconstructing the historical optimizer.
"""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bitsandbytes as bnb
import peft
import torch
import torch.nn.functional as F
import transformers
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'methods'))
from train_fixed_margin_controls import read_split, encode, make_probes, evaluate
from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens
from audit_margin_artifacts import load_loss
from update_direction_metrics import cosine, tensor_inner, effective_factors, factor_inner, validate_resume

TARGETS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
MODELS=('full_sft','lora','steer_full','steer_lora')


def dump(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def dot_dict(a,b):
    assert a.keys()==b.keys()
    return sum(tensor_inner(a[n],b[n]) for n in a)


def layer_name(name):
    return name.split('.layers.')[1].split('.')[0] if '.layers.' in name else 'other'


def layer_dots(a,b):
    assert a.keys()==b.keys()
    result={}
    for name in a:
        key=layer_name(name)
        result[key]=result.get(key,0.)+tensor_inner(a[name],b[name])
    return result


def effective_layer_dots(a,b,adapter):
    result={}
    for name in a:
        key=layer_name(name)
        value=factor_dot(a[name],b[name]) if adapter else tensor_inner(a[name],b[name])
        result[key]=result.get(key,0.)+value
    return result


def projection(name):
    return name.endswith('.weight') and name.split('.')[-2] in TARGETS


def restore(parameters,before):
    with torch.no_grad():
        for name,p in parameters:
            p.copy_(before[name].to(p.device))
    # Sample integrity checks supplement exact copy_ of every trainable tensor.
    for name,p in parameters[::max(1,len(parameters)//12)]:
        assert torch.equal(p.detach().cpu(),before[name]),name


def make_effective(before,after,adapter,scale):
    if not adapter:
        return {n:after[n] for n in after if projection(n)}
    factors={}
    for name in before:
        if '.lora_A.' not in name:continue
        other=name.replace('.lora_A.','.lora_B.')
        canonical=name.split('.lora_A.')[0].removeprefix('base_model.model.')+'.weight'
        factors[canonical]=effective_factors(before[name].float(),before[other].float(),
                                            after[name].float(),after[other].float(),scale)
    return factors


def effective_dot(a,b,adapter):
    assert a.keys()==b.keys()
    if adapter:return sum(factor_dot(a[n],b[n]) for n in a)
    return sum(tensor_inner(a[n],b[n]) for n in a)


def factor_dot(a,b=None):
    b=a if b is None else b
    return factor_inner(*(x.to('cuda') for x in (*a,*b)))


def objective_step(model,parameters,before,batches,objective,seed,lr,adapter,scale):
    restore(parameters,before);model.zero_grad(set_to_none=True)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model.train()
    decay=[p for n,p in parameters if not (n.endswith('bias') or 'layernorm' in n or '.norm.' in n)]
    no_decay=[p for n,p in parameters if n.endswith('bias') or 'layernorm' in n or '.norm.' in n]
    groups=[{'params':decay,'weight_decay':.1}]
    if no_decay:groups.append({'params':no_decay,'weight_decay':0.})
    optimizer=bnb.optim.AdamW8bit(groups,lr=lr,betas=(.9,.999),eps=1e-8)
    denominator=sum(int(labels[:,1:].ne(-100).sum()) for _,labels in batches)
    if denominator<=0:raise ValueError('No assistant tokens in diagnostic batch')
    loss_sum=0.;steer=load_loss()(1.5)
    started=time.monotonic()
    for ids,labels in batches:
        ids,labels=ids.to('cuda'),labels.to('cuda')
        outputs=model(input_ids=ids,use_cache=False)
        if objective=='ce':
            loss=F.cross_entropy(outputs.logits[:,:-1].reshape(-1,outputs.logits.size(-1)).float(),
                                 labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/denominator
        else:loss=steer(outputs,labels,denominator)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite diagnostic loss')
        loss.backward();loss_sum+=loss.item()
        del loss,outputs,ids,labels
    norm_before=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.).item()
    if not math.isfinite(norm_before):raise FloatingPointError('Nonfinite gradient')
    gradients={n:p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(before[n]) for n,p in parameters}
    optimizer.step()
    with torch.no_grad():
        updates={n:(p.detach().float()-before[n].to(p.device).float()).cpu() for n,p in parameters}
        after={n:p.detach().cpu() for n,p in parameters} if adapter else updates
    effective=make_effective(before,after,adapter,scale)
    gradient_layers=layer_dots(gradients,gradients);update_layers=layer_dots(updates,updates)
    gradient_sq=sum(gradient_layers.values());update_sq=sum(update_layers.values())
    gu=dot_dict(gradients,updates)
    eff_sq=effective_dot(effective,effective,adapter)
    layer_norms={}
    for name,values in effective.items():
        layer=name.split('.')[2]
        value=factor_dot(values) if adapter else tensor_inner(values,values)
        layer_norms[layer]=layer_norms.get(layer,0.)+value
    record={'objective':objective,'loss':loss_sum,'assistant_tokens':denominator,
            'gradient_norm_before_clip':norm_before,'gradient_norm_after_clip':math.sqrt(gradient_sq),
            'layer_native_gradient_l2':{k:math.sqrt(max(v,0.)) for k,v in gradient_layers.items()},
            'layer_native_update_l2':{k:math.sqrt(max(v,0.)) for k,v in update_layers.items()},
            'clip_factor':min(1.,1/max(norm_before,1e-30)),
            'native_update_l2':math.sqrt(update_sq),'gradient_update_cosine':cosine(gu,gradient_sq,update_sq),
            'effective_common_update_l2':math.sqrt(max(eff_sq,0.)),
            'changed_native_parameters':sum(int(v.ne(0).sum()) for v in updates.values()),
            'layer_common_update_norm':{k:math.sqrt(max(v,0.)) for k,v in layer_norms.items()},
            'elapsed_seconds':time.monotonic()-started,'optimizer_moments':'fresh zeros, not historical',
            'learning_rate':lr}
    model.zero_grad(set_to_none=True);del optimizer,after
    torch.cuda.empty_cache()
    return record,gradients,updates,effective


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,default=ROOT/'artifacts/samsum/fair-methods-2epoch-log25-tmux')
    parser.add_argument('--base-model',type=Path,default=ROOT/'artifacts/base/Qwen2-7B')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--batches',type=int,default=4)
    parser.add_argument('--accumulation',type=int,default=64)
    parser.add_argument('--dev-samples',type=int,default=64)
    parser.add_argument('--boolq-samples',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--resume',action='store_true',help='Reuse completed pairs; extend batch count only with identical scope')
    args=parser.parse_args()
    if min(args.batches,args.accumulation,args.dev_samples,args.boolq_samples)<=0:raise ValueError('Positive sample counts required')
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    previous=json.loads((args.output_dir/'run_manifest.json').read_text()) if args.resume else None
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(args.base_model,local_files_only=True)
    configure_qwen2_base_tokenizer(tokenizer,str(args.base_model));tokenizer.padding_side='left'
    train,dev=read_split('train'),read_split('dev')
    indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed))[:args.batches*args.accumulation].tolist()
    assert len(indices)==args.batches*args.accumulation
    encoded={i:encode(tokenizer,train[i]['messages']) for i in indices}
    probes=make_probes(tokenizer,dev,args.dev_samples,args.seed)
    documents=[json.loads(l) for l in (args.artifact_root/'reports/margin-mechanism/boolq.validation.jsonl').read_text().splitlines()]
    selected=torch.randperm(len(documents),generator=torch.Generator().manual_seed(args.seed))[:args.boolq_samples].tolist()
    docs=[documents[i] for i in selected];labels=torch.tensor([d['label'] for d in docs])
    raw=[f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:" for d in docs]
    prompts={'raw':raw,'chat':[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in raw]}
    choices={fmt:torch.tensor([tokenizer.encode((' ' if fmt=='raw' else '')+s,add_special_tokens=False)[0] for s in ('no','yes')]) for fmt in prompts}
    for fmt in prompts:
        assert all(len(tokenizer.encode((' ' if fmt=='raw' else '')+s,add_special_tokens=False))==1 for s in ('no','yes'))
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(),
              'models':args.models,'batches':args.batches,'accumulation':args.accumulation,'train_indices':indices,
              'dev_indices':[p[0] for p in probes],'boolq_doc_ids':[d['doc_id'] for d in docs],
              'seed':args.seed,'learning_rate':args.learning_rate,'fixed_margin':math.log(2.5),
              'optimizer':'AdamW8bit, beta .9/.999, eps1e-8, original-style decay groups .1/0',
              'optimizer_scope':'Every CE/Steer trial resets moments; old optimizer state absent. No historical continuation.',
              'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if previous is not None:
        validate_resume(previous,manifest)
        manifest['initial_started_utc']=previous.get('initial_started_utc',previous['started_utc'])
        manifest['resume_history']=previous.get('resume_history',[])+[{'previous_batches':previous['batches'],
            'previous_source_sha256':previous['source_sha256'],'resumed_utc':manifest['started_utc']}]
    dump(args.output_dir/'run_manifest.json',manifest)
    base=AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
    sync_model_special_tokens(base,tokenizer)
    base_result,reference=evaluate(base,tokenizer,probes,prompts,choices,labels,None,args.output_dir,'base',0)
    dump(args.output_dir/'base_probe.json',base_result)
    del base;gc.collect();torch.cuda.empty_cache()
    results=json.loads((args.output_dir/'results.json').read_text()) if args.resume and (args.output_dir/'results.json').exists() else []
    completed={(r['model'],r['batch']) for r in results}
    for name in args.models:
        if all((name,i) in completed for i in range(args.batches)):continue
        path=args.artifact_root/'models'/name;adapter=(path/'adapter_config.json').exists()
        model=AutoModelForCausalLM.from_pretrained(args.base_model if adapter else path,torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='sdpa',local_files_only=True)
        if adapter:model=PeftModel.from_pretrained(model,path,is_trainable=True,local_files_only=True)
        sync_model_special_tokens(model,tokenizer)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if adapter:model.enable_input_require_grads()
        parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        before={n:p.detach().cpu() for n,p in parameters}
        scale=2. if adapter else None
        if adapter:
            config=json.loads((path/'adapter_config.json').read_text());scale=config['lora_alpha']/config['r']
        initial,_=evaluate(model,tokenizer,probes,prompts,choices,labels,reference,args.output_dir,name,0)
        dump(args.output_dir/(name+'.before.json'),initial)
        for batch_index in range(args.batches):
            if (name,batch_index) in completed:continue
            pair_started=time.monotonic()
            batch_ids=indices[batch_index*args.accumulation:(batch_index+1)*args.accumulation]
            batch=[encoded[i] for i in batch_ids];trial_data={}
            for objective in ('ce','steer'):
                dump(args.output_dir/'status.json',{'phase':'diagnostic','model':name,'batch':batch_index,'objective':objective})
                print(f'{name} batch {batch_index+1}/{args.batches} {objective}: same checkpoint, fresh moments',flush=True)
                record,gradient,update,effective=objective_step(model,parameters,before,batch,objective,args.seed+batch_index,
                                                               args.learning_rate,adapter,scale)
                post,_=evaluate(model,tokenizer,probes,prompts,choices,labels,reference,args.output_dir,
                                name+'-'+objective,batch_index+1)
                record['probe_after']=post
                record['probe_delta']={'dev_nll':post['samsum']['model_target_nll_nats']-initial['samsum']['model_target_nll_nats'],
                    'base_confident_probability':post['base_confident_probability_delta']-initial['base_confident_probability_delta'],
                    'boolq_chat_yes_no':post['boolq']['chat']['mean_yes_no_logodds']-initial['boolq']['chat']['mean_yes_no_logodds'],
                    'boolq_chat_gold_no_yes_no':post['boolq']['chat']['gold_no_logodds_mean']-initial['boolq']['chat']['gold_no_logodds_mean'],
                    'boolq_chat_accuracy':post['boolq']['chat']['accuracy']-initial['boolq']['chat']['accuracy']}
                trial_data[objective]=(record,gradient,update,effective)
            ce,cg,cu,cf=trial_data['ce'];st,sg,su,sf=trial_data['steer']
            cgn,sgn=ce['gradient_norm_after_clip']**2,st['gradient_norm_after_clip']**2
            cun,sun=ce['native_update_l2']**2,st['native_update_l2']**2
            cfn,sfn=ce['effective_common_update_l2']**2,st['effective_common_update_l2']**2
            cross_effective=effective_layer_dots(cf,sf,adapter)
            cross_gradient=layer_dots(cg,sg);cross_update=layer_dots(cu,su)
            paired={'model':name,'batch':batch_index,'train_indices':batch_ids,
                    'gradient_cosine_post_clip':cosine(sum(cross_gradient.values()),cgn,sgn),
                    'native_adamw_update_cosine':cosine(sum(cross_update.values()),cun,sun),
                    'effective_common_update_cosine':cosine(sum(cross_effective.values()),cfn,sfn),
                    'layer_effective_update_cosine':{k:cosine(v,ce['layer_common_update_norm'][k]**2,
                        st['layer_common_update_norm'][k]**2) for k,v in cross_effective.items()},
                    'layer_native_gradient_cosine':{k:cosine(v,ce['layer_native_gradient_l2'][k]**2,
                        st['layer_native_gradient_l2'][k]**2) for k,v in cross_gradient.items()},
                    'layer_native_update_cosine':{k:cosine(v,ce['layer_native_update_l2'][k]**2,
                        st['layer_native_update_l2'][k]**2) for k,v in cross_update.items()},
                    'pair_elapsed_seconds':time.monotonic()-pair_started,
                    'completed_utc':datetime.now(timezone.utc).isoformat(),
                    'trials':{'ce':ce,'steer':st}}
            results.append(paired);dump(args.output_dir/'results.json',results)
            print(f'{name} batch {batch_index+1}: gradient cosine={paired["gradient_cosine_post_clip"]}; actual common-weight update cosine={paired["effective_common_update_cosine"]}',flush=True)
            del trial_data,cg,sg,cu,su,cf,sf,gradient,update,effective
            gc.collect()
        restore(parameters,before);model.zero_grad(set_to_none=True)
        del model,parameters,before;gc.collect();torch.cuda.empty_cache()
    lines=['# Existing checkpoint local update diagnostic','',manifest['optimizer_scope'],'',
           'Original checkpoints are restored for every trial. Same batch and dropout RNG for CE/Steer. Full benchmark scores are reused separately in part1; these probe results are not full benchmark reruns.','',
           '| Checkpoint | Batch | Gradient cosine | Native AdamW update cosine | Effective common-weight update cosine |','|---|---:|---:|---:|---:|']
    for r in results:
        values=[r[k] for k in ('gradient_cosine_post_clip','native_adamw_update_cosine','effective_common_update_cosine')]
        lines.append(f'| {r["model"]} | {r["batch"]} | '+' | '.join('undefined' if v is None else f'{v:.6f}' for v in values)+' |')
    lines+=['','Full/native and LoRA/native coordinate systems differ. Compare CE with Steer within each checkpoint; effective-weight coordinates cover all28 layers of the shared projections. Reported optimizer sensitivity uses reset moments and diagnostic LR1e-5, not the original end-of-training scheduler or historical trajectory. BF16 rounding and 8-bit states are retained.','']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines))
    dump(args.output_dir/'status.json',{'phase':'complete','trials':len(results)*2})
    print('Completed checkpoint update diagnostics',flush=True)


if __name__=='__main__':main()
