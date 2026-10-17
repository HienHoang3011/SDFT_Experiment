"""Matched causal controls for the existing Steer objective, with fixed margin.

All arms start from Base, share LoRA initialization and token/order/schedule.
Historical Base/Full/LoRA/Steer/FAPM results are reused as context, not rerun.
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
from types import SimpleNamespace

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
import transformers
import peft
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eval'))
from audit_margin_artifacts import load_loss
from gpu_probe_utils import samsum_forward, boolq_forward, log
from distribution_metrics import token_metrics, summarize
from eval_samsum import extract_summary
from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens

ARMS=('ce','steer','ce_half','steer_floor','ce_anchor','steer_anchor')


def selected_logits(outputs,labels):
    targets=labels[:,1:].reshape(-1)
    active=targets.ne(-100)
    logits=outputs.logits[:,:-1].reshape(-1,outputs.logits.size(-1))[active].float()
    return logits,targets[active]


def control_loss(outputs,labels,denominator,arm,reference=None,x_factor=1.5,floor=.1):
    logits,targets=selected_logits(outputs,labels)
    ce=F.cross_entropy(logits,targets,reduction='none')
    original=load_loss()(x_factor)(outputs,labels,denominator)
    with torch.no_grad():
        rows=torch.arange(len(targets),device=logits.device)
        other=logits.detach().clone();other[rows,targets]=-torch.inf
        boost=(other.max(-1).values+math.log1p(x_factor)-logits.detach()[rows,targets]).clamp_min(0)
        py=logits.detach().softmax(-1)[rows,targets]
        scale=1-1/(1+boost.expm1()*py)
    if arm.startswith('ce'):
        loss=ce.sum()/denominator*(.5 if arm=='ce_half' else 1.)
    else:
        loss=original
        if arm=='steer_floor':
            loss=loss+((floor-scale).clamp_min(0).detach()*ce).sum()/denominator
    kl=logits.new_zeros(())
    if arm.endswith('_anchor'):
        if reference is None or reference.shape!=logits.shape:raise ValueError('Frozen Base reference required')
        kl=F.kl_div(logits.log_softmax(-1),reference,log_target=True,reduction='sum')
        loss=loss+kl/denominator
    metrics={'ce_sum':ce.detach().sum().item(),'steer_loss_sum':original.detach().item()*denominator,
             'active_sum':int((boost>0).sum()),'ce_weight_sum':scale.sum().item(),
             'floor_applied_sum':int((scale<floor).sum()) if arm=='steer_floor' else 0,
             'anchor_kl_sum':kl.detach().item(),'tokens':len(targets)}
    return loss,metrics


@torch.no_grad()
def teacher(model,ids,labels):
    model.eval()
    with model.disable_adapter():
        outputs=model(input_ids=ids,use_cache=False)
        logits,_=selected_logits(outputs,labels)
        reference=logits.log_softmax(-1).detach()
    model.train()
    return reference


def auc(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores)
    order=np.argsort(scores,kind='stable');ranks=np.empty(len(scores),float)
    i=0
    while i<len(scores):
        j=i+1
        while j<len(scores) and scores[order[j]]==scores[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    positives=int(labels.sum());negatives=len(labels)-positives
    return float((ranks[labels==1].sum()-positives*(positives+1)/2)/(positives*negatives))


def read_split(split):
    return ipc.open_stream(str(ROOT/f'data/samsum_emnlp2019/{split}/data-00000-of-00001.arrow')).read_all().to_pylist()


def encode(tokenizer,messages):
    enc=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=True,
        return_assistant_tokens_mask=True,add_generation_prompt=False)
    ids=torch.tensor([enc['input_ids'][:2048]])
    mask=torch.tensor([enc['assistant_masks'][:2048]],dtype=torch.bool)
    labels=ids.clone();labels[~mask]=-100
    return ids,labels


def make_probes(tokenizer,data,count,seed):
    indices=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed))[:count].tolist()
    probes=[]
    for i in indices:
        enc=tokenizer.apply_chat_template(data[i]['messages'],tokenize=True,return_dict=True,
            return_assistant_tokens_mask=True,add_generation_prompt=False)
        positions=[j-1 for j in range(1,len(enc['input_ids'])) if enc['assistant_masks'][j]]
        if len(positions)>32:
            positions=[positions[j] for j in torch.linspace(0,len(positions)-1,32).round().long().tolist()]
        if positions:probes.append((i,torch.tensor([enc['input_ids']]),positions))
    return probes


@torch.inference_mode()
def evaluate(model,tokenizer,probes,prompts,choice_ids,labels,reference,out,arm,step):
    model.eval();buckets={};lengths=[];confident=[];correct_delta=[]
    current=samsum_forward(model,probes)
    if reference is None:
        base_samsum=list(current);current=iter(base_samsum)
        reference={'samsum':base_samsum}
    for (_,ids,positions),p,q in zip(probes,reference['samsum'],current):
        metrics=token_metrics(p.to('cuda'),q.to('cuda'),ids[0,torch.tensor(positions)+1])
        for k,v in metrics.items():buckets.setdefault(k,[]).append(v.cpu())
        lengths.append(len(positions));mask=metrics['base_target_probability']>=.9
        confident.append((metrics['model_target_probability']-metrics['base_target_probability'])[mask].cpu())
        correct_delta.append((metrics['model_target_nll_nats']-metrics['base_target_nll_nats'])[mask].cpu())
    stats=summarize({k:torch.cat(v) for k,v in buckets.items()},lengths,42)
    result={'arm':arm,'step':step,'samsum':{k:v['mean'] for k,v in stats.items()},
            'samsum_ci':{k:v['example_bootstrap_ci95'] for k,v in stats.items()},
            'samsum_examples':[{'example_id':probe[0],'tokens':len(values),
                'target_nll_sum':values.sum().item()} for probe,values in zip(probes,buckets['model_target_nll_nats'])],
            'base_confident_tokens':sum(len(v) for v in confident),
            'base_confident_probability_delta':torch.cat(confident).mean().item(),
            'base_confident_nll_delta':torch.cat(correct_delta).mean().item(),'boolq':{}}
    for fmt in ('raw','chat'):
        scores=[]
        for distribution,_ in boolq_forward(model,prompts[fmt],tokenizer,4,choice_ids[fmt],False):
            scores.append(distribution[:,choice_ids[fmt]])
        scores=torch.cat(scores);predicted=scores.argmax(-1);logodds=(scores[:,1]-scores[:,0]).numpy()
        correct=predicted.eq(labels)
        result['boolq'][fmt]={'accuracy':correct.float().mean().item(),
            'predict_yes_fraction':predicted.float().mean().item(),'roc_auc':auc(labels.numpy(),logodds),
            'gold_no_accuracy':correct[labels==0].float().mean().item(),
            'gold_yes_accuracy':correct[labels==1].float().mean().item(),
            'mean_yes_no_logodds':float(logodds.mean()),
            'gold_no_logodds_mean':float(logodds[labels.numpy()==0].mean()),
            'gold_yes_logodds_mean':float(logodds[labels.numpy()==1].mean())}
        (out/f'{arm}.step{step}.boolq-{fmt}.json').write_text(json.dumps(
            {'choice_logprobs':scores.tolist(),'labels':labels.tolist()},indent=2)+'\n')
    model.train()
    return result,reference


@torch.inference_mode()
def rouge(model,tokenizer,data,indices):
    from rouge_score.rouge_scorer import RougeScorer
    model.eval();scorer=RougeScorer(['rougeL'],use_stemmer=True);rows=[]
    for start in range(0,len(indices),4):
        selected=indices[start:start+4]
        prompts=[tokenizer.apply_chat_template([{'role':'user','content':data[i]['prompt']}],tokenize=False,add_generation_prompt=True) for i in selected]
        inputs=tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
        generated=model.generate(**inputs,do_sample=False,max_new_tokens=128,use_cache=True)
        texts=tokenizer.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        for i,text in zip(selected,texts):
            prediction=extract_summary(text);score=scorer.score(data[i]['summary'],prediction)['rougeL'].fmeasure
            rows.append({'test_index':i,'prediction':prediction,'rougeL':score})
    model.train()
    return {'rougeL':sum(r['rougeL'] for r in rows)/len(rows),'examples':len(rows),'predictions':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--base-model',type=Path,default=ROOT/'artifacts/base/Qwen2-7B')
    parser.add_argument('--reference-root',type=Path,default=ROOT/'artifacts/samsum/fair-methods-2epoch-log25-tmux')
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--accumulation',type=int,default=64)
    parser.add_argument('--probe-every',type=int,default=32)
    parser.add_argument('--dev-samples',type=int,default=64)
    parser.add_argument('--boolq-samples',type=int,default=512)
    parser.add_argument('--rouge-samples',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--min-free-gib',type=float,default=34)
    parser.add_argument('--no-save-adapters',action='store_true')
    args=parser.parse_args()
    if (args.output_dir/'run_manifest.json').exists():raise FileExistsError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    while torch.cuda.mem_get_info()[0]<args.min_free_gib*2**30:
        log('Waiting for GPU memory before starting the matched controls');time.sleep(20)
    tokenizer=AutoTokenizer.from_pretrained(args.base_model,local_files_only=True)
    configure_qwen2_base_tokenizer(tokenizer,str(args.base_model));tokenizer.padding_side='left'
    train,dev,test=read_split('train'),read_split('dev'),read_split('test')
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed)).tolist()
    needed=args.steps*args.accumulation
    if needed>len(order):order=(order*math.ceil(needed/len(order)))[:needed]
    else:order=order[:needed]
    encoded={i:encode(tokenizer,train[i]['messages']) for i in set(order)}
    probes=make_probes(tokenizer,dev,args.dev_samples,args.seed)
    docs=[json.loads(l) for l in (args.reference_root/'reports/margin-mechanism/boolq.validation.jsonl').read_text().splitlines()]
    selected=torch.randperm(len(docs),generator=torch.Generator().manual_seed(args.seed))[:args.boolq_samples].tolist()
    docs=[docs[i] for i in selected];labels=torch.tensor([d['label'] for d in docs])
    raw=[f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:" for d in docs]
    prompts={'raw':raw,'chat':[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in raw]}
    choices={}
    for fmt in prompts:
        ids=[tokenizer.encode((' ' if fmt=='raw' else '')+s,add_special_tokens=False) for s in ('no','yes')]
        assert all(len(i)==1 for i in ids);choices[fmt]=torch.tensor([i[0] for i in ids])
    test_indices=torch.randperm(len(test),generator=torch.Generator().manual_seed(args.seed))[:args.rouge_samples].tolist()
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'steps':args.steps,'effective_batch_size':args.accumulation,'seed':args.seed,'arms':args.arms,
        'fixed_x_factor':1.5,'fixed_margin':math.log(2.5),'ce_half_scale':.5,'ce_gradient_floor':.1,'base_kl_weight':1.,
        'optimizer':'AdamW8bit','learning_rate':1e-5,'weight_decay':.1,'max_grad_norm':1.,
        'scheduler':'Prefix of original 458-step cosine schedule, warmup 14 steps',
        'lora':{'rank':256,'alpha':512,'dropout':.3},'training_order':order,
        'training_order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        'dev_indices':[p[0] for p in probes],'boolq_doc_ids':[d['doc_id'] for d in docs],'test_indices':test_indices,
        'historical_artifact_root':str(args.reference_root),
        'note':'Matched shorter-run causal controls, one seed. Historical two-epoch results are context, not equal-budget comparisons.',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reference=None;trajectories=[]
    for arm in args.arms:
        (args.output_dir/'status.json').write_text(json.dumps({'phase':'training','arm':arm,'step':0})+'\n')
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        model=AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='sdpa',local_files_only=True)
        config=LoraConfig(r=256,lora_alpha=512,lora_dropout=.3,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM',bias='none')
        model=get_peft_model(model,config);sync_model_special_tokens(model,tokenizer)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=bnb.optim.AdamW8bit(parameters,lr=1e-5,weight_decay=.1)
        scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=14,num_training_steps=458)
        initial,reference=evaluate(model,tokenizer,probes,prompts,choices,labels,reference,args.output_dir,arm,0)
        trajectories.append(initial)
        with (args.output_dir/f'{arm}.train.jsonl').open('w') as handle:
            for step in range(1,args.steps+1):
                started=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
                indices=order[(step-1)*args.accumulation:step*args.accumulation]
                denominator=sum(int(encoded[i][1][:,1:].ne(-100).sum()) for i in indices)
                totals={};loss_total=0.
                for i in indices:
                    ids,target=encoded[i];ids=ids.to('cuda');target=target.to('cuda')
                    ref=teacher(model,ids,target) if arm.endswith('_anchor') else None
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        outputs=model(input_ids=ids,use_cache=False)
                        loss,metrics=control_loss(outputs,target,denominator,arm,ref)
                    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss: {arm}, step {step}')
                    loss.backward();loss_total+=loss.detach().item()
                    for k,v in metrics.items():totals[k]=totals.get(k,0)+v
                    del outputs,loss,ref
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.).item()
                if not math.isfinite(norm):raise FloatingPointError(f'Nonfinite gradients: {arm}, step {step}')
                # Measure actual AdamW parameter motion at the same probe steps.
                record_motion=(step%args.probe_every==0 or step==args.steps)
                before=[p.detach().clone() for p in parameters] if record_motion else None
                lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                motion=None
                if before is not None:
                    motion=math.sqrt(sum((p.detach().float()-old.float()).square().sum().item() for p,old in zip(parameters,before)))
                    del before
                record={'step':step,'loss':loss_total,'ce':totals['ce_sum']/denominator,
                    'self_steer_kl':totals['steer_loss_sum']/denominator,
                    'steer_active_fraction':totals['active_sum']/denominator,
                    'mean_steer_CE_weight':totals['ce_weight_sum']/denominator,
                    'floor_applied_fraction':totals['floor_applied_sum']/denominator,
                    'base_anchor_kl':totals['anchor_kl_sum']/denominator,
                    'grad_norm_before_clip':norm,'clipping_factor':min(1.,1./max(norm,1e-30)),
                    'learning_rate':lr,'actual_adapter_update_l2':motion,'assistant_tokens':denominator,
                    'elapsed_seconds':time.monotonic()-started}
                handle.write(json.dumps(record)+'\n');handle.flush()
                (args.output_dir/'status.json').write_text(json.dumps({'phase':'training','arm':arm,'step':step,'steps':args.steps})+'\n')
                log(f'{arm} step {step}/{args.steps}: CE={record["ce"]:.4f}; steer KL={record["self_steer_kl"]:.4f}; grad={norm:.4f}; {record["elapsed_seconds"]:.1f}s')
                if record_motion:
                    result,reference=evaluate(model,tokenizer,probes,prompts,choices,labels,reference,args.output_dir,arm,step)
                    trajectories.append(result)
                    (args.output_dir/'trajectories.json').write_text(json.dumps(trajectories,indent=2)+'\n')
                    log(f'{arm} probe step {step}: BoolQ chat={100*result["boolq"]["chat"]["accuracy"]:.2f}%; yes={100*result["boolq"]["chat"]["predict_yes_fraction"]:.2f}%; AUC={result["boolq"]["chat"]["roc_auc"]:.4f}')
        quality=rouge(model,tokenizer,test,test_indices)
        (args.output_dir/f'{arm}.rouge.json').write_text(json.dumps(quality,indent=2)+'\n')
        if not args.no_save_adapters:
            model.save_pretrained(args.output_dir/'models'/arm);tokenizer.save_pretrained(args.output_dir/'models'/arm)
        del optimizer,scheduler,model,parameters
        gc.collect();torch.cuda.empty_cache()
    lines=['# Fixed-margin causal controls','',manifest['note'],'',
        'All arms use the same Base, LoRA initialization, examples/order, assistant masking, dropout, AdamW8bit and learning-rate schedule. Margin stays log(2.5).','',
        '| Arm | SAMSum dev NLL | Base-confident Δ P(y) | ChatML BoolQ % | ChatML yes % | ChatML AUC | Raw BoolQ % | SAMSum test-subset ROUGE-L % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in args.arms:
        r=next(r for r in reversed(trajectories) if r['arm']==arm)
        q=json.loads((args.output_dir/f'{arm}.rouge.json').read_text());b=r['boolq']
        lines.append(f"| {arm} | {r['samsum']['model_target_nll_nats']:.4f} | {r['base_confident_probability_delta']:+.4f} | {100*b['chat']['accuracy']:.2f} | {100*b['chat']['predict_yes_fraction']:.2f} | {b['chat']['roc_auc']:.4f} | {100*b['raw']['accuracy']:.2f} | {100*q['rougeL']:.2f} |")
    # Pair examples within the new matched run, preserving identical sample IDs.
    generator=np.random.default_rng(args.seed)
    def accuracy_vector(arm,fmt='chat'):
        payload=json.loads((args.output_dir/f'{arm}.step{args.steps}.boolq-{fmt}.json').read_text())
        assert payload['labels']==labels.tolist()
        return (np.asarray(payload['choice_logprobs']).argmax(-1)==np.asarray(payload['labels'])).astype(float)
    def ci(values):
        draws=values[generator.integers(len(values),size=(2000,len(values)))].mean(axis=1)
        return [float(v) for v in np.quantile(draws,[.025,.975])]
    contrasts=[]
    for first,second in [('steer','ce'),('ce_half','ce'),('steer_floor','steer'),
                         ('ce_anchor','ce'),('steer_anchor','steer')]:
        if first in args.arms and second in args.arms:
            values=accuracy_vector(first)-accuracy_vector(second)
            contrasts.append({'contrast':first+' minus '+second,'boolq_chat_delta_pp':100*values.mean(),
                              'paired_ci95_pp':[100*v for v in ci(values)]})
    if all(arm in args.arms for arm in ('ce','steer','ce_anchor','steer_anchor')):
        values=(accuracy_vector('steer_anchor')-accuracy_vector('steer'))-(accuracy_vector('ce_anchor')-accuracy_vector('ce'))
        contrasts.append({'contrast':'Anchor gain on Steer minus anchor gain on CE',
            'boolq_chat_delta_pp':100*values.mean(),'paired_ci95_pp':[100*v for v in ci(values)]})
    (args.output_dir/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    lines+=['','## Paired contrasts on the same BoolQ examples','',
            '| Contrast | ChatML accuracy delta (pp) | Paired example-bootstrap CI 95% |','|---|---:|---|']
    for c in contrasts:
        lo,hi=c['paired_ci95_pp'];lines.append(f"| {c['contrast']} | {c['boolq_chat_delta_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['','Intervals describe probe sampling, not variation across training seeds.','',
            '## Historical two-epoch artifacts: reused context, different training budget','',
            '| Model | SAMSum ROUGE-L % | IFEval % | BoolQ benchmark % | BBH % |','|---|---:|---:|---:|---:|']
    historical=json.loads((args.reference_root/'stable-five/summary.json').read_text())
    scores={(r['model'],r['task']):r['score'] for r in historical['rows']}
    for model_name in ('base','full_sft','lora','steer_lora','steer_full','fapm'):
        quality=json.loads((args.reference_root/'reports'/model_name/'downstream_results.json').read_text())
        lines.append(f"| {model_name} | {100*quality['rougeL']:.2f} | {100*scores[model_name,'ifeval']:.2f} | {100*scores[model_name,'boolq']:.2f} | {100*scores[model_name,'bbh_cot_zeroshot']:.2f} |")
    lines+=['','Interpretation was specified in EXPERIMENT_DESIGN.md. These are subset diagnostics, not full benchmark reruns. Historical Base/Full/LoRA/Steer/FAPM artifacts are retained unchanged.','']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for arm in args.arms:
        points=[r for r in trajectories if r['arm']==arm]
        steps=[r['step'] for r in points]
        for ax,values,title in [(axes[0,0],[r['samsum']['model_target_nll_nats'] for r in points],'SAMSum dev target NLL'),
            (axes[0,1],[r['base_confident_probability_delta'] for r in points],'Delta P(y): confident Base tokens'),
            (axes[1,0],[100*r['boolq']['chat']['predict_yes_fraction'] for r in points],'ChatML predicted yes (%)'),
            (axes[1,1],[r['boolq']['chat']['roc_auc'] for r in points],'ChatML yes/no ROC AUC')]:
            ax.plot(steps,values,marker='o',label=arm);ax.set_title(title)
    for ax in axes.flat:ax.set_xlabel('Optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(args.output_dir/'trajectories.png',dpi=160);plt.close(fig)
    (args.output_dir/'status.json').write_text(json.dumps({'phase':'complete','arms':args.arms})+'\n')
    log('Completed fixed-margin controls: '+str(args.output_dir/'REPORT.md'))


if __name__=='__main__':main()
