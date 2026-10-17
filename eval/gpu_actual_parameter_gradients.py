"""Compare actual CE/Steer gradients on full weights and LoRA parameters.

Uses the unchanged SteeredLoss, assistant masking and training dropout. Margin
variants share one saved training forward. Probe gradient dot products measure
first-order SGD sensitivity, not an AdamW step or a training trajectory.
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

import torch
import torch.nn.functional as F
import pyarrow.ipc as ipc
import peft
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from audit_margin_artifacts import load_loss
from gpu_probe_utils import log
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods'))
from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens


def read_split(split):
    return ipc.open_stream(f'data/samsum_emnlp2019/{split}/data-00000-of-00001.arrow').read_all().to_pylist()


def encode(tokenizer, messages):
    enc=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=True,
        return_assistant_tokens_mask=True,add_generation_prompt=False)
    ids=torch.tensor([enc['input_ids'][:2048]],device='cuda')
    mask=torch.tensor([enc['assistant_masks'][:2048]],device='cuda',dtype=torch.bool)
    labels=ids.clone();labels[~mask]=-100
    return ids,labels


def cpu_gradients(loss, parameters, retain_graph=False):
    grads=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    result=[None if g is None else g.detach().cpu() for g in grads]
    del grads
    return result


def inner(first,second):
    # Dot products/norms accumulate in fp64; chunks keep GPU overhead small.
    total=torch.zeros((),device='cuda',dtype=torch.float64)
    for a,b in zip(first,second):
        if a is None or b is None:continue
        av,bv=a.reshape(-1),b.reshape(-1)
        for start in range(0,av.numel(),1048576):
            ac=av[start:start+1048576].to('cuda',dtype=torch.float32)
            bc=bv[start:start+1048576].to('cuda',dtype=torch.float32)
            total+=(ac.double()*bc.double()).sum()
    return total.item()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,default=Path('artifacts/samsum/fair-methods-2epoch-log25-tmux'))
    parser.add_argument('--base-model',type=Path,default=Path('artifacts/base/Qwen2-7B'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--models',nargs='+',default=['lora','steer_lora','full_sft','steer_full'])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-examples',type=int,default=2)
    parser.add_argument('--probe-examples',type=int,default=2)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(args.base_model,local_files_only=True)
    configure_qwen2_base_tokenizer(tokenizer,str(args.base_model))
    train=read_split('train');dev=read_split('dev')
    train_indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed))[:args.train_examples].tolist()
    batches=[encode(tokenizer,train[i]['messages']) for i in train_indices]
    denominator=sum(int(labels[:,1:].ne(-100).sum()) for _,labels in batches)
    confident={}
    with (args.artifact_root/'reports/margin-mechanism/samsum/base.tokens.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['metrics']['base_target_probability']>=.9:
                confident.setdefault(row['example_id'],[]).append((row['prediction_position'],row['target_token_id']))
    dev_indices=torch.randperm(len(dev),generator=torch.Generator().manual_seed(args.seed)).tolist()
    dev_indices=[i for i in dev_indices if i in confident][:args.probe_examples]
    dev_batches=[]
    for i in dev_indices:
        ids,_=encode(tokenizer,dev[i]['messages'])
        selected=[(pos,target) for pos,target in confident[i] if pos<len(ids[0])-1]
        dev_batches.append((ids,torch.tensor([pos for pos,_ in selected],device='cuda'),
                            torch.tensor([target for _,target in selected],device='cuda')))
    docs={d['doc_id']:d for d in [json.loads(l) for l in (args.artifact_root/'reports/margin-mechanism/boolq.validation.jsonl').read_text().splitlines()]}
    bool_ids=[]
    with (args.artifact_root/'reports/margin-mechanism/boolq-chat/base.examples.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['label']==0 and row['base_prediction']==0:bool_ids.append(row['doc_id'])
    order=torch.randperm(len(bool_ids),generator=torch.Generator().manual_seed(args.seed))[:args.probe_examples].tolist()
    bool_ids=[bool_ids[i] for i in order]
    bool_inputs=[]
    for doc_id in bool_ids:
        d=docs[doc_id]
        prompt=f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:"
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        bool_inputs.append(tokenizer(text,add_special_tokens=False,return_tensors='pt').to('cuda'))
    no_yes=[tokenizer.encode(t,add_special_tokens=False) for t in ('no','yes')]
    assert all(len(t)==1 for t in no_yes)
    choice_ids=torch.tensor([t[0] for t in no_yes],device='cuda')
    runtime={'started_utc':datetime.now(timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'seed':args.seed,'train_examples':train_indices,'train_tokens':denominator,
        'train_lengths':[len(ids[0]) for ids,_ in batches],'dev_examples':dev_indices,
        'base_confident_dev_tokens':sum(len(t) for _,_,t in dev_batches),'boolq_gold_no_base_correct_doc_ids':bool_ids,
        'gradient_scope':'Actual trainable LoRA/full-model parameters. SteeredLoss unchanged; dropout on for train, eval for probes.',
        'sensitivity_scope':'Gradient inner products give first-order SGD response. No optimizer step, no checkpoint update, no benchmark rerun.',
        'loss_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'methods/train_steer.py').read_bytes()).hexdigest()}
    (args.output_dir/'run_manifest.json').write_text(json.dumps(runtime,indent=2)+'\n')
    log(f"Actual-parameter analysis: {denominator} assistant train tokens; {runtime['base_confident_dev_tokens']} Base-confident dev tokens; {len(bool_ids)} BoolQ gold-no probes.")
    results=[]
    for name in args.models:
        start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        path=args.artifact_root/'models'/name
        adapter=(path/'adapter_config.json').exists()
        model=AutoModelForCausalLM.from_pretrained(args.base_model if adapter else path,
            torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
        if adapter:model=PeftModel.from_pretrained(model,path,is_trainable=True,local_files_only=True)
        sync_model_special_tokens(model,tokenizer)
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if adapter:assert all('lora_' in n for n,_ in named)
        parameters=[p for _,p in named]
        log(f'{name}: {sum(p.numel() for p in parameters):,} actual trainable parameters; collecting probe gradients')
        # Nonreentrant checkpointing allows autograd.grad and recomputes the
        # same saved dropout mask during backward, reducing activation memory.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if adapter:model.enable_input_require_grads()
        model.eval()
        dev_terms=[]
        for ids,positions,targets in dev_batches:
            out=model(input_ids=ids,logits_to_keep=positions,use_cache=False)
            dev_terms.append(F.cross_entropy(out.logits[0].float(),targets,reduction='sum'))
        dev_nll=sum(dev_terms)/sum(len(t) for _,_,t in dev_batches)
        dev_value=dev_nll.item();dev_grad=cpu_gradients(dev_nll,parameters)
        del dev_terms,dev_nll,out
        terms=[]
        for inputs in bool_inputs:
            out=model(**inputs,logits_to_keep=1,use_cache=False)
            scores=out.logits[0,-1,choice_ids].float()
            terms.append(scores[1]-scores[0])
        yes_no=sum(terms)/len(terms)
        yes_no_value=yes_no.item();yes_no_grad=cpu_gradients(yes_no,parameters)
        del terms,yes_no,out,scores
        dev_norm=math.sqrt(max(inner(dev_grad,dev_grad),0))
        bool_norm=math.sqrt(max(inner(yes_no_grad,yes_no_grad),0))
        log(f'{name}: probe gradients collected; one shared training forward for CE and all margins')
        model.train();torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        outputs=[model(input_ids=ids,use_cache=False) for ids,_ in batches]
        ce_loss=sum(F.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.size(-1)).float(),
                    labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum') for out,(_,labels) in zip(outputs,batches))/denominator
        ce_value=ce_loss.item()
        ce_grad=cpu_gradients(ce_loss,parameters,retain_graph=True)
        ce_norm=math.sqrt(max(inner(ce_grad,ce_grad),0))
        objectives=[]
        for x in [None,0.,.5,1.5,3.,9.]:
            if x is None:
                gradient=ce_grad;loss_value=ce_value;title='CE';active=None;weight=None
            else:
                loss_fn=load_loss()(x)
                loss=sum(loss_fn(out,labels,denominator) for out,(_,labels) in zip(outputs,batches))
                loss_value=loss.item();title=f'Steer_x{x:g}'
                gradient=cpu_gradients(loss,parameters,retain_graph=True)
                del loss
                scales=[];activation=[]
                with torch.no_grad():
                    for out,(_,labels) in zip(outputs,batches):
                        logits=out.logits[:,:-1].reshape(-1,out.logits.size(-1)).float()
                        targets=labels[:,1:].reshape(-1);mask=targets.ne(-100)
                        logits=logits[mask];targets=targets[mask]
                        row=torch.arange(len(targets),device='cuda')
                        other=logits.clone();other[row,targets]=-torch.inf
                        boost=(other.max(-1).values+math.log1p(x)-logits[row,targets]).clamp_min(0)
                        p_y=logits.softmax(-1)[row,targets]
                        scales.append(1-1/(1+boost.expm1()*p_y));activation.append((boost>0).float())
                active=torch.cat(activation).mean().item();weight=torch.cat(scales).mean().item()
            norm=math.sqrt(max(inner(gradient,gradient),0))
            dot_ce=inner(gradient,ce_grad);dot_dev=inner(gradient,dev_grad);dot_bool=inner(gradient,yes_no_grad)
            row={'objective':title,'loss':loss_value,'parameter_gradient_norm':norm,
                'norm_ratio_to_CE':norm/max(ce_norm,1e-30),'cosine_to_CE':dot_ce/max(norm*ce_norm,1e-30),
                'active_fraction':active,'mean_token_CE_weight':weight,
                'cosine_to_base_confident_dev_NLL_gradient':dot_dev/max(norm*dev_norm,1e-30),
                'cosine_to_boolq_yes_minus_no_gradient':dot_bool/max(norm*bool_norm,1e-30),
                'first_order_dev_NLL_delta_per_unit_gradient_step':-dot_dev/max(norm,1e-30),
                'first_order_yes_no_delta_per_unit_gradient_step':-dot_bool/max(norm,1e-30),
                'first_order_dev_NLL_delta_at_SGD_lr_1e-5':-1e-5*dot_dev,
                'first_order_yes_no_delta_at_SGD_lr_1e-5':-1e-5*dot_bool}
            objectives.append(row)
            log(f'{name} {title}: cosine CE={row["cosine_to_CE"]:.4f}; norm/CE={row["norm_ratio_to_CE"]:.4f}; dev NLL sensitivity={row["first_order_dev_NLL_delta_per_unit_gradient_step"]:+.5f}; yes-no sensitivity={row["first_order_yes_no_delta_per_unit_gradient_step"]:+.5f}')
            if x is not None:del gradient
        result={'model':name,'parameter_mode':'LoRA' if adapter else 'full',
            'trainable_parameters':sum(p.numel() for p in parameters),'trainable_tensors':len(parameters),
            'dropout_train':True,'probe_eval':True,'dev_NLL_before':dev_value,'boolq_yes_minus_no_before':yes_no_value,
            'objectives':objectives,'elapsed_seconds':time.monotonic()-start,
            'peak_own_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
        results.append(result)
        (args.output_dir/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output_dir/'summary.json').write_text(json.dumps({'runtime':runtime,'rows':results},indent=2)+'\n')
        del ce_grad,dev_grad,yes_no_grad,outputs,ce_loss,model,parameters,named,out
        gc.collect();torch.cuda.empty_cache()
    lines=['# Gradient trên đúng tham số train của phương pháp','',
           runtime['gradient_scope'],'',runtime['sensitivity_scope'],'',
           f"Cùng {denominator} assistant tokens từ {len(batches)} train ví dụ; {runtime['base_confident_dev_tokens']} dev token Base P(y) ≥ 0.9; {len(bool_ids)} BoolQ gold-no mà Base đã đúng. Đây là probe cơ chế nhỏ, không đại diện toàn bộ dữ liệu.",'',
           '| Checkpoint | Objective | Cosine CE | Norm / CE | Active % | Δ dev NLL / unit grad step | Δ yes−no / unit grad step |',
           '|---|---|---:|---:|---:|---:|---:|']
    for result in results:
        for o in result['objectives']:
            active='—' if o['active_fraction'] is None else f"{100*o['active_fraction']:.2f}"
            lines.append(f"| {result['model']} | {o['objective']} | {o['cosine_to_CE']:.4f} | {o['norm_ratio_to_CE']:.4f} | {active} | {o['first_order_dev_NLL_delta_per_unit_gradient_step']:+.6f} | {o['first_order_yes_no_delta_per_unit_gradient_step']:+.6f} |")
    lines+=['','Cosine khác 1: margin đổi hướng gradient tổng trên tham số thực tế do tái trọng số token, dù gradient từng token cùng hướng CE. Δ dev NLL dương nghĩa là hướng cập nhật làm xấu các token Base tự tin ở bậc một. Δ yes−no dương nghĩa là đẩy sang yes trên các câu gold-no được chọn. Các hiệu ứng có thể khác dấu giữa checkpoint/margin.','',
            'Dropout LoRA bật lúc train như recipe; probe chạy eval. Gradient checkpointing có thể recompute layer forwards trong backward, nhưng mọi margin dùng cùng saved forward và dropout realization.','',
            'FAPM là hậu xử lý, không có training gradient riêng; Base là reference. Điểm của cả sáu phương pháp vẫn nằm trong artifact benchmark đã đo. Báo cáo này so sánh bốn phương pháp train, không gán giả một loss train cho FAPM.','',
            'Không chạy optimizer AdamW, không resume optimizer state và không đo trajectory sau nhiều bước. Inner-product sensitivity là SGD bậc một; không được coi là kết quả benchmark hay khẳng định nhân quả cho toàn bộ run. Muốn so chất lượng sau đổi margin phải train ablation.','']
    (args.output_dir/'report.md').write_text('\n'.join(lines))
    log('Completed actual parameter gradients: '+str((args.output_dir/'report.md').resolve()))


if __name__=='__main__':main()
