"""Exact streaming final displacement comparison for existing checkpoints."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from update_direction_metrics import gram_rows, cosine

METHODS = ('full_sft', 'lora', 'steer_lora', 'steer_full', 'fapm')
TARGETS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')


class TensorStore:
    def __init__(self, path):
        self.path = Path(path)
        index = self.path/'model.safetensors.index.json'
        if index.exists():
            self.index = json.loads(index.read_text())['weight_map']
        else:
            file = next(self.path.glob('*.safetensors'))
            with safe_open(file, framework='pt', device='cpu') as handle:
                self.index = {k:file.name for k in handle.keys()}
        self.handles = {}
        for file in set(self.index.values()):
            self.handles[file] = safe_open(self.path/file, framework='pt', device='cpu')

    def shape(self, key):
        return self.handles[self.index[key]].get_slice(key).get_shape()

    def slice(self, key, start, end, device):
        return self.handles[self.index[key]].get_slice(key)[start:end].to(device).float()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,default=Path('artifacts/samsum/fair-methods-2epoch-log25-tmux'))
    parser.add_argument('--base-model',type=Path,default=Path('artifacts/base/Qwen2-7B'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--chunk-elements',type=int,default=1048576)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    base=TensorStore(args.base_model)
    stores={n:TensorStore(args.artifact_root/'models'/n) for n in ('full_sft','steer_full','fapm')}
    adapters={n:load_file(str(args.artifact_root/'models'/n/'adapter_model.safetensors')) for n in ('lora','steer_lora')}
    configs={n:json.loads((args.artifact_root/'models'/n/'adapter_config.json').read_text()) for n in adapters}
    for config in configs.values():
        assert not config.get('use_rslora') and not config.get('fan_in_fan_out') and not config.get('use_dora')
    started=time.monotonic();total=torch.zeros(5,5,dtype=torch.float64)
    common=total.clone();layers={};tensor_rows=[];counts={n:0 for n in METHODS}
    elements=0;full_changed=fapm_changed=0;retained_exact=0
    for i,key in enumerate(sorted(base.index)):
        shape=base.shape(key);width=math.prod(shape[1:]) if len(shape)>1 else 1
        rows=max(1,args.chunk_elements//width)
        projection=key.endswith('.weight') and key.split('.')[-2] in TARGETS
        layer='layer.'+key.split('.')[2] if key.startswith('model.layers.') else 'other'
        local=torch.zeros_like(total)
        factors={}
        if projection:
            module=key[:-len('.weight')]
            for name in adapters:
                prefix='base_model.model.'+module
                a=adapters[name][prefix+'.lora_A.weight'].to(device).float()
                b=adapters[name][prefix+'.lora_B.weight'].to(device).float()
                factors[name]=(a,b,configs[name]['lora_alpha']/configs[name]['r'])
        for start in range(0,shape[0],rows):
            end=min(shape[0],start+rows)
            original=base.slice(key,start,end,device)
            vectors=[]
            for name in METHODS:
                if name in stores:
                    assert stores[name].shape(key)==shape,(name,key)
                    delta=stores[name].slice(key,start,end,device)-original
                elif projection:
                    a,b,scale=factors[name];delta=(b[start:end]@a)*scale
                else:
                    delta=torch.zeros_like(original)
                vectors.append(delta)
                counts[name]+=int(delta.ne(0).sum())
            local+=gram_rows(vectors)
            full_changed+=int(vectors[0].ne(0).sum());fapm_changed+=int(vectors[4].ne(0).sum())
            retained_exact+=int((vectors[4].ne(0)&vectors[4].eq(vectors[0])).sum())
            elements+=original.numel()
            del vectors,original,delta
        total+=local
        if projection:common+=local
        layers.setdefault(layer,torch.zeros_like(total)).add_(local)
        tensor_rows.append({'tensor':key,'shape':shape,'common_projection':projection,'gram':local.tolist()})
        del factors
        if i%20==0:
            print(f'{i+1}/{len(base.index)} tensors; {elements:,} parameters; {time.monotonic()-started:.1f}s',flush=True)
    def table(gram):
        return [[cosine(gram[i,j].item(),gram[i,i].item(),gram[j,j].item()) for j in range(5)] for i in range(5)]
    inputs=[]
    for path in [args.base_model]+[args.artifact_root/'models'/n for n in METHODS]:
        for file in sorted(path.glob('*.safetensors')):
            digest=hashlib.sha256()
            with file.open('rb') as handle:
                for chunk in iter(lambda:handle.read(8388608),b''):digest.update(chunk)
            inputs.append({'path':str(file.resolve()),'bytes':file.stat().st_size,'sha256':digest.hexdigest()})
    result={'methods':METHODS,'parameters':elements,'device':device,'elapsed_seconds':time.monotonic()-started,
            'whole_gram':total.tolist(),'common_gram':common.tolist(),'whole_cosine':table(total),'common_cosine':table(common),
            'norms':{n:math.sqrt(total[i,i].item()) for i,n in enumerate(METHODS)},
            'common_energy_fraction':{n:common[i,i].item()/total[i,i].item() if total[i,i]>0 else None for i,n in enumerate(METHODS)},
            'nonzero_counts':counts,'layers':{k:{'gram':g.tolist(),'cosine':table(g)} for k,g in layers.items()},
            'fapm':{'full_changed':full_changed,'fapm_changed':fapm_changed,'changed_equal_full':retained_exact},
            'input_files':inputs,'scope':'Net final displacement in effective weight space, not historical optimizer steps.'}
    (args.output_dir/'directions.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output_dir/'tensor_directions.json').write_text(json.dumps(tensor_rows,indent=2)+'\n')
    benchmarks=json.loads((args.artifact_root/'stable-five/summary.json').read_text())
    lines=['# Existing checkpoint update directions','','All values use the existing final checkpoints. LoRA deltas are alpha/r times B@A in the same full weight coordinates. No models were trained or benchmarked again.','','## Whole-space cosine','',
           '| Method | '+' | '.join(METHODS)+' |','|---|'+'---:|'*5]
    for i,name in enumerate(METHODS):lines.append('| '+name+' | '+' | '.join('undefined' if v is None else f'{v:.6f}' for v in result['whole_cosine'][i])+' |')
    lines+=['','## Norm and common-projection coverage','','| Method | Final displacement L2 | Common projection energy % | Changed parameters |','|---|---:|---:|---:|']
    for name in METHODS:
        coverage=result['common_energy_fraction'][name]
        lines.append(f'| {name} | {result["norms"][name]:.6f} | {100*coverage:.3f} | {counts[name]} |')
    lines+=['','## Existing full benchmark scores: reused','','| Model | Task | Score % |','|---|---|---:|']
    for row in benchmarks['rows']:lines.append(f'| {row["model"]} | {row["task"]} | {100*row["score"]:.3f} |')
    lines+=['','A final net displacement does not reconstruct the training path or establish a causal reason for a score gap. Whole-space and common-projection cosine have different scopes. Zero norm is reported as undefined. FAPM is postprocessing, not an optimizer-gradient method.','']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    matrix=np.asarray([[np.nan if v is None else v for v in row] for row in result['whole_cosine']])
    plot=axes[0].imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(plot,ax=axes[0])
    axes[0].set_xticks(range(5),METHODS,rotation=30,ha='right');axes[0].set_yticks(range(5),METHODS)
    axes[0].set_title('Final effective-weight displacement cosine')
    axes[1].bar(METHODS,[result['norms'][n] for n in METHODS]);axes[1].set_yscale('log')
    axes[1].tick_params(axis='x',rotation=30);axes[1].set_title('Final displacement L2 (log scale)')
    fig.savefig(args.output_dir/'directions.png',dpi=160);plt.close(fig)
    print('Completed checkpoint direction comparison',result['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
