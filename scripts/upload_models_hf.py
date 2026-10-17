"""Upload trained LoRA adapters under artifacts/ to the Hugging Face Hub.

Each adapter directory (one with adapter_config.json) becomes its own repo:
    <namespace>/<prefix>-<run>-<method>
The adapters were trained against a local copy of Qwen2-7B, so their
adapter_config.json and README.md point at a /data/... path. Those two files
are rewritten to Qwen/Qwen2-7B on the fly; the files on disk are untouched.

Auth: export HF_TOKEN=... (write scope) or run `hf auth login` first.

Examples:
    python scripts/upload_models_hf.py --namespace USER --dry-run
    python scripts/upload_models_hf.py --namespace USER --only fixed-margin
    python scripts/upload_models_hf.py --namespace USER --only steer_lora --public
"""
import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from methods.qwen2_base import QWEN2_BASE_MODEL_ID


def find_adapters(root):
    return sorted(p.parent for p in root.rglob('adapter_config.json'))


def repo_name(adapter_dir,root,prefix):
    # artifacts/<task>/<run>/models/<method> -> <prefix>-<run>-<method>
    parts=[p for p in adapter_dir.relative_to(root).parts if p!='models']
    name='-'.join([prefix,*parts[1:]]) if len(parts)>1 else f'{prefix}-{parts[0]}'
    return re.sub(r'[^A-Za-z0-9._-]+','-',name).strip('-.')[:96]


def patched_bytes(path):
    text=path.read_text()
    if path.name=='adapter_config.json':
        cfg=json.loads(text)
        cfg['base_model_name_or_path']=QWEN2_BASE_MODEL_ID
        return (json.dumps(cfg,indent=2)+'\n').encode()
    # Base path was saved both absolute and relative to ROOT.
    text=text.replace(f'{ROOT}/artifacts/base/Qwen2-7B',QWEN2_BASE_MODEL_ID)
    return text.replace('artifacts/base/Qwen2-7B',QWEN2_BASE_MODEL_ID).encode()


def build_operations(adapter_dir):
    ops=[]
    for f in sorted(adapter_dir.iterdir()):
        if not f.is_file():
            continue
        src=patched_bytes(f) if f.name in ('adapter_config.json','README.md') else str(f)
        ops.append(CommitOperationAdd(path_in_repo=f.name,path_or_fileobj=src))
    return ops


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--namespace',required=True,help='HF username or org')
    ap.add_argument('--prefix',default='sdft-qwen2-7b')
    ap.add_argument('--artifacts',type=Path,default=ROOT/'artifacts')
    ap.add_argument('--only',nargs='+',default=[],help='upload only adapters whose path contains any of these substrings')
    ap.add_argument('--public',action='store_true',help='create repos as public (default: private)')
    ap.add_argument('--dry-run',action='store_true',help='list what would be uploaded, upload nothing')
    args=ap.parse_args()

    adapters=[d for d in find_adapters(args.artifacts) if not args.only or any(s in str(d) for s in args.only)]
    if not adapters:
        sys.exit('No adapter directories matched.')

    api=HfApi()
    if not args.dry_run:
        print(f'Logged in as {api.whoami()["name"]}')

    for d in adapters:
        repo_id=f'{args.namespace}/{repo_name(d,args.artifacts,args.prefix)}'
        size=sum(f.stat().st_size for f in d.iterdir() if f.is_file())/1e9
        print(f'{d.relative_to(args.artifacts)}  ->  {repo_id}  ({size:.2f} GB)')
        if args.dry_run:
            continue
        api.create_repo(repo_id,private=not args.public,exist_ok=True)
        commit=api.create_commit(
            repo_id,
            operations=build_operations(d),
            commit_message=f'Upload {d.relative_to(args.artifacts)}',
        )
        print(f'  done: {commit.commit_url}')


if __name__=='__main__':
    main()
