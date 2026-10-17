"""Upload model directories to W&B as artifacts; optionally delete the local copy after verification.

With --dtype bfloat16 (default), floating-point safetensors shards are cast to bf16 in a staging
directory before upload, so an fp32 checkpoint is stored at bf16 size; config.json and the shard
index are updated to match. A local directory is deleted only with --delete, and only after the
logged artifact is committed and its manifest lists every staged file with the same size.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
import wandb
from safetensors.torch import load_file, save_file


def stage_bf16(source, staging):
    staging.mkdir(parents=True)
    total = 0
    for path in source.iterdir():
        if path.suffix == '.safetensors':
            tensors = {key: value.to(torch.bfloat16) if value.is_floating_point() else value
                       for key, value in load_file(str(path)).items()}
            save_file(tensors, str(staging / path.name), metadata={'format': 'pt'})
            total += sum(t.numel() * t.element_size() for t in tensors.values())
            del tensors
        elif path.is_file():
            shutil.copy2(path, staging / path.name)
    config = staging / 'config.json'
    if config.exists():
        data = json.loads(config.read_text())
        for key in ('torch_dtype', 'dtype'):
            if key in data:
                data[key] = 'bfloat16'
        config.write_text(json.dumps(data, indent=2) + '\n')
    index = staging / 'model.safetensors.index.json'
    if index.exists():
        data = json.loads(index.read_text())
        data.setdefault('metadata', {})['total_size'] = total
        index.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='sdft-experiment')
    parser.add_argument('--entity', default=None)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--model', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--dtype', choices=['bfloat16', 'original'], default='bfloat16')
    parser.add_argument('--staging-root', type=Path, default=Path('/root/sdft-fp32-runs/upload-staging'))
    parser.add_argument('--metadata', type=Path, help='JSON file attached to every artifact')
    parser.add_argument('--delete', action='store_true', help='Delete each local directory after verification')
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, job_type='upload-models')
    results = []
    for spec in args.model:
        name, _, path = spec.partition('=')
        source = Path(path)
        staging = None
        upload_dir = source
        if args.dtype == 'bfloat16':
            args.staging_root.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(dir=args.staging_root)) / name
            stage_bf16(source, staging)
            upload_dir = staging
        local = {str(p.relative_to(upload_dir)): p.stat().st_size for p in upload_dir.rglob('*') if p.is_file()}
        artifact = wandb.Artifact(f'{args.run_name}-{name}', type='model',
                                  metadata={**metadata, 'model': name, 'stored_dtype': args.dtype})
        artifact.add_dir(str(upload_dir))
        logged = run.log_artifact(artifact)
        logged.wait()
        remote = {entry.path: entry.size for entry in logged.manifest.entries.values()}
        verified = remote == local
        results.append({'model': name, 'artifact': logged.qualified_name, 'dtype': args.dtype,
                        'files': len(local), 'bytes': sum(local.values()), 'verified': verified})
        print(json.dumps(results[-1]), flush=True)
        if staging is not None:
            shutil.rmtree(staging.parent)
        if args.delete and verified:
            shutil.rmtree(source)
            print(f'Deleted local copy: {source}', flush=True)
        elif args.delete:
            print(f'Kept local copy, manifest mismatch: {source}', flush=True)
    run.finish()
    if not all(r['verified'] for r in results):
        raise SystemExit('Some uploads did not verify; local copies were kept')


if __name__ == '__main__':
    main()
