# SAMSum continual-learning method comparison

The retained historical run compares Qwen/Qwen2-7B Base, Full SFT and LoRA
after two SAMSum epochs. The active code now prepares a fresh, matched run of
Full SFT, LoRA, empirical-Fisher EWC, Steer-LoRA, Steer-Full, and FAPM.
Steer is this project's own logit-margin/KL method; it is not the published
on-policy SDFT algorithm.

## Data and fairness contract

All trainable methods start from the same Qwen2-7B base, use the same prepared
SAMSum train/dev splits, 2 epochs, learning rate 1e-5, effective batch size 64,
sequence length 2048, AdamW/cosine schedule, warmup 0.03, weight decay 0.1,
and seed 42. LoRA and Steer-LoRA use the same rank 256, alpha 512, dropout
0.3 and target modules. Steer-Full and EWC train full weights. FAPM is a
post-processing method derived from this run's Full SFT checkpoint, not an
independent training budget.

EWC estimates its diagonal empirical Fisher before fine-tuning from up to 256
examples in the same SAMSum train split, masking user and system tokens so the
Fisher objective matches the assistant-only SFT loss. Its anchor is the initial
base checkpoint. This avoids requiring a separately prepared Fisher dataset
and keeps the data split shared. Since this experiment has no explicit earlier
task, this is Fisher-weighted regularization for SAMSum adaptation, not the
usual continual-learning setup where Fisher comes from an old task and protects
that task against forgetting. The sample cap and lambda are recorded in the
run manifest.

## Future matched run — not started yet

```bash
RUN_NAME=fair-methods-2epoch bash run_samsum_stable.sh
```

The script refuses to overwrite populated model/report directories. It trains
the six derived methods, evaluates SAMSum ROUGE-L with one evaluator, then
runs the same five lm-eval tasks/metrics for Base and every method.
Outputs go to `artifacts/samsum/$RUN_NAME/`. You may set `PYTHON_BIN`,
`EWC_LAMBDA`, `FISHER_MAX_SAMPLES`, `STEER_X_FACTOR` and `FAPM_K_RATIO`
explicitly; these method-specific choices should be selected using
validation data, never benchmark test scores.

## Re-evaluate existing 2-epoch checkpoints

The one active capability runner is `benchmarks/stable_five/run.py`. By
default it compares the retained Base, Full SFT and LoRA checkpoints on full
GSM8K zero-shot, HumanEval Instruct, IFEval, BoolQ and BBH CoT zero-shot
splits, using lm-eval 0.4.12. It persists each model/task result and resumes
only when saved settings and model paths match.

```bash
.venv/bin/python -u benchmarks/stable_five/run.py
```

Use `--model-path name=/path/to/checkpoint` and `--output-dir` to evaluate
additional methods with exactly the same tasks and scorer. The report shows
all selected methods and percentage-point gaps to Base.

## Source and historical artifacts

`eval/` contains only the active SAMSum evaluator and manifest writer.
`legacy_eval/` preserves the previous broad-suite evaluator, custom tasks and
report generator, including its pre-existing local edits; it is not called by
the new matched run. Historical reports and logs under
`artifacts/samsum/stable-2epoch/` and `stable-4epoch/` remain untouched.
See [ARTIFACTS.md](ARTIFACTS.md) for the layout. No training or benchmark was
launched during this code change.
