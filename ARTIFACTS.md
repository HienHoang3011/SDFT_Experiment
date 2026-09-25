# Artifact layout

```text
artifacts/samsum/
├── stable-2epoch/
│   ├── models/{full_sft,lora}/           # retained historical checkpoints
│   ├── reports/{base,full_sft,lora}/     # retained historical reports
│   ├── logs/
│   ├── new-benchmarks/                  # retained historical results
│   ├── new-benchmarks-smoke/            # retained historical results
│   ├── extra-lmeval-generation-heavy/  # retained historical results/log
│   └── stable-five/                     # focused rerun, if launched later
├── stable-4epoch/
│   ├── reports/{base,full_sft,lora}/     # retained historical reports
│   └── logs/
└── fair-methods-2epoch/                 # future new run; not created yet
    ├── models/{full_sft,lora,ewc,steer_lora,steer_full,fapm}/
    ├── reports/{base,full_sft,lora,ewc,steer_lora,steer_full,fapm}/
    └── stable-five/                     # same five tasks for every method
```

The last directory is a planned output layout, not an existing result.
Each new run writes a fairness manifest with EWC's SAMSum train Fisher
settings and other method-specific parameters. Use a fresh `RUN_NAME` to avoid
overwriting historical data. Historical reports were not removed.
