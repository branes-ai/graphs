# graphs
Collection of compute graph definitions

Repo structure for synthetic workloads and FX graph-based workload characterization and estimation

```txt
graphs/
├── src/                 # Existing synthetic DNN definitions
│   └── graphs/              # Example: parameterized MLP
│       └── models/                 # Existing synthetic DNN definitions
│       │   └── mlp.py              # Example: parameterized MLP
│       │   └── resnet_block.py     # Example: synthetic ResNet block
│       │   └── conv2d_stack.py     # Example: stacked Conv2D layers
│       │
│       └── characterize/           # Characterization pipeline
│           ├── __init__.py
│       │   ├── arch_profiles.py    # ArchitectureProfile + scheduler/tiling models
│       │   ├── tiling.py           # TilingStrategy base + CPU/GPU/TPU/KPU strategies
│       │   ├── fused_ops.py        # FusedOpRegistry + estimators
│       │   ├── introspect.py       # OperatorIntrospector + FusedOpIntrospector
│       │   ├── walker.py           # FXGraphWalker using introspection + estimators
│       │   ├── sweep.py            # SweepHarness to run across models and profiles
│       │   └── visualize.py        # Optional: matplotlib plots for latency/energy
│
├       └── scripts/
│           └── run_characterization.py  # CLI entry point to run sweeps
│       │ 
├       └── tests/
│           └── test_characterize.py     # Unit tests for estimators and walker
│
├── README.md
└── requirements.txt
```
