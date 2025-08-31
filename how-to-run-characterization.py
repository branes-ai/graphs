To run the `run_characterization.py` script from your `graphs` repo, here’s a step-by-step guide:

---

## 1. Set Up Your Environment

Make sure you have Python installed (≥3.8 recommended), and then install the required packages:

```bash
pip install torch torchvision pandas
```

If you’re using a virtual environment, activate it first.

---

## 📁 2. Directory Structure Check

Ensure your repo looks something like this:

```
graphs/
├── models/
│   └── mlp.py
│   └── conv2d_stack.py
│   └── resnet_block.py
├── characterize/
│   └── arch_profiles.py
│   └── tiling.py
│   └── fused_ops.py
│   └── walker.py
│   └── sweep.py
├── scripts/
│   └── run_characterization.py
```

Each model file (e.g. `mlp.py`) should expose a factory function like:

```python
def make_mlp(in_dim, hidden_dim, out_dim):
    return ParamMLP(in_dim, hidden_dim, out_dim)
```

---

## 3. Run the Script

From the root of your repo (where the `graphs/` folder lives), run:

```bash
python graphs/scripts/run_characterization.py
```

This will:
- Instantiate synthetic models
- Run FX tracing and shape propagation
- Characterize each model across CPU, GPU, TPU, and KPU
- Print metrics to the console
- Save results to `sweep_results.csv`

---

## 📊 4. View Results

After the run, you’ll find:

```bash
sweep_results.csv
```

This file contains rows like:

```
Model,Architecture,FLOPs,Memory,Tiles,Latency,Energy
MLP,CPU,...,...,...,...
Conv2D,GPU,...,...,...,...
...
```

You can open it in a spreadsheet or visualize it using matplotlib or seaborn.
