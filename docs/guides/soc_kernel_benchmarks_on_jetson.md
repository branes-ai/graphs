# Running the SoC stage-kernel benchmarks on a Jetson

This guide turns an Orin Nano or Orin AGX into measured efficiencies for the SoC
analyzer (graphs#269, Phase 5 step 5.3). It covers four steps: running the
benchmark on the board, checking the output, getting the file back, and turning
it into an efficiency table on a workstation.

**Why this matters.** The analyzer prices a stage on an engine as
`ops / (peak x efficiency)`. `default_v1` knows only two efficiencies, and
every other pair is a gap. The Orin-vs-KPU study
(`docs/assessments/soc-orin-vs-kpu-heterogeneous.md`) cannot be decided until
those gaps are measured. This run measures them.

The benchmark measures these kernel classes, at stage-like shapes:

- dense GEMM and 3x3 conv (FP32, FP16, and INT8 GEMM)
- attention prefill
- GEMV decode
- layer norm
- FFT
- batched Cholesky
- KKT solves
- scatter-add
- SGM cost volume and min-plus path aggregation (FP32, FP16)
- KLT feature tracking
- TSDF raycasting (FP32, FP16)
- ESDF wavefront propagation
- graph frontier relaxation
- an ISP pixel pipeline (FP32, FP16)

The last six were added in graphs#269 6.5, and each is sized like the stage
it stands for: `sgm`, `vio`, `tsdf` and `gain`, `esdf`, `graph` and `mono`.

It samples the device clock throughout every timed loop. GPU kernels use the
whole GPU. CPU kernels run pinned to one core with one thread, because the
analyzer treats a CPU core as one server.

## What you need

- **A Jetson Orin Nano or Orin AGX** on JetPack 6.
- **NVIDIA's JetPack build of PyTorch, with CUDA.** A PyTorch wheel from PyPI
  has no CUDA on a Jetson.
- **This repository, checked out on the board.**

## 1. Get the code onto the board

```bash
cd ~/dev/branes/clones/graphs && git pull
```

If it isn't there yet, clone it:

```bash
git clone <your graphs remote> ~/dev/branes/clones/graphs
cd ~/dev/branes/clones/graphs
```

## 2. Check the Python environment

**Do not run `pip install -e .` on the Jetson.** The package declares `torch`
and `torchvision` as dependencies, so pip can replace NVIDIA's CUDA build with
a CPU-only PyPI wheel. The benchmark does not need the package installed: the
CLI adds `src/` to its own path.

Check that torch sees the GPU:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

This must print `True`. If it prints `False`, or torch is missing, install
NVIDIA's PyTorch for your JetPack release first. See NVIDIA's "Installing
PyTorch for Jetson Platform" documentation.

Install the few small packages the benchmark imports, and check that it loads:

```bash
pip3 install numpy pyyaml tqdm pydot
python3 cli/benchmark_soc_kernels.py --help
```

If `--help` fails with `ModuleNotFoundError: No module named 'X'`, run
`pip3 install X`, unless X is `torch`.

## 3. Pick the top power mode and lock the clocks

A result counts as CALIBRATED only when the clock stayed steady while it ran:
at least five samples, all within 5% of each other. Under DVFS the clock
wanders, and the result drops to INTERPOLATED. The 2026-02 calibrations
recorded an idle 306 MHz at MAXN, which made their efficiencies meaningless.

```bash
sudo nvpmodel -q --verbose     # lists the modes: note the name and number of the highest-power one
sudo nvpmodel -m <mode-number>
sudo jetson_clocks             # pins CPU, GPU and EMC clocks at their maximum for that mode
sudo jetson_clocks --show      # optional: confirms the pinned frequencies
```

Mode numbers differ between boards and JetPack releases, so read them from
`nvpmodel -q --verbose` rather than assuming:

- On an Orin AGX, MAXN is usually mode 0.
- On an Orin Nano, the top mode is `MAXN_SUPER` on JetPack 6.2, and `15W`
  before it.

Use the mode's name as `--power-mode` in the next step. The name only labels
the file; it does not set the mode.

## 4. Run the benchmark

**Orin Nano:**

```bash
python3 cli/benchmark_soc_kernels.py --hardware jetson_orin_nano_8gb --power-mode MAXN_SUPER
```

**Orin AGX 64 GB:**

```bash
python3 cli/benchmark_soc_kernels.py --hardware jetson_orin_agx_64gb --power-mode MAXN
```

The run measures CPU and GPU kernels (every available device) and takes a few
minutes; the FP32 GEMM on one CPU core is the slowest. The six kernels added
in 6.5 add about half a minute of GPU work and a few minutes on one CPU core
(32 s of it on a 4.9 GHz desktop core, so scale by your clock). It writes the file named
below and prints a table.

```
soc_designs/efficiency/measurements/<hardware>_<power-mode>_<YYYYMMDD>.json
```

Useful options:

| Option | Default | Use |
|---|---|---|
| `--hardware NAME` | required | What the board is, in lower case with underscores. It goes into the file name and the JSON. |
| `--power-mode NAME` | none | The nvpmodel mode you set. It is a record only. |
| `--devices cpu,cuda` | every available | Measure one side only, for example `--devices cuda`. |
| `--kernels a,b` | all | Measure some kernel classes only, for example `--kernels dense_conv_gemm,small_qp`. |
| `--min-seconds S` | 0.5 | Length of each timed loop. Raise it if clocks are verified but results vary. |
| `--cpu-core N` | 0 | The core CPU kernels are pinned to. |
| `--output PATH` | the path above | Write somewhere else. |
| `--quick` | off | Small shapes, for a smoke test only. The ingest refuses quick runs. |
| `-v` | off | Also print the JSON. |

## 5. Read the table

| Column | What to look for |
|---|---|
| `status` | `ok` is measured. `unsupported` is expected where the harness has no path, such as INT8 GEMM on the CPU. `error` is worth reporting; the run continues past it. |
| `GOP/s` | Attained throughput, with ops counted as the Data Annex counts them (a multiply-accumulate is 2). |
| `clock_GHz` | The median clock sampled during that kernel. |
| `clock_ok` | **Should be `yes`.** `NO` means the clock moved more than 5% during the run. Check that `jetson_clocks` is active, then re-run. |
| `cpu_x` | CPU rows only: process CPU time over wall time. **Should be about 1.0**, because a CPU kernel is measured on one core. `WIDE` means it ran on several cores. The CLI sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before torch loads, so check that nothing in your shell overrides them. The ingest refuses wide rows. |

A note printed after the table says how many kernels ran with an unverified
clock.

## 6. Get the file back

From the board, commit it:

```bash
git checkout -b data/orin-nano-kernel-bench
git add soc_designs/efficiency/measurements/jetson_orin_nano_8gb_*.json
git commit -m "data(soc): Orin Nano stage-kernel benchmarks"
git push -u origin data/orin-nano-kernel-bench
```

Alternatively, copy it to a workstation and commit it there:

```bash
scp <jetson>:~/dev/branes/clones/graphs/soc_designs/efficiency/measurements/*.json \
    soc_designs/efficiency/measurements/
```

## 7. Turn it into an efficiency table (on a workstation)

This step needs the full environment, including `embodied-schemas`, so run it
on a development machine, not the Jetson.

```bash
# Orin Nano: 8 SMs, so the AGX-class design is scaled to match.
python tools/ingest_soc_kernel_benchmarks.py \
    --design orin_class_reference --count gpu_sm=8 \
    --table orin_nano_measured_v1 \
    soc_designs/efficiency/measurements/jetson_orin_nano_8gb_*.json

# Orin AGX: the design already has 16 SMs.
python tools/ingest_soc_kernel_benchmarks.py \
    --design orin_class_reference \
    --table orin_agx_measured_v1 \
    soc_designs/efficiency/measurements/jetson_orin_agx_64gb_*.json
```

The ingest writes `soc_designs/efficiency/<table>.yaml`, a table layered over
`default_v1`. Measured pairs replace unknowns, and everything else is
inherited. Commit it next to the run files. Re-running the same command with
`--check` fails if the committed table no longer matches its runs.

Efficiency is attained throughput divided by the design's per-server peak at
the *measured* clock. The ingest applies these rules:

- **Confidence.** An entry is CALIBRATED when its clock was verified, and
  INTERPOLATED otherwise.
- **No clock samples:** no entry.
- **No template peak for the format:** no entry. The Orin SM template states
  INT8, FP16 (tensor) and FP32 peaks but no FP64, so GPU FP64 results are
  reported but not ingested.
- **Efficiency above 1:** refused, not clipped.

Each skipped result is printed with its reason. Then use the table:

```bash
python cli/analyze_soc.py --design orin_class_reference --regime "far flight" \
    --efficiency orin_nano_measured_v1 --mapping ilp -v
python cli/sweep_soc.py --designs orin_class_reference,kpu_heterogeneous_h64 \
    --nodes tsmc_n16,tsmc_n7,tsmc_n5 --efficiency annex_v1,orin_nano_measured_v1 \
    --pareto area,power --union
```

## What this run does not measure

- **The DLA.** DLA kernels need TensorRT. `default_v1` keeps its DLA INT8 entry
  from the 2026-02 TensorRT runs.
- **kNN tree search.** It is the one kernel class still without a kernel:
  no stage of `branes_7tier_v1` uses it, and a kd-tree search is
  pointer-chasing that no torch kernel does faithfully. A brute-force
  distance matrix is a different algorithm, not this class, so it is not
  offered as one. The JSON lists it under `not_covered` and it stays a gap.

- **What an optimized library would do.** Each kernel measures *this*
  implementation on the engine. NVIDIA's VPI has a hardware SGM path and a
  fixed-function ISP, and an optimized CUDA raycaster would beat
  `grid_sample`; those are different numbers for the same stage. The entry
  records the kernel and shape it came from, as the INT8 `_int_mm` entry
  already does.
- **INT8 on the CPU.** The harness has no CPU INT8 path.
- **Multi-threaded CPU throughput.** By design, the analyzer's CPU server is
  one core.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | torch is a CPU build. Install NVIDIA's JetPack PyTorch. |
| `clock_ok` is `NO` | The clocks were not pinned. Run `sudo jetson_clocks`, check with `--show`, and re-run. |
| The INT8 GEMM row says `error` | The PyTorch build lacks `torch._int_mm`, or its CUDA path. Other kernels are unaffected, and the ingest reports the row as not ingested. |
| `ModuleNotFoundError` for a module other than torch | `pip3 install <module>`. |
| `cpu_x` shows `WIDE` | The BLAS thread pool ran on several cores. Unset any `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` in your shell (the CLI defaults them to 1) and re-run. |
| GPU FP32 above the FP32 peak | TF32 was on. The harness now turns it off (`tf32_disabled` in the JSON). Runs made before 2026-09-19 did not, and the ingest refuses their GPU FP32 rows. |
| The ingest says `efficiency ... > 1` | The design's peak for that engine is too low for the measured board, or the clock reading was wrong. Report it rather than editing the number. |

## Smoke test on a workstation

To check that the harness runs, without producing data the analysis uses:

```bash
python cli/benchmark_soc_kernels.py --hardware dev_box --quick --output /tmp/smoke.json
```

Related:

- `src/graphs/benchmarks/soc_kernels.py`: the harness.
- `tools/ingest_soc_kernel_benchmarks.py`: the ingest.
- `docs/plans/soc-phase5-execution-plan.md`: decision P5-D1.
