# V4 ground-truth capture: how to run on a target box

The V4 validation harness has two phases that run on **different** machines:

* **Capture-on-target** (this doc) — runs the workload on the actual silicon
  with a `Measurer` backend (RAPL/NVML/INA3221), writes a CSV baseline.
* **Validate-anywhere** — reads the committed CSV baseline, compares against
  the analyzer's predictions. Hardware-agnostic: runs on any developer
  laptop or in CI.

Capture is a one-time-per-hardware operation (re-run only when the analyzer
changes the *measurement* contract or you want fresher numbers). The CSV
gets committed under `validation/model_v4/results/baselines/<hw>_<op>.csv`
and from then on every developer can validate without the target box.

---

## Currently supported targets

| Hardware key                | Backend                | Measurer file                                                |
|-----------------------------|------------------------|--------------------------------------------------------------|
| `i7_12700k`                 | wall-clock + Intel RAPL | `validation/model_v4/ground_truth/pytorch_cpu.py`            |
| `h100_sxm5_80gb`            | cudaEvent + NVML        | `validation/model_v4/ground_truth/pytorch_cuda.py`           |
| `jetson_orin_nano_8gb`      | cudaEvent + INA3221     | `validation/model_v4/ground_truth/pytorch_jetson.py`         |
| `jetson_orin_agx_64gb`      | cudaEvent + INA3221     | `validation/model_v4/ground_truth/pytorch_jetson.py`         |
| `jetson_orin_nx_16gb`       | cudaEvent + INA3221     | `validation/model_v4/ground_truth/pytorch_jetson.py`         |

---

## Adding a new hardware target

Single point of change is `validation/model_v4/sweeps/_augment.py`:

1. Add a `_Target(hw_key=..., dtype=..., factory=...)` entry to `KNOWN_TARGETS`.
2. Run `python -m validation.model_v4.sweeps._augment --hw <new_hw_key>` to
   classify the existing sweep entries against the new mapper. This is purely
   additive — the original i7 / H100 / Jetson regimes stay byte-identical.
3. Add the same key to `_MEASURER_FACTORY` in
   `validation/model_v4/cli/capture_ground_truth.py` mapped to the right
   backend tag (`pytorch_cuda`, `pytorch_jetson`, `pytorch_cpu`, etc.).
4. Add the same key to `SWEEP_HW_TO_MAPPER` in
   `validation/model_v4/harness/runner.py` so the validate path can resolve
   it back to a mapper-registry name.
5. Update `tests/validation_model_v4/test_sweeps.py` `hw` fixture so
   `test_recorded_regime_labels_still_match_classifier` covers the new key.

---

## Capture procedure (target box)

The flow is identical for all targets — only the dependencies change.

### Common preflight (any target)

```bash
git clone https://github.com/branes-ai/graphs.git
cd graphs
pip install -e .
pip install pytest               # only needed if you want to run tests
```

The `cli/capture_ground_truth.py` script writes a CSV per `(hardware, op)`
pair. The validate-anywhere path expects exactly two CSVs per hardware
(matmul + linear). Both calibration and validation sweep entries are
captured into the same CSV — the harness picks which to use at validate
time based on the sweep file name.

### Target-specific dependencies

#### CPU (Intel, RAPL)

* Read access to `/sys/class/powercap/intel-rapl:0/energy_uj` (typically
  works for any user; some hardened systems may require `setcap` or root).

#### Desktop / server NVIDIA (H100, A100, RTX, ...)

```bash
pip install pynvml torch        # torch with CUDA wheels
nvidia-smi                       # confirm the driver is loaded
```

#### Jetson (Orin Nano / NX / AGX, Thor)

```bash
# JetPack already provides torch with cuda; verify:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"

# INA3221 sysfs is present on all Jetson devices but may need a one-time
# permission grant for non-root reads. The exact path varies by JetPack
# version; the measurer auto-detects both layouts.
ls /sys/bus/i2c/drivers/ina3221*/
```

If the rails come back empty, run as root once or `chmod -R a+r` the
ina3221 hwmon tree (some JetPack base images don't grant read to
non-root users).

### Capture command

```bash
# Replace <hw> with the hardware key from the table above.
# Both ops, both purposes (calibration + validation):
python -m validation.model_v4.cli.capture_ground_truth \
    --hw <hw> --op matmul --purpose calibration
python -m validation.model_v4.cli.capture_ground_truth \
    --hw <hw> --op matmul --purpose validation
python -m validation.model_v4.cli.capture_ground_truth \
    --hw <hw> --op linear --purpose calibration
python -m validation.model_v4.cli.capture_ground_truth \
    --hw <hw> --op linear --purpose validation
```

Wall time:

* CPU (i7-12700K, ~78 shapes per CSV, 11 trials each): ~5 minutes per CSV
* H100 (mostly sub-millisecond kernels): ~30 seconds per CSV
* Jetson Orin Nano (slower kernels, 8 ms INA3221 averaging adds settle
  time): ~10 minutes per CSV

Output CSVs land in `validation/model_v4/results/baselines/<hw>_<op>.csv`.

### Smoke test before the full run

To debug the capture flow without burning ~30 minutes, use `--limit`:

```bash
python -m validation.model_v4.cli.capture_ground_truth \
    --hw jetson_orin_nano_8gb --op matmul --purpose validation --limit 3
```

Inspect the CSV. If `latency_ms` is sane and `energy_mJ` is non-empty (or
the notes column explains why energy is `n/a`), you're good to launch the
full run.

### Commit the baseline back

From the target box:

```bash
git add validation/model_v4/results/baselines/<hw>_*.csv
git commit -m "feat(validation): V4 baseline for <hw>"
git push
```

Then the floor tests in `tests/validation_model_v4/test_v4_against_baseline.py`
should be extended with new `test_<hw>_matmul_pass_*_floor` and
`test_<hw>_linear_pass_*_floor` cases that pin the post-capture pass rates.

---

## Known limitations

* **NVML on Jetson**: Jetson's NVML reports an aggregate that sometimes
  excludes the memory-controller draw, which is why the `pytorch_jetson.py`
  measurer uses INA3221 sysfs instead. NVML on Jetson is *not* a fallback
  -- if INA3221 is unavailable, energy is reported as None.
* **Tegra power averaging**: INA3221 has an 8 ms internal averaging window.
  For sub-8 ms kernels the measurer adds a "below 8ms averaging period"
  warning to the notes column; treat those energy figures as indicative
  only. The same pattern as the CPU sub-1ms RAPL skip in `assertions.py`
  could be applied here as a follow-up.
* **Whole-machine assumption**: every backend (RAPL, NVML, INA3221)
  measures *device-level* power. Make the test box otherwise idle when
  capturing -- background processes inflate the energy figure for every
  sub-second kernel in the sweep.
