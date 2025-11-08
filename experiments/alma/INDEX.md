# Alma CPU Benchmarking - File Index

## Start Here

**New to Alma on CPU?** → `cpu_minimal_example.py` (run it!)

## Files Overview

### Working Examples

| File | Description | Status | When to Use |
|------|-------------|--------|-------------|
| **cpu_minimal_example.py** | Minimal CPU-only example | ✅ Ready | **START HERE** - Works on any CPU server |
| alma_integration.py | Advanced multi-tier validation | ⚠️ May OOM | Only if you have GPU or lots of RAM |

### Documentation

| File | Description | Audience |
|------|-------------|----------|
| **CLI_USAGE.md** | Command-line usage guide | Everyone using --model flag |
| **QUICKREF.md** | Quick reference card | Everyone - keep this handy |
| **CPU_SETUP_SUMMARY.md** | Detailed CPU configuration guide | Developers debugging config issues |
| **README.md** | Complete documentation | Full context and background |
| INDEX.md | This file | Navigation |

### Analysis & Background

| File | Description | Audience |
|------|-------------|----------|
| ALMA_ANALYSIS.md | Alma vs inductor_validation comparison | Researchers, decision makers |
| RCA.md | Root cause analysis | Developers fixing issues |
| QUICKSTART.md | Original quickstart (now superseded) | Historical reference |

## Recommended Reading Order

1. **First time?**
   - Run: `cpu_minimal_example.py`
   - Read: `QUICKREF.md`
   - Done! You're now benchmarking.

2. **Configuration issues?**
   - Read: `CPU_SETUP_SUMMARY.md`
   - Check: Lines 26, 228-233 in `cpu_minimal_example.py`
   - Verify: Run example and check printed config

3. **Want to understand more?**
   - Read: `README.md` (full context)
   - Read: `ALMA_ANALYSIS.md` (comparison with other tools)

4. **Advanced usage?**
   - Read: `README.md` sections on Tier 2/3
   - Consider: `alma_integration.py` (if GPU available)

## Quick Commands

```bash
# Run minimal example (recommended)
python3 experiments/alma/cpu_minimal_example.py

# Expected output:
# - CPU threads: 12
# - Baseline latency: ~2 ms
# - Best conversion: COMPILE_INDUCTOR or ONNX_CPU
# - Speedup: 1.5-4x

# Check if Alma is installed
python3 -c "from alma import benchmark_model; print('Alma OK')"

# Check CPU configuration
python3 -c "import torch; print(f'Threads: {torch.get_num_threads()}, CUDA: {torch.cuda.is_available()}')"

# View just the configuration output
python3 experiments/alma/cpu_minimal_example.py 2>&1 | grep -A 20 "CPU ENVIRONMENT"
```

## Configuration Quick Check

**Three critical settings** (see `QUICKREF.md` for details):

1. **CPU Threads** (line 44): Should match physical cores
2. **allow_cuda** (line 229): MUST be `False` on CPU-only
3. **allow_device_override** (line 233): MUST be `False`

Check these if you get:
- CUDA errors → Check `allow_cuda=False`
- Wrong performance → Check thread count
- GPU-related errors → Check `allow_device_override=False`

## File Sizes

```
cpu_minimal_example.py    17 KB   ← Main example
CPU_SETUP_SUMMARY.md      8.2 KB  ← Configuration guide
QUICKREF.md               4.5 KB  ← Quick reference
README.md                 17 KB   ← Full documentation
alma_integration.py       19 KB   ← Advanced (use with caution)
ALMA_ANALYSIS.md          23 KB   ← Background analysis
```

## What's New (2025-11-07)

✅ **NEW**: `cpu_minimal_example.py` - Minimal CPU-only example
✅ **NEW**: `CPU_SETUP_SUMMARY.md` - Detailed configuration guide
✅ **NEW**: `QUICKREF.md` - Quick reference card
✅ **UPDATED**: `README.md` - Added CPU-only quick start section

## Known Issues & Limitations

| Issue | File | Solution |
|-------|------|----------|
| OOM on large models | alma_integration.py | Use cpu_minimal_example.py instead |
| Process hangs | alma_integration.py | Use cpu_minimal_example.py (multiprocessing=False) |
| CUDA errors on CPU | alma_integration.py | Use cpu_minimal_example.py (allow_cuda=False) |
| COMPILE_INDUCTOR fails | cpu_minimal_example.py | Normal - some models don't support torch.compile |

## Help & Support

**Question**: "Which file should I use?"
**Answer**: `cpu_minimal_example.py` - it's designed for CPU-only servers

**Question**: "How do I configure for my CPU?"
**Answer**: See `QUICKREF.md` section "Critical Configuration Lines"

**Question**: "It's running out of memory"
**Answer**: Reduce `n_samples` from 128 to 64 (line 223)

**Question**: "COMPILE_INDUCTOR backend failed"
**Answer**: Normal - not all models support torch.compile. EAGER still works.

**Question**: "How do I add more backends?"
**Answer**: See `QUICKREF.md` section "Tested Backends"

---

**Last Updated**: 2025-11-07
**Maintainer**: graphs package team
**Status**: ✅ Production ready
