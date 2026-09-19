# Stage-kernel benchmark runs

Raw `soc_kernel_bench/1` JSON files from `cli/benchmark_soc_kernels.py`, one per
board, power mode and date:

    <hardware>_<power-mode>_<YYYYMMDD>.json

These are the measurements behind the measured efficiency tables in
`soc_designs/efficiency/`. `tools/ingest_soc_kernel_benchmarks.py` turns them into
those tables, and its `--check` flag keeps each table in step with its runs.
Commit the JSON unedited, exactly as the benchmark wrote it.

**Guide:** how to run the benchmark on a Jetson Orin Nano or AGX, check the output,
bring the file back and ingest it is in
[`docs/guides/soc_kernel_benchmarks_on_jetson.md`](../../../docs/guides/soc_kernel_benchmarks_on_jetson.md).
