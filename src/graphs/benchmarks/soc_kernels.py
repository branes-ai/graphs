"""Stage-kernel benchmarks for the SoC analyzer's efficiency tables
(graphs#269 PR 5.1).

The SoC analyzer prices a stage on an engine as ops over (peak x efficiency),
keyed ``(kernel_class, engine_kind, precision)``. ``default_v1`` knows two
such efficiencies; every other pair is a gap. This harness measures them on
real hardware -- the Orin Nano and AGX first -- so the gaps close with data
rather than estimates (Phase 5 decision P5-D1: the measurement path).

Two things the older calibration files got wrong, and this gets right:

* **The clock is sampled while the kernel runs.** A background thread polls
  the device clock during every timed loop. The 2026-02 Orin calibrations
  recorded a 306 MHz "under load" reading at MAXN, which makes any
  efficiency computed from them meaningless. Here a run's clock is
  ``verified`` only when enough samples were taken and they agree within
  5%; the ingest (PR 5.2) marks an efficiency CALIBRATED only then.
* **A CPU kernel runs on one pinned core, single-threaded,** because the
  analyzer treats a CPU core as one server -- and the run proves it: process
  CPU time over wall time is recorded per kernel, and a kernel above
  ``SINGLE_THREAD_LIMIT`` is flagged (the first Orin Nano run used all six
  cores through the BLAS pool). A GPU kernel uses the whole GPU.
* **FP32 means FP32.** TF32 is switched off for matmul and cuDNN before GPU
  kernels run (``tf32_disabled`` in the document).

Each kernel counts ops the way the Data Annex does: a multiply-accumulate is
2 ops, an add or compare 1. The count is stated per kernel. Kernel classes
without a representative kernel here are listed in ``NOT_COVERED`` and stay
gaps.

A kernel measures *this* implementation of a class on the engine, not the
best one: NVIDIA's VPI has a hardware SGM path and a fixed-function ISP, and
an optimized CUDA raycaster would beat ``grid_sample``. The ingest records
the kernel and shape behind every entry for that reason.

The output is a JSON document (schema ``soc_kernel_bench/1``); nothing here
computes an efficiency, because that needs the composed block's peak, which
the ingest owns.
"""

from __future__ import annotations

import os
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

SCHEMA = "soc_kernel_bench/1"

#: Clock samples must agree within this fraction to count as verified.
CLOCK_TOLERANCE = 0.05
MIN_CLOCK_SAMPLES = 5

#: A CPU kernel is single-threaded when its process CPU time is at most this
#: multiple of wall time (1.0 plus the clock sampler and interpreter overhead).
SINGLE_THREAD_LIMIT = 1.25

#: Kernel classes with no representative kernel in this harness yet.
#: Kernel classes with no representative kernel here, and why. ``knn_tree``
#: is the only one left: no stage of ``branes_7tier_v1`` uses it, and a
#: kd-tree search is pointer-chasing that no torch kernel does faithfully --
#: a brute-force distance matrix is a different algorithm, not this class.
NOT_COVERED = ("knn_tree",)


# ---------------------------------------------------------------------------
# Clock sampling
# ---------------------------------------------------------------------------

_GPU_CLOCK_PATHS = (
    "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq",  # Orin Nano/NX, JP6
    "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq",                     # Orin AGX
    "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
    "/sys/class/devfreq/17000000.gpu/cur_freq",
)


def _read_hz(path: str) -> Optional[float]:
    try:
        with open(path) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def gpu_clock_reader() -> Optional[Callable[[], Optional[float]]]:
    """A reader of the GPU clock in Hz, or None where none is readable.

    Jetson exposes devfreq in Hz. A discrete NVIDIA GPU reports through
    NVML in MHz."""
    for path in _GPU_CLOCK_PATHS:
        if os.path.exists(path):
            return lambda p=path: _read_hz(p)
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM) * 1e6
    except Exception:  # noqa: BLE001 -- NVML absent or no GPU: no reader
        return None


def cpu_clock_reader(core: int) -> Optional[Callable[[], Optional[float]]]:
    """A reader of one core's clock in Hz (cpufreq reports kHz)."""
    path = f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq"
    if not os.path.exists(path):
        return None
    return lambda: (lambda v: None if v is None else v * 1e3)(_read_hz(path))


@dataclass
class ClockSamples:
    source: str
    samples_hz: List[float] = field(default_factory=list)

    @property
    def median_hz(self) -> Optional[float]:
        return statistics.median(self.samples_hz) if self.samples_hz else None

    @property
    def spread(self) -> Optional[float]:
        if not self.samples_hz:
            return None
        med = self.median_hz
        return (max(self.samples_hz) - min(self.samples_hz)) / med if med else None

    @property
    def verified(self) -> bool:
        return (len(self.samples_hz) >= MIN_CLOCK_SAMPLES
                and self.spread is not None and self.spread <= CLOCK_TOLERANCE)

    def to_dict(self) -> dict:
        return {"source": self.source, "n": len(self.samples_hz),
                "median_hz": self.median_hz, "min_hz": min(self.samples_hz, default=None),
                "max_hz": max(self.samples_hz, default=None), "spread": self.spread,
                "verified": self.verified}


class ClockSampler:
    """Polls a clock reader on a background thread while a block runs."""

    def __init__(self, reader: Optional[Callable[[], Optional[float]]], source: str,
                 period_s: float = 0.01):
        self.reader, self.period_s = reader, period_s
        self.result = ClockSamples(source if reader else "unavailable")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.reader()
            if value:
                self.result.samples_hz.append(value)
            self._stop.wait(self.period_s)

    def __enter__(self) -> "ClockSampler":
        if self.reader is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelSpec:
    """One representative kernel for a kernel class at a precision."""

    kernel_class: str
    name: str
    precision: str
    shape: str
    ops_per_call: float
    ops_rule: str  # how ops_per_call was counted
    build: Callable  # (device, dtype) -> zero-arg callable that runs one call
    devices: Tuple[str, ...] = ("cpu", "cuda")


def _dtype(precision: str):
    import torch  # noqa: PLC0415

    return {"fp32": torch.float32, "fp16": torch.float16, "fp64": torch.float64,
            "int8": torch.int8}[precision]


def _gemm(m: int, n: int, k: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        if dtype == torch.int8:
            a = torch.randint(-8, 8, (m, k), device=device, dtype=torch.int8)
            b = torch.randint(-8, 8, (k, n), device=device, dtype=torch.int8)
            return lambda: torch._int_mm(a, b)
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        return lambda: a @ b
    return build


def _conv(c_in: int, c_out: int, hw: int, k: int = 3):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        x = torch.randn(1, c_in, hw, hw, device=device, dtype=dtype)
        w = torch.randn(c_out, c_in, k, k, device=device, dtype=dtype)
        return lambda: torch.nn.functional.conv2d(x, w, padding=k // 2)
    return build


def _sdpa(heads: int, seq: int, dim: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        q, k_, v = (torch.randn(1, heads, seq, dim, device=device, dtype=dtype) for _ in range(3))
        return lambda: torch.nn.functional.scaled_dot_product_attention(q, k_, v)
    return build


def _gemv(n: int, k: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        x = torch.randn(1, k, device=device, dtype=dtype)
        w = torch.randn(k, n, device=device, dtype=dtype)
        return lambda: x @ w
    return build


def _layernorm(rows: int, cols: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        x = torch.randn(rows, cols, device=device, dtype=dtype)
        return lambda: torch.nn.functional.layer_norm(x, (cols,))
    return build


def _fft(rows: int, cols: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        x = torch.randn(rows, cols, device=device, dtype=dtype)
        return lambda: torch.fft.fft2(x)
    return build


def _cholesky_solve(batch: int, n: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        a = torch.randn(batch, n, n, device=device, dtype=dtype)
        spd = a @ a.transpose(-1, -2) + n * torch.eye(n, device=device, dtype=dtype)
        b = torch.randn(batch, n, 1, device=device, dtype=dtype)

        def run():
            factor = torch.linalg.cholesky(spd)
            return torch.cholesky_solve(b, factor)
        return run
    return build


def _kkt_solve(batch: int, n: int, m: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        size = n + m
        a = torch.randn(batch, size, size, device=device, dtype=dtype)
        kkt = a + size * torch.eye(size, device=device, dtype=dtype)
        rhs = torch.randn(batch, size, 1, device=device, dtype=dtype)
        return lambda: torch.linalg.solve(kkt, rhs)
    return build


def _scatter_add(points: int, bins: int):
    def build(device, dtype):
        import torch  # noqa: PLC0415

        idx = torch.randint(0, bins, (points,), device=device)
        src = torch.randn(points, device=device, dtype=dtype)
        out = torch.zeros(bins, device=device, dtype=dtype)
        return lambda: out.index_add_(0, idx, src)
    return build


def _sgm(width: int, height: int, disparities: int, paths: int):
    """Semi-global matching: a cost volume, then min-plus path aggregation.

    The aggregation is sequential along the scan direction -- that is what
    makes SGM SGM -- so each step is a vectorized (D x H) update and the
    scan is a Python loop, as an unfused implementation must be.
    """
    def build(device, dtype):
        import torch  # noqa: PLC0415

        left = torch.rand(height, width, device=device, dtype=dtype)
        right = torch.rand(height, width, device=device, dtype=dtype)
        penalty = torch.tensor(0.5, device=device, dtype=dtype)

        def run():
            shifted = torch.stack([torch.roll(right, d, dims=1) for d in range(disparities)])
            cost = (left.unsqueeze(0) - shifted).abs()          # D x H x W
            # The paths run as a batch dimension: one Python step per column,
            # not one per column per path, so the loop is the algorithm's
            # sequential dependence and not a launch-rate measurement.
            carry = cost[:, :, 0].expand(paths, -1, -1).contiguous()
            total = torch.empty_like(cost)
            total[:, :, 0] = carry.sum(dim=0)
            for x in range(1, width):
                best = torch.minimum(
                    carry,
                    torch.minimum(torch.roll(carry, 1, dims=1),
                                  torch.roll(carry, -1, dims=1)) + penalty)
                carry = cost[:, :, x].unsqueeze(0) + best - best.amin(dim=1, keepdim=True)
                total[:, :, x] = carry.sum(dim=0)
            return total.argmin(dim=0)      # a disparity map, H x W
        return run
    return build


def _klt(features: int, patch: int, iterations: int):
    """KLT feature tracking: patch gradients and a 2x2 solve per iteration."""
    def build(device, dtype):
        import torch  # noqa: PLC0415

        patches = torch.rand(features, patch, patch, device=device, dtype=dtype)
        target = torch.rand(features, patch, patch, device=device, dtype=dtype)
        eye = torch.eye(2, device=device, dtype=dtype) * 1e-3

        def run():
            flow = torch.zeros(features, 2, 1, device=device, dtype=dtype)
            gx = patches[:, :, 1:] - patches[:, :, :-1]
            gy = patches[:, 1:, :] - patches[:, :-1, :]
            gx, gy = gx[:, :-1, :], gy[:, :, :-1]
            g = torch.stack([gx.reshape(features, -1), gy.reshape(features, -1)], dim=2)
            hessian = g.transpose(1, 2) @ g + eye                 # N x 2 x 2
            for _ in range(iterations):
                residual = (target[:, :-1, :-1] - patches[:, :-1, :-1]).reshape(features, -1, 1)
                rhs = g.transpose(1, 2) @ residual
                flow = flow + torch.linalg.solve(hessian, rhs)
            return flow
        return run
    return build


def _raycast(rays: int, steps: int, volume: int):
    """Marching rays through a TSDF volume, trilinear at every step."""
    def build(device, dtype):
        import torch  # noqa: PLC0415

        vol = torch.rand(1, 1, volume, volume, volume, device=device, dtype=dtype)
        grid = torch.rand(1, 1, rays, steps, 3, device=device, dtype=dtype) * 2 - 1

        def run():
            sampled = torch.nn.functional.grid_sample(
                vol, grid, mode="bilinear", align_corners=False)      # trilinear in 3D
            return sampled.clamp_(-1, 1).mean()
        return run
    return build


def _wavefront(size: int, iterations: int):
    """ESDF propagation: min-plus relaxation over a 3D grid's 6 neighbours."""
    def build(device, dtype):
        import torch  # noqa: PLC0415

        field = torch.rand(size, size, size, device=device, dtype=dtype) * size
        step = torch.tensor(1.0, device=device, dtype=dtype)

        def run():
            out = field.clone()
            for _ in range(iterations):
                for dim in (0, 1, 2):
                    out = torch.minimum(out, torch.roll(out, 1, dims=dim) + step)
                    out = torch.minimum(out, torch.roll(out, -1, dims=dim) + step)
            return out
        return run
    return build


def _graph_search(nodes: int, degree: int, iterations: int):
    """Frontier relaxation over a sampled graph: the data-parallel form of
    a shortest-path search, with a collision cost on every edge."""
    def build(device, dtype):
        import torch  # noqa: PLC0415

        src = torch.arange(nodes, device=device).repeat_interleave(degree)
        dst = torch.randint(0, nodes, (nodes * degree,), device=device)
        weight = torch.rand(nodes * degree, device=device, dtype=dtype)

        def run():
            dist = torch.full((nodes,), float("inf"), device=device, dtype=dtype)
            dist[0] = 0
            for _ in range(iterations):
                dist = dist.scatter_reduce(0, dst, dist[src] + weight, reduce="amin")
            return dist
        return run
    return build


def _isp(width: int, height: int):
    """A mono front-end's pixel pipeline: demosaic, white balance, colour
    matrix and gamma, as a programmable engine runs it."""
    def build(device, dtype):
        import torch  # noqa: PLC0415

        bayer = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        demosaic = torch.rand(3, 1, 3, 3, device=device, dtype=dtype)
        gains = torch.tensor([1.9, 1.0, 1.6], device=device, dtype=dtype).view(1, 3, 1, 1)
        ccm = torch.rand(3, 3, device=device, dtype=dtype)

        def run():
            rgb = torch.nn.functional.conv2d(bayer, demosaic, padding=1) * gains
            corrected = torch.einsum("ij,bjhw->bihw", ccm, rgb)
            return corrected.clamp_(1e-6, 1.0).pow(1 / 2.2)
        return run
    return build


def _cubic_lu(n: int) -> float:
    return 2.0 * n ** 3 / 3.0


def kernel_suite(quick: bool = False) -> List[KernelSpec]:
    """The kernels, sized like the stages they stand for (``quick`` shrinks
    them for smoke tests; ops counts follow the shapes)."""
    s = 4 if quick else 1
    g, c_hw, seq, v_k = 2048 // s, 80 // s, 1024 // s, 4096 // s
    kf = 1024 // s
    specs: List[KernelSpec] = []
    for prec in ("fp32", "fp16", "int8"):
        specs.append(KernelSpec(
            "dense_conv_gemm", "gemm", prec, f"{g}x{g}x{g}", 2.0 * g ** 3,
            "2 x M x N x K", _gemm(g, g, g),
            devices=("cuda",) if prec == "int8" else ("cpu", "cuda")))
    for prec in ("fp32", "fp16"):
        specs.append(KernelSpec(
            "dense_conv_gemm", "conv3x3", prec, f"256->256 @ {c_hw}x{c_hw}",
            2.0 * 256 * 256 * 9 * c_hw * c_hw, "2 x Cin x Cout x 9 x H x W",
            _conv(256, 256, c_hw)))
        specs.append(KernelSpec(
            "attention_prefill", "sdpa", prec, f"16 heads x {seq} x 64",
            2 * 2.0 * 16 * seq * seq * 64, "2 matmuls x 2 x H x S x S x D",
            _sdpa(16, seq, 64)))
        specs.append(KernelSpec(
            "weight_stream_decode", "gemv", prec, f"1x{v_k} . {v_k}x{v_k}",
            2.0 * v_k * v_k, "2 x K x N", _gemv(v_k, v_k)))
        specs.append(KernelSpec(
            "elementwise_norm", "layernorm", prec, f"{v_k}x{v_k}",
            5.0 * v_k * v_k, "5 per element (mean, var, normalize)", _layernorm(v_k, v_k)))
    specs.append(KernelSpec(
        "fft", "fft2", "fp32", f"{kf}x128", 5.0 * kf * 128 * __import__("math").log2(kf * 128),
        "5 N log2 N (complex, radix-2 convention)", _fft(kf, 128)))
    for prec in ("fp32", "fp64"):
        specs.append(KernelSpec(
            "small_dense_linalg", "cholesky_solve", prec, "256 x 64x64",
            256 * (64 ** 3 / 3.0 + 2.0 * 64 ** 2), "n^3/3 factor + 2 n^2 solve, per matrix",
            _cholesky_solve(256, 64)))
        specs.append(KernelSpec(
            "small_qp", "kkt_solve", prec, "64 x (32+16)^2",
            64 * (_cubic_lu(48) + 2.0 * 48 ** 2),
            "2n^3/3 LU + 2 n^2 solve on the KKT system: one QP iteration, not a full solve",
            _kkt_solve(64, 32, 16)))
    specs.append(KernelSpec(
        "sparse_hash_scatter", "index_add", "fp32", f"{1_000_000 // s} pts -> 262144 bins",
        float(1_000_000 // s), "1 add per point", _scatter_add(1_000_000 // s, 262144)))

    # The classes the 2026-09 suite left uncovered (graphs#269 5.1), each
    # sized like the stage it stands for.
    sw, sh, disp, sgm_paths = 1280 // s, 720 // s, 64 // s, 4
    for prec in ("fp32", "fp16"):
        specs.append(KernelSpec(
            "cost_volume_dp", "sgm", prec, f"{sw}x{sh} x {disp} disp x {sgm_paths} paths",
            disp * sh * sw + sgm_paths * disp * sh * (sw - 1) * 7.0,
            "1 per cost-volume element (D x H x W), then 7 per element per path per column "
            "for the min-plus scan (2 compares, penalty, cost, row min, subtract, sum)",
            _sgm(sw, sh, disp, sgm_paths)))
    feats, patch, klt_iters = 1024 // s, 15, 5
    m = (patch - 1) ** 2
    specs.append(KernelSpec(
        "feature_track", "klt", "fp32", f"{feats} features x {patch}x{patch} x {klt_iters} iters",
        feats * m * 10.0 + klt_iters * (feats * m * 5.0 + feats * 12.0),
        "2 per pixel for the gradients and 8 for the Hessian, then per iteration 1 for the "
        "residual, 4 for G^T r and 12 per feature for the 2x2 solve",
        _klt(feats, patch, klt_iters)))
    rays, ray_steps, vol = 65536 // s, 64, 128
    for prec in ("fp32", "fp16"):
        specs.append(KernelSpec(
            "raycast", "tsdf_raycast", prec, f"{rays} rays x {ray_steps} steps in {vol}^3",
            rays * ray_steps * 24.0,
            "24 per sample: 8 trilinear weights (14 multiplies, 7 adds) and the clamp",
            _raycast(rays, ray_steps, vol)))
    grid, sweeps = 160 // s, 8
    specs.append(KernelSpec(
        "wavefront", "esdf_propagate", "fp32", f"{grid}^3 x {sweeps} sweeps",
        sweeps * 6.0 * grid ** 3 * 2.0,
        "6 neighbours per sweep, 2 per neighbour per voxel (add and min)",
        _wavefront(grid, sweeps)))
    nodes, degree, relax = 32768 // s, 8, 30
    specs.append(KernelSpec(
        "graph_search", "frontier_relax", "fp32",
        f"{nodes} nodes x degree {degree} x {relax} relaxations",
        relax * nodes * degree * 2.0,
        "1 add and 1 min per edge per relaxation",
        _graph_search(nodes, degree, relax)))
    iw, ih = 1920 // s, 1080 // s
    for prec in ("fp32", "fp16"):
        specs.append(KernelSpec(
            "pixel_fixed_function", "isp_pipeline", prec, f"{iw}x{ih}",
            iw * ih * 84.0,
            "per pixel: 54 for a 3x3 demosaic to 3 channels, 3 white balance, 18 for the "
            "3x3 colour matrix, 1 clamp and 8 for the gamma",
            _isp(iw, ih)))
    return specs


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class KernelResult:
    kernel_class: str
    name: str
    precision: str
    engine_kind: str
    device: str
    shape: str
    ops_per_call: float
    ops_rule: str
    calls: int
    seconds_per_call: Optional[float] = None
    attained_ops_per_s: Optional[float] = None
    clock: Optional[dict] = None
    #: CPU kernels: process CPU time over wall time during the timed loop.
    #: About 1.0 for one thread; the core count when a BLAS pool ran wide.
    cpu_parallelism: Optional[float] = None
    single_thread: Optional[bool] = None
    status: str = "ok"  # ok | unsupported | error
    message: str = ""


def _sync(device: str) -> None:
    if device == "cuda":
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()


def run_kernel(spec: KernelSpec, device: str, min_seconds: float = 0.5, warmup: int = 3,
               cpu_core: int = 0) -> KernelResult:
    """Time one kernel on one device, sampling the clock throughout."""
    engine = "gpu" if device == "cuda" else "cpu"
    base = dict(kernel_class=spec.kernel_class, name=spec.name, precision=spec.precision,
                engine_kind=engine, device=device, shape=spec.shape,
                ops_per_call=spec.ops_per_call, ops_rule=spec.ops_rule, calls=0)
    if device not in spec.devices:
        return KernelResult(**base, status="unsupported",
                            message=f"{spec.name} {spec.precision} has no {device} path here")
    try:
        call = spec.build(device, _dtype(spec.precision))
        for _ in range(warmup):
            call()
        _sync(device)
        reader = gpu_clock_reader() if device == "cuda" else cpu_clock_reader(cpu_core)
        source = "gpu" if device == "cuda" else f"cpu{cpu_core}"
        calls, start, cpu_start = 0, time.perf_counter(), time.process_time()
        with ClockSampler(reader, source) as sampler:
            while True:
                call()
                calls += 1
                if calls % 8 == 0 or device == "cpu":
                    _sync(device)
                    if time.perf_counter() - start >= min_seconds:
                        break
            _sync(device)
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start
    except Exception as exc:  # noqa: BLE001 -- recorded per kernel, the run continues
        return KernelResult(**base, status="error", message=f"{type(exc).__name__}: {exc}")
    base["calls"] = calls
    per_call = elapsed / calls
    parallelism = cpu_elapsed / elapsed if device == "cpu" and elapsed > 0 else None
    return KernelResult(**base, seconds_per_call=per_call,
                        attained_ops_per_s=spec.ops_per_call / per_call,
                        clock=sampler.result.to_dict(), cpu_parallelism=parallelism,
                        single_thread=None if parallelism is None else parallelism <= SINGLE_THREAD_LIMIT)


def _pin_cpu(core: int) -> str:
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    try:
        os.sched_setaffinity(0, {core})
        return f"pinned to core {core}, 1 thread"
    except (AttributeError, OSError) as exc:
        return f"not pinned ({exc}); 1 thread"


def _lock_fp32() -> bool:
    """Make FP32 mean FP32 on the GPU. PyTorch lets cuDNN run FP32
    convolutions as TF32 on tensor cores by default: an Orin Nano "FP32"
    conv measured twice the FP32 peak. Returns True once both are off."""
    import torch  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def _environment() -> dict:
    import torch  # noqa: PLC0415

    env = {"platform": platform.platform(), "machine": platform.machine(),
           "python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.cuda.is_available(),
           "threads": {v: os.environ.get(v) for v in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "torch_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    nvp = Path("/etc/nvpmodel.conf")
    if nvp.exists():
        env["jetson"] = True
    return env


def run_suite(devices: List[str], hardware: str, power_mode: str = "", quick: bool = False,
              min_seconds: float = 0.5, cpu_core: int = 0,
              kernels: Optional[List[str]] = None) -> dict:
    """Run the suite and return the ``soc_kernel_bench/1`` document."""
    specs = [k for k in kernel_suite(quick) if not kernels or k.kernel_class in kernels]
    notes = {}
    if "cpu" in devices:
        notes["cpu"] = _pin_cpu(cpu_core)
    tf32_disabled = _lock_fp32() if "cuda" in devices else None
    results = [run_kernel(spec, dev, min_seconds=min_seconds, cpu_core=cpu_core)
               for dev in devices for spec in specs]
    return {
        "schema": SCHEMA,
        "hardware": hardware,
        "power_mode": power_mode,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "cpu_threading": notes.get("cpu", ""),
        "single_thread_limit": SINGLE_THREAD_LIMIT,
        "tf32_disabled": tf32_disabled,
        "clock_tolerance": CLOCK_TOLERANCE,
        "not_covered": list(NOT_COVERED),
        "results": [asdict(r) for r in results],
    }


__all__ = ["CLOCK_TOLERANCE", "ClockSampler", "ClockSamples", "KernelResult", "KernelSpec",
           "NOT_COVERED", "SCHEMA", "kernel_suite", "run_kernel", "run_suite"]
