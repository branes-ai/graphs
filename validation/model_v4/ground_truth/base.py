"""Ground-truth measurer protocol + Measurement dataclass.

Principle 2 of the v4 plan: ground truth comes from outside this repo.
Each ground-truth backend (PyTorch CPU + RAPL, PyTorch CUDA + NVML,
KPU simulator CSV reader, ...) implements the same ``Measurer``
interface so the harness runner is agnostic to the source.

A ``Measurement`` is the outcome of running a ``WorkloadArtifact`` on
real (or simulated) silicon. Energy is ``Optional`` because not every
target exposes it (e.g., Coral TPU lacks an energy counter, RAPL may
require root, NVML may not be installed).

Identity (op, shape, dtype) is **not** stored in the Measurement -- it
belongs to the WorkloadArtifact / sweep entry that produced it. The
disk cache adds the (hardware, op, shape, dtype) tuple as the lookup
key when persisting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class Measurement:
    """Outcome of running one workload on real or simulated silicon.

    All times in seconds, all energies in joules. ``trial_count`` is
    the number of repetitions averaged into the median latency, useful
    when a measurer wants to surface its own variance estimate later.
    """
    latency_s: float
    energy_j: Optional[float]
    trial_count: int
    notes: str = ""


class Measurer(Protocol):
    """Backend-agnostic measurement protocol.

    Implementations:

    * ``ground_truth.pytorch_cpu.PyTorchCPUMeasurer`` -- PyTorch on CPU
      with optional perf+RAPL energy capture.
    * ``ground_truth.pytorch_cuda.PyTorchCUDAMeasurer`` (V4-4) --
      PyTorch on CUDA with cudaEvent timing and NVML energy.
    * ``ground_truth.simulator_kpu.SimulatorKPUMeasurer`` (V4-5) --
      reads cycle-accurate sim CSV and converts cycles to seconds.

    Implementations are stateful only insofar as they may cache
    hardware-handle objects (CUDA context, NVML handle, RAPL fd).
    """

    name: str          # short identifier, e.g., "pytorch_cpu_rapl"
    hardware: str      # hardware key, e.g., "i7_12700k"

    def measure(self, model, inputs) -> Measurement:
        """Run ``model(*inputs)`` and return a Measurement.

        Implementations should:
        - run a few warm-up iterations before the timed window
        - take the median (not mean) over N trials
        - never raise on missing-energy: instead return Measurement
          with energy_j=None and a note explaining why
        - clean up any hardware handles in ``__exit__`` if used as a
          context manager (optional)
        """
        ...
