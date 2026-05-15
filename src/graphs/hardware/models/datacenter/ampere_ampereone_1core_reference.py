from .cpu_arm import cpu_arm_resource_model
from ...resource_model import HardwareResourceModel


def ampere_ampereone_1core_reference_resource_model() -> HardwareResourceModel:
    """Single-core AmpereOne reference design (synthetic).

    NOT a shipping product. Ampere only sells full-die AmpereOne parts
    (128c / 192c / 96c). This factory returns a one-core slice of the
    same architectural unit -- same core IP (ARM v8.6+ Neoverse-class),
    same process node (TSMC 5nm), same per-core SIMD width -- as a
    reference data point for sanity-checking what the 192-core SKU
    "should" look like on workloads that can't realistically fan out
    across all cores (e.g. batch=1 matvec).

    Use case: filed as part of issue #175 (CPU mapper batch=1 fanout
    overcount). The 192-core SKU should not report 192x the throughput
    of this single-core reference on a workload that's geometrically
    incapable of using more than a handful of cores.

    TDP estimate: 350W / 192c = ~1.8 W/core core-side. Round up to 5 W
    to absorb a per-core share of uncore (memory controllers, mesh
    interconnect, idle floor). This is an estimate, not a datasheet
    value -- the real per-core power isn't separately spec'd.

    L2: cpu_arm_resource_model parameterises l2_cache_total as
    num_cores * 512 KB, so this single-core variant has 512 KB of L2
    available -- which is exactly what one core would have in isolation.
    Memory bandwidth: helper's hardcoded 80 GB/s default. A single core
    can't saturate >100 GB/s anyway, so this is fine for the reference
    role even though the real AmpereOne datasheet quotes 332.8 GB/s
    aggregate (separately tracked in issue #175).
    """
    return cpu_arm_resource_model(
        num_cores=1,
        process_node_nm=5,
        scalar_freq_ghz=3.6,
        name_suffix="AmpereOne-1core-ref",
        tdp_watts=5.0,
    )
