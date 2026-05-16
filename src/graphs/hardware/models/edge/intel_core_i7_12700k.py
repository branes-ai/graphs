"""Intel Core i7-12700K resource model.

PR 5 of the CPU sprint scoped at #182. As of this PR, the chip's
architectural and thermal-profile data lives in the canonical YAML
at ``embodied-schemas:data/compute_products/intel/intel_core_i7_12700k.yaml``
(landed in embodied-schemas#24) and is loaded via ``cpu_yaml_loader``.
The previous ~425-LOC hand-coded ``HardwareResourceModel`` constructor
was retired here; its parity with the YAML-loaded model is established
in PR #185's ``test_cpu_yaml_loader_i7_12700k_parity.py``.

The public function name and signature are preserved so existing
callers (``create_i7_12700k_mapper``, layer1 reporting, validation
scripts, ``cli/compare_*.py``) continue to work unchanged.

**No overlays** -- unlike the GPU SKU factories (AGX Orin in #181,
Thor in #183) which layer a ``BOMCostProfile`` and per-profile
``memory_clock_mhz`` after loading, the i7 factory has nothing to
overlay. The original hand-coded factory carried no ``bom_cost_profile``
field, and CPU DDR5 doesn't gate on thermal profile so per-profile
memory_clock_mhz isn't a CPU concern. PR 5's job is the pure
substitution.

Two known parity gaps remain between this factory and the prior
hand-coded values; both documented in the loader's parity test and
accepted as v3 trade-offs:
  - Per-precision peak ops ~5% higher (loader uses chip-level scalar
    clock for both clusters; hand-coded used cluster-specific clocks).
    v4 fix: thread per-cluster ``ClockDomain`` through chip-level
    ``Power.thermal_profiles``.
  - FP32 energy 2x lower (YAML uses Intel 7 ``hp_logic:fp32`` from the
    process node entry; hand-coded used Horowitz 10nm ``simd_packed``).
    v4 fix: add SIMD-packed energy entries to the process-node YAML.

The legacy resource-model ``name`` ("Intel-Core-i7-12700K", no spaces)
is preserved for backward compatibility with the existing graphs
CPUMapper + layer1 reporting + microarch validation report. The YAML's
``name`` field is "Intel Core i7-12700K" -- that's the human-readable
form future schema-aware consumers should prefer.
"""

from ...resource_model import HardwareResourceModel
from ..datacenter.cpu_yaml_loader import load_cpu_resource_model_from_yaml


_LEGACY_NAME = "Intel-Core-i7-12700K"
_YAML_BASE_ID = "intel_core_i7_12700k"


def intel_core_i7_12700k_resource_model() -> HardwareResourceModel:
    """Intel Core i7-12700K (12th Gen Alder Lake hybrid 8P + 4E).

    See the YAML for the canonical chip description (Golden Cove P-cores
    + Gracemont E-cores, AVX2 + AVX-VNNI, 25 MB shared L3 LLC,
    DDR5-4800, snoopy MESI, double-ring NoC) and the migration notes
    at ``docs/designs/cpu-compute-product-schema-extension.md``.
    """
    return load_cpu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )
