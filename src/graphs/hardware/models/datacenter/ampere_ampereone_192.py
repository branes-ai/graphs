from .cpu_arm import cpu_arm_resource_model
from ...resource_model import HardwareResourceModel


def ampere_ampereone_192_resource_model() -> HardwareResourceModel:
    """
    Ampere AmpereOne 192-core ARM Server Processor.
    Uses ARM CPU helper with Ampere One specs.

    TDP: 350W (datacenter ARM CPU with 192 cores @ 5nm)
    """
    # Use ARM CPU helper with Ampere One specs
    return cpu_arm_resource_model(
        num_cores=192,
        process_node_nm=5,  # TSMC 5nm
        scalar_freq_ghz=3.6,  # Up to 3.6 GHz all-core
        name_suffix="AmpereOne-192",
        tdp_watts=350.0  # Datacenter class TDP
    )