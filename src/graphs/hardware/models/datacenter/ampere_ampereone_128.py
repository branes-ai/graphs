from .cpu_arm import cpu_arm_resource_model
from ...resource_model import HardwareResourceModel


def ampere_ampereone_128_resource_model() -> HardwareResourceModel:
    """
    Ampere AmpereOne 128-core ARM Server Processor.
    Uses ARM CPU helper with Ampere One specs.

    TDP: 250W (datacenter ARM CPU with 128 cores @ 5nm)
    """
    # Use ARM CPU helper with Ampere One specs
    return cpu_arm_resource_model(
        num_cores=128,
        process_node_nm=5,  # TSMC 5nm
        scalar_freq_ghz=3.6,  # Up to 3.6 GHz all-core
        name_suffix="AmpereOne-128",
        tdp_watts=250.0  # Datacenter class TDP
    )
