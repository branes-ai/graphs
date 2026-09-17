"""The autonomy workload port, checked against the model it came from.

``docs/workload-model/`` (Autonomy Workload Data Annex, 2026-09-17) is the
reference implementation: it derives every stage unit cost forward from
algorithm structure and instantiates eighteen missions as sensor suites and
update rates. ``graphs.core.pipeline_workload`` is a port, and
``workloads/pipelines/autonomy/branes_7tier_v1.yaml`` is generated from it.

So the first test is parity: run the model, run the port, require the same
numbers. The rest check the published figures the model is supposed to
reproduce, and the properties the comparison work will lean on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from graphs.core.pipeline_workload import (
    CLASS_NAMES,
    EffectiveThroughput,
    PipelineWorkload,
    Stage,
    load_autonomy_workload,
)

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "docs" / "workload-model"


@pytest.fixture(scope="module")
def workload() -> PipelineWorkload:
    return load_autonomy_workload()


@pytest.fixture(scope="module")
def reference():
    """The reference model's own summaries, keyed by (form factor, name)."""
    sys.path.insert(0, str(MODEL_DIR))
    try:
        import profiles  # noqa: PLC0415
    finally:
        sys.path.remove(str(MODEL_DIR))
    return {(p.ff, p.name): p.summary() for p in profiles.PROFILES}


# ---------------------------------------------------------------------------
# Parity with the reference implementation
# ---------------------------------------------------------------------------


def test_every_profile_reproduces_the_reference_model(workload, reference):
    assert len(workload.profiles) == len(reference) == 18
    for profile in workload.profiles:
        ref = reference[(profile.form_factor, profile.name)]
        got = workload.summary(profile)
        where = profile.name
        assert got.tops == pytest.approx(ref["tops"], rel=1e-12), where
        assert got.gb_per_s == pytest.approx(ref["bw"], rel=1e-12), where
        assert got.oversubscription == pytest.approx(ref["occupancy"], rel=1e-12), where
        assert got.reactive_chain_ms == pytest.approx(ref["chain_ms"], rel=1e-12), where
        shares = got.class_shares()
        for name, want in zip(CLASS_NAMES, (ref["A"], ref["B"], ref["C"])):
            assert shares[name] == pytest.approx(want, rel=1e-12), (where, name)
        assert set(got.stages_over()) == set(ref["over"]), where


def test_the_yaml_is_what_the_model_generates(workload):
    """The catalog copy is generated, not maintained. If someone edits it by
    hand, or the model changes and it is not regenerated, this fails."""
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import generate_autonomy_workload as gen  # noqa: PLC0415
    finally:
        sys.path.remove(str(REPO / "tools"))
    assert gen.build().to_dict() == workload.to_dict()


def test_parametric_stages_carry_their_mission_configuration(workload):
    """SGM at 1280x720 with 96 disparities is not SGM at 1440x1080 with 128.
    Storing only the reference unit cost would silently misprice every
    mission whose sensor suite differs from it."""
    interceptor = workload.profile("air superiority")
    inspection = workload.profile("Inspection (structure / asset)")
    sgm_i = next(d for d in workload.demands(interceptor) if d.stage.key == "sgm")
    sgm_n = next(d for d in workload.demands(inspection) if d.stage.key == "sgm")
    assert sgm_i.stage.ops_per_call > sgm_n.stage.ops_per_call
    # The humanoid's 60-state MPC costs far more per solve than the drone's 12.
    house = workload.profile("House work (open-world, long-horizon)")
    mpc_h = next(d for d in workload.demands(house) if d.stage.key == "mpc")
    mpc_i = next(d for d in workload.demands(interceptor) if d.stage.key == "mpc")
    assert mpc_h.stage.ops_per_call > 50 * mpc_i.stage.ops_per_call


def test_profiles_carry_the_sensor_suite_they_are_configured_with(workload):
    """A fixed-function core states contract limits -- the SGM core is rated
    to 1920x1080 at 30 fps with 128 disparities, the Navion-class VIO core to
    752x480 -- and they can only be checked against a mission whose sensor
    configuration is stated. Rates and unit costs alone cannot answer it."""
    interceptor = workload.profile("air superiority")
    assert interceptor.sensors["stereo"] == [1, 1440, 1080, 40, 128]
    assert interceptor.sensors["vio"] == [2, 1440, 1080, 40]
    assert interceptor.sensors["mpc_dims"] == [12, 4, 30]
    for profile in workload.profiles:
        assert profile.sensors, profile.name
        for key in ("stereo", "mono", "vio"):
            if key in profile.sensors:
                assert len(profile.sensors[key]) in (4, 5), (profile.name, key)


# ---------------------------------------------------------------------------
# The published figures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, tops, gb_s, oversub, over", [
    ("far flight", 10.47, 306, 16.2, 3),
    ("air superiority", 12.46, 102, 16.6, 3),
])
def test_annex_published_rows(workload, name, tops, gb_s, oversub, over):
    """Data Annex section 4, the two drone rows that are also regimes."""
    got = workload.summary(workload.profile(name))
    assert got.tops == pytest.approx(tops, abs=0.005)
    assert got.gb_per_s == pytest.approx(gb_s, abs=0.5)
    assert got.oversubscription == pytest.approx(oversub, abs=0.05)
    assert len(got.stages_over()) == over
    assert got.reactive_chain_ms == pytest.approx(147, abs=0.5)


def test_the_two_derivations_agree_within_four_percent(workload):
    """The annex's own validation (section 6): this model and the 2026-09-13
    argument document reach the regimes independently, sharing no
    coefficient. Agreement is a check, not a construction -- so it is stated
    as a tolerance, not an equality."""
    for profile in workload.regimes():
        got = workload.summary(profile)
        published = profile.published
        assert abs(got.tops / published["tops"] - 1) <= 0.04, profile.regime
        assert abs(
            got.oversubscription / published["oversubscription"] - 1
        ) <= 0.04, profile.regime


def test_the_worst_profile_is_the_driverless_car(workload):
    """Data Annex section 4: SAE L4/L5 demands 48 seconds of compute per
    second of mission, with 8 stages unable to sustain their own rate."""
    worst = max(workload.summaries(), key=lambda s: s.oversubscription)
    assert worst.profile.name.startswith("SAE L4")
    assert worst.oversubscription == pytest.approx(48.0, abs=0.05)
    assert len(worst.stages_over()) == 8
    assert worst.tops == pytest.approx(25.06, abs=0.005)


def test_the_cheapest_profile_needs_no_accelerator(workload):
    """The other end of the 500x span: an edge device doing event detection
    demands 0.05 TOP/s and is not oversubscribed."""
    got = workload.summary(workload.profile("Event detection & classification"))
    assert got.tops == pytest.approx(0.05, abs=0.005)
    assert got.oversubscription < 1.0
    assert got.stages_over() == ()


# ---------------------------------------------------------------------------
# Structure the comparison work depends on
# ---------------------------------------------------------------------------


def test_stage_catalog_shape(workload):
    assert len(workload.stages) == 19
    assert sorted({s.pipeline_tier for s in workload.stages.values()}) == [
        f"T{i}" for i in range(1, 8)
    ]
    for stage in workload.stages.values():
        assert sum(stage.class_split) == pytest.approx(1.0)
        assert stage.basis, f"{stage.key} has no derivation basis"
        assert stage.unit


def test_most_of_the_pipeline_is_below_fifteen_op_per_byte(workload):
    """The argument document's central memory claim, on the annex's 19
    stages: 12 sit below 15 op per byte, where the memory system sets the
    rate however wide the multiplier array is. Detection is the outlier a
    dense accelerator can actually feed."""
    below = [s.key for s in workload.stages.values() if s.op_per_byte < 15]
    assert len(below) == 12
    assert workload.stages["det"].op_per_byte > 500
    assert "det" not in below


def test_precision_floors_are_what_the_classes_say(workload):
    """Class C work -- factor graphs, signed-distance fields, MPC, the
    barrier filter -- has an FP32/FP64 floor. A die without FP32 cannot run
    these at all, which is the finding the die comparison turns on."""
    floors = {k: s.precision_floor for k, s in workload.stages.items()}
    assert floors["mpc"] == floors["cbf"] == floors["ba"] == "C"
    assert floors["det"] == "B"  # INT8 trunk, FP16 heads
    class_c = [k for k, v in floors.items() if v == "C"]
    assert len(class_c) == 14  # 14 of 19 stages cannot run without FP32/FP64


def test_effective_throughput_is_a_parameter_not_a_constant(workload):
    """The annex's sensitivity: tripling the Class B and C figures takes
    air-superiority oversubscription from ~16.6 to ~9, so it reduces the
    deficit without removing it. The port must take the throughputs as
    inputs for that to be checkable at all."""
    profile = workload.profile("air superiority")
    base = workload.summary(profile).oversubscription
    tripled = PipelineWorkload(
        stages=workload.stages,
        profiles=workload.profiles,
        throughput=EffectiveThroughput(a=2000.0, b=900.0, c=45.0),
    )
    relaxed = tripled.summary(profile).oversubscription
    assert base == pytest.approx(16.6, abs=0.05)
    assert relaxed == pytest.approx(9.06, abs=0.05)
    assert relaxed < base and relaxed > 1.0


def test_a_class_split_must_be_a_split():
    with pytest.raises(ValueError, match="sums to"):
        Stage(
            key="x", name="x", pipeline_tier="T1", unit="per call",
            ops_per_call=1.0, bytes_per_call=1.0, class_split=(0.5, 0.2, 0.0),
        )


def test_round_trip_through_yaml(workload, tmp_path):
    path = tmp_path / "workload.yaml"
    workload.save(path)
    again = PipelineWorkload.load(path)
    assert again.to_dict() == workload.to_dict()
    assert again.summary(again.profile("far flight")).tops == pytest.approx(
        workload.summary(workload.profile("far flight")).tops
    )
