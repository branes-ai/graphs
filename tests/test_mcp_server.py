"""Tests for the graphs MCP server tool definitions and dispatch."""

import json
import os

import pytest

from graphs.mcp.server import (
    execute_mcp_tool,
    get_mcp_tool_definitions,
    _precision_enum,
)


def test_tool_definitions_valid():
    """All tool definitions have required fields and valid JSON schemas."""
    tools = get_mcp_tool_definitions()
    assert len(tools) == 10
    names = {t["name"] for t in tools}
    assert names == {
        "analyze_model",
        "estimate_latency",
        "estimate_energy",
        "estimate_memory",
        "compare_hardware",
        "list_hardware",
        "get_hardware_specs",
        "list_soc_designs",
        "analyze_soc",
        "sweep_soc_study",
    }
    for tool in tools:
        assert tool["description"], f"{tool['name']} missing description"
        assert tool["input_schema"], f"{tool['name']} missing input_schema"
        assert tool["input_schema"]["type"] == "object"


def test_precision_enum_mapping():
    """Precision string-to-enum mapping covers all advertised values."""
    for name in ("fp32", "fp16", "bf16", "int8", "int4"):
        p = _precision_enum(name)
        assert p.value == name

    with pytest.raises(ValueError, match="Unknown precision"):
        _precision_enum("fp128")


def test_list_hardware():
    """list_hardware returns a JSON array of hardware IDs."""
    result = execute_mcp_tool("list_hardware", {})
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) > 0
    assert all("id" in item for item in data)


def test_list_hardware_by_type():
    """list_hardware filters by device type."""
    result = execute_mcp_tool("list_hardware", {"device_type": "gpu"})
    data = json.loads(result)
    assert isinstance(data, list)
    for item in data:
        if "device_type" in item:
            assert item["device_type"] == "gpu"


def test_get_hardware_specs_not_found():
    """get_hardware_specs returns error for unknown hardware."""
    result = execute_mcp_tool("get_hardware_specs", {"hardware_id": "nonexistent_hw_xyz"})
    data = json.loads(result)
    assert "error" in data


def test_unknown_tool():
    """Unknown tool names return an error."""
    result = execute_mcp_tool("nonexistent_tool", {})
    data = json.loads(result)
    assert "error" in data


def test_auth_disabled_without_token():
    """Auth raises RuntimeError when GRAPHS_MCP_TOKEN is unset."""
    old = os.environ.pop("GRAPHS_MCP_TOKEN", None)
    try:
        from graphs.mcp.auth import get_token

        with pytest.raises(RuntimeError, match="GRAPHS_MCP_TOKEN"):
            get_token()
    finally:
        if old is not None:
            os.environ["GRAPHS_MCP_TOKEN"] = old


def test_auth_validates_token():
    """Auth validates Bearer tokens correctly."""
    os.environ["GRAPHS_MCP_TOKEN"] = "test-secret-token"
    try:
        from graphs.mcp.auth import validate_bearer

        assert validate_bearer("Bearer test-secret-token") is True
        assert validate_bearer("Bearer wrong-token") is False
        assert validate_bearer("Basic dXNlcjpwYXNz") is False
        assert validate_bearer("") is False
    finally:
        del os.environ["GRAPHS_MCP_TOKEN"]


# ---------------------------------------------------------------------------
# SoC tools (graphs#269 PR 4.4)
# ---------------------------------------------------------------------------


def test_soc_tool_descriptions_warn_about_bounds():
    """An agent must not read a lower bound as a value: the analyze and sweep
    tools say how bounds and open feasibility are reported."""
    by_name = {t["name"]: t for t in get_mcp_tool_definitions()}
    for name in ("analyze_soc", "sweep_soc_study"):
        assert "LOWER BOUNDS" in by_name[name]["description"]
        assert "limited_by" in by_name[name]["description"]


def test_list_soc_designs():
    out = json.loads(execute_mcp_tool("list_soc_designs", {}))
    ids = {d["id"] for d in out["designs"]}
    assert {"orin_class_reference", "kpu_heterogeneous_h64"} <= ids
    assert all(d["silicon_complete"] is False for d in out["designs"])
    assert {"annex_v1", "default_v1"} <= {t["id"] for t in out["efficiency_tables"]}
    assert "orin_vs_kpu_heterogeneous" in {s["id"] for s in out["studies"]}
    assert {"far flight", "air superiority"} <= {p["regime"] for p in out["profiles"]}


def test_analyze_soc_summary_keeps_the_bounds():
    out = json.loads(execute_mcp_tool(
        "analyze_soc", {"design": "orin_class_reference", "profile": "far flight"}))
    assert out["die"]["area_is_lower_bound"] is True
    assert out["power"]["total_is_lower_bound"] is True
    assert out["power"]["useful_tops_per_w"] is None
    assert out["summary"]["feasible"] is False  # far flight's DRAM demand
    assert out["confidence_summary"]["limited_by"]
    assert {s["stage"] for s in out["stages_of_note"]} == set(out["summary"]["stages_over"])


def test_analyze_soc_full_is_the_whole_result():
    full = json.loads(execute_mcp_tool("analyze_soc", {
        "design": "orin_class_reference", "profile": "air superiority", "detail": "full"}))
    assert {"peak", "stages", "engines", "die", "power"} <= set(full)
    assert len(full["stages"]) == 16


def test_analyze_soc_errors_are_json():
    out = json.loads(execute_mcp_tool("analyze_soc", {"design": "nope", "profile": "far flight"}))
    assert out["tool"] == "analyze_soc" and "no SoC design" in out["error"]
    out = json.loads(execute_mcp_tool("analyze_soc", {
        "design": "orin_class_reference", "profile": "far flight", "detail": "brief"}))
    assert "detail" in out["error"]


def test_sweep_soc_study_with_reports():
    out = json.loads(execute_mcp_tool("sweep_soc_study", {
        "study": "orin_node_scaling", "pareto": ["area", "power"], "union": True}))
    assert len(out["points"]) == 12
    assert {p["pareto"] for p in out["points"]} == {"undecided"}
    assert out["union_of_regimes"]["minimum"] is None


def test_sweep_soc_study_ad_hoc_and_errors():
    out = json.loads(execute_mcp_tool("sweep_soc_study", {
        "designs": ["kpu_heterogeneous_h64"], "nodes": ["tsmc_n7"], "profiles": ["far flight"]}))
    assert len(out["points"]) == 1 and out["points"][0]["feasible"] is False
    assert "error" in json.loads(execute_mcp_tool("sweep_soc_study", {}))
