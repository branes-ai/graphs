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
    assert len(tools) == 7
    names = {t["name"] for t in tools}
    assert names == {
        "analyze_model",
        "estimate_latency",
        "estimate_energy",
        "estimate_memory",
        "compare_hardware",
        "list_hardware",
        "get_hardware_specs",
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
