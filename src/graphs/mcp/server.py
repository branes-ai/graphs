"""MCP server exposing graphs quantitative estimators.

Follows the same pattern as the branes MCP server: plain dict-based tool
definitions with a JSON-string dispatcher.  No ``mcp`` pip package required.

Tools
-----
analyze_model      – Full unified roofline + energy + memory analysis
estimate_latency   – Roofline-based latency prediction
estimate_energy    – Component-wise energy breakdown
estimate_memory    – Peak memory and activation timeline
compare_hardware   – Multi-target performance ranking
list_hardware      – Hardware catalog discovery
get_hardware_specs – Detailed hardware profile
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast; only import heavy deps on first tool call
# ---------------------------------------------------------------------------

_analyzer = None
_registry = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from graphs.estimation.unified_analyzer import UnifiedAnalyzer

        _analyzer = UnifiedAnalyzer(verbose=False)
    return _analyzer


def _get_registry():
    global _registry
    if _registry is None:
        from graphs.hardware.registry import get_registry

        _registry = get_registry()
    return _registry


def _precision_enum(name: str):
    """Convert a precision string like 'fp16' to the Precision enum value."""
    # Use the canonical Precision so dict lookups against
    # HardwareResourceModel.precision_profiles succeed (issue #59).
    from graphs.hardware.resource_model import Precision

    mapping = {p.value: p for p in Precision}
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown precision '{name}'. Valid: {list(mapping.keys())}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _safe_value(v: Any) -> Any:
    """Recursively convert a value to a JSON-safe form."""
    if isinstance(v, dict):
        return {k: _safe_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_value(i) for i in v]
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if hasattr(v, "value"):  # enum
        return v.value
    if hasattr(v, "__dataclass_fields__"):
        return _safe_asdict(v)
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def _safe_asdict(obj: Any) -> Any:
    """Convert a dataclass to a JSON-safe dict."""
    try:
        d = asdict(obj)
    except TypeError:
        d = {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return {k: _safe_value(v) for k, v in d.items()}


def _result_summary(result) -> dict:
    """Extract the executive summary from a UnifiedAnalysisResult."""
    summary = result.get_executive_summary()

    if result.roofline_report is not None:
        rr = result.roofline_report
        summary["roofline"] = {
            "total_latency_ms": rr.total_latency * 1000,
            "compute_time_ms": rr.total_compute_time * 1000,
            "memory_time_ms": rr.total_memory_time * 1000,
            "avg_flops_utilization": rr.average_flops_utilization,
            "avg_bandwidth_utilization": rr.average_bandwidth_utilization,
            "num_compute_bound": rr.num_compute_bound,
            "num_memory_bound": rr.num_memory_bound,
        }
    if result.energy_report is not None:
        er = result.energy_report
        summary["energy"] = {
            "total_energy_mj": er.total_energy_mj,
            "compute_energy_j": er.compute_energy_j,
            "memory_energy_j": er.memory_energy_j,
            "static_energy_j": er.static_energy_j,
            "average_power_w": er.average_power_w,
            "efficiency": er.average_efficiency,
            "wasted_energy_pct": er.wasted_energy_percent,
        }
    if result.memory_report is not None:
        mr = result.memory_report
        summary["memory"] = {
            "peak_memory_mb": mr.peak_memory_mb,
            "activation_memory_bytes": mr.activation_memory_bytes,
            "weight_memory_bytes": mr.weight_memory_bytes,
            "workspace_memory_bytes": mr.workspace_memory_bytes,
            "fits_on_device": mr.fits_on_device,
        }
    return summary


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def get_mcp_tool_definitions() -> list[dict[str, Any]]:
    """Return MCP tool definitions for the graphs estimator engine."""
    return [
        {
            "name": "analyze_model",
            "description": (
                "Run unified roofline + energy + memory analysis for a model on "
                "target hardware.  Returns latency, energy, peak memory, bottleneck, "
                "utilisation, and confidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model identifier (e.g. 'yolov8n', 'resnet50')",
                    },
                    "hardware_name": {
                        "type": "string",
                        "description": (
                            "Hardware target (e.g. 'jetson_orin_nano', 'h100_sxm5')"
                        ),
                    },
                    "batch_size": {"type": "integer", "default": 1},
                    "precision": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "bf16", "int8", "int4"],
                        "default": "fp16",
                    },
                    "thermal_profile": {
                        "type": "string",
                        "description": "Thermal/power profile (e.g. '15W', '30W')",
                    },
                },
                "required": ["model_name", "hardware_name"],
            },
        },
        {
            "name": "estimate_latency",
            "description": (
                "Predict inference latency using the roofline model.  Returns "
                "compute vs memory time, bottleneck, per-subgraph breakdown, and "
                "confidence level (CALIBRATED / INTERPOLATED / THEORETICAL)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "hardware_name": {"type": "string"},
                    "batch_size": {"type": "integer", "default": 1},
                    "precision": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "bf16", "int8", "int4"],
                        "default": "fp16",
                    },
                    "thermal_profile": {"type": "string"},
                },
                "required": ["model_name", "hardware_name"],
            },
        },
        {
            "name": "estimate_energy",
            "description": (
                "Estimate energy consumption with compute / memory / static "
                "breakdown.  Supports power-gating and thermal-aware TDP."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "hardware_name": {"type": "string"},
                    "batch_size": {"type": "integer", "default": 1},
                    "precision": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "bf16", "int8", "int4"],
                        "default": "fp16",
                    },
                    "power_gating_enabled": {"type": "boolean", "default": False},
                    "thermal_profile": {"type": "string"},
                },
                "required": ["model_name", "hardware_name"],
            },
        },
        {
            "name": "estimate_memory",
            "description": (
                "Analyse peak memory usage, activation timeline, and reuse "
                "patterns.  Reports whether the model fits in on-device memory."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "hardware_name": {"type": "string"},
                    "batch_size": {"type": "integer", "default": 1},
                    "precision": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "bf16", "int8", "int4"],
                        "default": "fp16",
                    },
                },
                "required": ["model_name", "hardware_name"],
            },
        },
        {
            "name": "compare_hardware",
            "description": (
                "Compare a model's predicted performance across multiple hardware "
                "targets.  Returns a ranked table with latency, energy, memory, "
                "and utilisation per target."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "hardware_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Hardware IDs to compare",
                    },
                    "batch_size": {"type": "integer", "default": 1},
                    "precision": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "bf16", "int8", "int4"],
                        "default": "fp16",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["latency", "energy", "memory"],
                        "default": "latency",
                    },
                },
                "required": ["model_name", "hardware_list"],
            },
        },
        {
            "name": "list_hardware",
            "description": (
                "List available hardware targets, optionally filtered by device "
                "type or search query."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "device_type": {
                        "type": "string",
                        "enum": [
                            "cpu", "gpu", "dsp", "tpu", "kpu", "accelerator",
                        ],
                        "description": "Filter by device category",
                    },
                    "query": {
                        "type": "string",
                        "description": "Fuzzy search (e.g. 'jetson', 'orin')",
                    },
                },
            },
        },
        {
            "name": "get_hardware_specs",
            "description": (
                "Get detailed specifications for a hardware target: peak FLOPS "
                "by precision, bandwidth, memory, TDP, architecture, calibration."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "hardware_id": {"type": "string"},
                },
                "required": ["hardware_id"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _analyze_model(args: dict[str, Any]) -> str:
    from graphs.estimation.unified_analyzer import AnalysisConfig

    config = AnalysisConfig()
    result = _get_analyzer().analyze_model(
        model_name=args["model_name"],
        hardware_name=args["hardware_name"],
        batch_size=args.get("batch_size", 1),
        precision=_precision_enum(args.get("precision", "fp16")),
        config=config,
        thermal_profile=args.get("thermal_profile"),
    )
    return json.dumps(_result_summary(result), indent=2)


def _estimate_latency(args: dict[str, Any]) -> str:
    from graphs.estimation.unified_analyzer import AnalysisConfig

    config = AnalysisConfig()
    config.run_energy = False
    config.run_memory = False
    config.run_concurrency = False
    result = _get_analyzer().analyze_model(
        model_name=args["model_name"],
        hardware_name=args["hardware_name"],
        batch_size=args.get("batch_size", 1),
        precision=_precision_enum(args.get("precision", "fp16")),
        config=config,
        thermal_profile=args.get("thermal_profile"),
    )
    if result.roofline_report is None:
        return json.dumps({"error": "Roofline analysis failed"})
    report = _safe_asdict(result.roofline_report)
    # Trim to top-10 subgraphs by latency
    if "latencies" in report and isinstance(report["latencies"], list):
        report["latencies"] = sorted(
            report["latencies"],
            key=lambda d: d.get("actual_latency", 0),
            reverse=True,
        )[:10]
    return json.dumps(report, indent=2)


def _estimate_energy(args: dict[str, Any]) -> str:
    from graphs.estimation.unified_analyzer import AnalysisConfig

    config = AnalysisConfig()
    config.run_memory = False
    config.run_concurrency = False
    config.power_gating_enabled = args.get("power_gating_enabled", False)
    result = _get_analyzer().analyze_model(
        model_name=args["model_name"],
        hardware_name=args["hardware_name"],
        batch_size=args.get("batch_size", 1),
        precision=_precision_enum(args.get("precision", "fp16")),
        config=config,
        thermal_profile=args.get("thermal_profile"),
    )
    if result.energy_report is None:
        return json.dumps({"error": "Energy analysis failed"})
    report = _safe_asdict(result.energy_report)
    if "energy_descriptors" in report and isinstance(report["energy_descriptors"], list):
        report["energy_descriptors"] = report["energy_descriptors"][:10]
    return json.dumps(report, indent=2)


def _estimate_memory(args: dict[str, Any]) -> str:
    from graphs.estimation.unified_analyzer import AnalysisConfig

    config = AnalysisConfig()
    config.run_roofline = False
    config.run_energy = False
    config.run_concurrency = False
    result = _get_analyzer().analyze_model(
        model_name=args["model_name"],
        hardware_name=args["hardware_name"],
        batch_size=args.get("batch_size", 1),
        precision=_precision_enum(args.get("precision", "fp16")),
        config=config,
    )
    if result.memory_report is None:
        return json.dumps({"error": "Memory analysis failed"})
    report = _safe_asdict(result.memory_report)
    if "memory_timeline" in report and isinstance(report["memory_timeline"], list):
        report["memory_timeline"] = report["memory_timeline"][:20]
    if "subgraph_descriptors" in report and isinstance(report["subgraph_descriptors"], list):
        report["subgraph_descriptors"] = report["subgraph_descriptors"][:10]
    return json.dumps(report, indent=2)


def _compare_hardware(args: dict[str, Any]) -> str:
    from graphs.estimation.unified_analyzer import AnalysisConfig

    config = AnalysisConfig()
    prec = _precision_enum(args.get("precision", "fp16"))
    analyzer = _get_analyzer()

    rows: list[dict] = []
    errors: list[dict] = []
    for hw in args["hardware_list"]:
        try:
            result = analyzer.analyze_model(
                model_name=args["model_name"],
                hardware_name=hw,
                batch_size=args.get("batch_size", 1),
                precision=prec,
                config=config,
            )
            rows.append({
                "hardware": hw,
                "latency_ms": result.total_latency_ms,
                "throughput_fps": result.throughput_fps,
                "energy_mj": result.total_energy_mj,
                "peak_memory_mb": result.peak_memory_mb,
                "utilization_pct": result.average_utilization_pct,
            })
        except Exception as exc:
            errors.append({"hardware": hw, "error": str(exc)})

    sort_key = {
        "latency": "latency_ms",
        "energy": "energy_mj",
        "memory": "peak_memory_mb",
    }.get(args.get("sort_by", "latency"), "latency_ms")
    rows.sort(key=lambda r: r.get(sort_key, float("inf")))

    payload: dict[str, Any] = {
        "model": args["model_name"],
        "precision": args.get("precision", "fp16"),
        "results": rows,
    }
    if errors:
        payload["errors"] = errors
    return json.dumps(payload, indent=2)


def _list_hardware(args: dict[str, Any]) -> str:
    registry = _get_registry()
    query = args.get("query")
    device_type = args.get("device_type")

    if query:
        profiles = registry.search(query)
        ids = [p.id for p in profiles]
    elif device_type:
        ids = registry.list_by_type(device_type)
    else:
        ids = registry.list_all()

    items: list[dict] = []
    for hw_id in sorted(ids):
        profile = registry.get(hw_id)
        if profile is None:
            items.append({"id": hw_id})
            continue
        items.append({
            "id": hw_id,
            "vendor": profile.vendor,
            "model": profile.model,
            "device_type": profile.device_type,
            "tdp_watts": profile.tdp_watts,
            "memory_gb": profile.memory_gb,
            "calibrated": profile.is_calibrated,
        })
    return json.dumps(items, indent=2)


def _get_hardware_specs(args: dict[str, Any]) -> str:
    registry = _get_registry()
    profile = registry.get(args["hardware_id"])
    if profile is None:
        return json.dumps({
            "error": f"Hardware '{args['hardware_id']}' not found",
            "available": registry.list_all(),
        })

    specs: dict[str, Any] = {
        "id": profile.id,
        "vendor": profile.vendor,
        "model": profile.model,
        "device_type": profile.device_type,
        "architecture": profile.architecture,
        "compute_units": profile.compute_units,
        "memory_gb": profile.memory_gb,
        "tdp_watts": profile.tdp_watts,
        "peak_bandwidth_gbps": profile.peak_bandwidth_gbps,
        "theoretical_peaks_gflops": profile.theoretical_peaks,
        "base_clock_mhz": profile.base_clock_mhz,
        "boost_clock_mhz": profile.boost_clock_mhz,
        "l2_cache_size_mb": profile.l2_cache_size_mb,
        "l3_cache_size_mb": profile.l3_cache_size_mb,
        "tags": profile.tags,
        "product_category": profile.product_category,
        "calibrated": profile.is_calibrated,
        "effective_peak_gflops": profile.effective_peak_gflops,
        "effective_bandwidth_gbps": profile.effective_bandwidth_gbps,
    }
    if profile.power_profiles:
        specs["power_profiles"] = profile.power_profiles
    if profile.calibration:
        specs["calibration"] = {
            "date": profile.calibration_date,
            "best_measured_gflops": profile.calibration.best_measured_gflops,
            "measured_bandwidth_gbps": profile.calibration.measured_bandwidth_gbps,
        }
    return json.dumps(specs, indent=2)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {
    "analyze_model": _analyze_model,
    "estimate_latency": _estimate_latency,
    "estimate_energy": _estimate_energy,
    "estimate_memory": _estimate_memory,
    "compare_hardware": _compare_hardware,
    "list_hardware": _list_hardware,
    "get_hardware_specs": _get_hardware_specs,
}


def execute_mcp_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Execute an MCP tool and return a JSON result string."""
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return handler(args)
    except Exception as exc:
        logger.exception("Tool %s failed", tool_name)
        return json.dumps({"error": str(exc), "tool": tool_name})
