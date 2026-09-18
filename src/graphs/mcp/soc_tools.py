"""MCP tools for the SoC study framework (graphs#269 PR 4.4).

Three tools for the Embodied-AI-Architect orchestrator, with the same
dict-definition / JSON-string-handler shape as ``server.py``:

list_soc_designs  -- designs, nodes, efficiency tables, studies and profiles
analyze_soc       -- one design on one mission profile (SoCAnalyzer)
sweep_soc_study   -- a study through the analyzer, with bound-aware reports

**Every figure travels with its bound.** The analyzer never fills a gap, so a
die area, power or oversubscription over an input with gaps is a lower
bound, feasibility can be ``null`` (open), and useful TOPS/W can be withheld.
The tool descriptions say so, because an agent reading ``die_area_mm2: 9.9``
without ``area_is_lower_bound: true`` would draw the wrong conclusion.
``confidence_summary.limited_by`` names each gap.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

_soc_analyzer = None

_BOUNDS_NOTE = (
    "Figures computed over inputs with gaps are LOWER BOUNDS (area, power, "
    "oversubscription) or UPPER BOUNDS (TOPS/W) and are flagged as such; "
    "feasible is true only when proven, false on a proven violation, and null "
    "when gaps leave it open. Nothing is estimated to fill a gap -- read "
    "confidence_summary.limited_by before using a number."
)


def _analyzer():
    global _soc_analyzer
    if _soc_analyzer is None:
        from graphs.estimation.soc import SoCAnalyzer

        _soc_analyzer = SoCAnalyzer()
    return _soc_analyzer


def soc_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "list_soc_designs",
            "description": (
                "List what the SoC analyzer can be asked about: SoC designs (with "
                "their default node and whether their silicon is fully priced), "
                "process nodes, efficiency tables, shipped studies, and the "
                "workload's mission profiles and named regimes."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "analyze_soc",
            "description": (
                "Run a pipeline mission profile on an SoC design at a process node: "
                "per-stage service time and constraint ratio, engine and DRAM "
                "utilization, sense-to-act latency vs deadline, power vs budget, "
                "feasibility and confidence. " + _BOUNDS_NOTE
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "design": {"type": "string", "description": "Design id, e.g. 'orin_class_reference'"},
                    "profile": {
                        "type": "string",
                        "description": "Profile id, regime ('far flight', 'air superiority') or mission name",
                    },
                    "node": {"type": "string", "description": "Process node id; default the design's own"},
                    "efficiency": {
                        "type": "string",
                        "default": "annex_v1",
                        "description": "annex_v1 (the Data Annex pooled model) or default_v1 "
                                       "(per engine, measured pairs only)",
                    },
                    "mapping": {
                        "type": "string",
                        "default": "auto",
                        "description": "auto, explicit or greedy",
                    },
                    "gate_idle": {"type": "boolean", "default": False},
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "default": "summary",
                        "description": "summary: verdicts, bounds, memory, power totals and the "
                                       "stages that are over or unpriced; full: the whole result",
                    },
                },
                "required": ["design", "profile"],
            },
        },
        {
            "name": "sweep_soc_study",
            "description": (
                "Sweep SoC designs across nodes, profiles, efficiency tables and "
                "allocation overrides, either a shipped study by id or an ad-hoc "
                "one. Optionally classify points on a bound-aware Pareto front "
                "(front / dominated / undecided -- a lower bound never lands on a "
                "front) and report the smallest design proven feasible in every "
                "profile. " + _BOUNDS_NOTE
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "study": {"type": "string", "description": "Shipped study id; or give designs"},
                    "designs": {"type": "array", "items": {"type": "string"}},
                    "nodes": {"type": "array", "items": {"type": "string"}},
                    "profiles": {
                        "description": "'all', 'regimes' (default) or a list of profiles",
                        "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                    },
                    "efficiency": {"type": "array", "items": {"type": "string"}},
                    "pareto": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["area", "power", "oversubscription"]},
                    },
                    "union": {"type": "boolean", "default": False},
                },
            },
        },
    ]


def _list_soc_designs(args: Dict[str, Any]) -> str:
    from graphs.estimation.soc.study import DEFAULT_STUDY_DIR, load_study
    from graphs.hardware.soc import compose_soc

    a = _analyzer()
    designs = []
    for design_id, design in sorted(a.designs.items()):
        soc = compose_soc(design, a.library, a.nodes)
        designs.append({"id": design_id, "name": design.name, "process_node": design.process_node,
                        "silicon_complete": soc.complete, "unanchored_lines": len(soc.gaps)})
    studies = []
    for file in sorted(DEFAULT_STUDY_DIR.glob("*.yaml")):
        study = load_study(file)
        studies.append({"id": study.id, "name": study.name})
    tables = [{"id": t.id, "name": t.name, "kind": t.kind} for t in a.tables.values()]
    profiles = [{"id": p.id, "regime": p.regime, "power_budget_w": p.power_budget_w,
                 "deadline_ms": p.deadline_ms} for p in a.workload.profiles]
    return json.dumps({"designs": designs, "nodes": sorted(a.nodes), "efficiency_tables": tables,
                       "studies": studies, "workload": a.workload.version, "profiles": profiles},
                      indent=2)


def _summary(full: Dict[str, Any]) -> Dict[str, Any]:
    over = set(full["summary"]["stages_over"])
    return {
        "design": full["design"], "node": full["node"], "profile": full["profile"],
        "regime": full["regime"], "efficiency": full["efficiency"], "mapping": full["mapping"],
        "complete": full["complete"],
        "confidence_summary": full["confidence_summary"],
        "summary": full["summary"],
        "die": {k: full["die"][k] for k in ("area_mm2", "area_is_lower_bound", "transistors_b")},
        "memory": full["memory"],
        "power": {k: full["power"][k] for k in ("total_w", "total_is_lower_bound", "budget_w",
                                                  "within_budget", "useful_tops_per_w",
                                                  "useful_tops_per_w_is_upper_bound",
                                                  "confidence", "confidence_source")},
        "stages_of_note": [s for s in full["stages"] if s["stage"] in over or s["gap"]],
    }


def _analyze_soc(args: Dict[str, Any]) -> str:
    result = _analyzer().analyze(
        args["design"], args["profile"], node=args.get("node"),
        efficiency=args.get("efficiency", "annex_v1"), mapping=args.get("mapping", "auto"),
        gate_idle=bool(args.get("gate_idle", False)),
    )
    full = result.to_dict()
    detail = args.get("detail", "summary")
    if detail not in ("summary", "full"):
        raise ValueError(f"detail must be 'summary' or 'full', got {detail!r}")
    return json.dumps(full if detail == "full" else _summary(full), indent=2)


def _sweep_soc_study(args: Dict[str, Any]) -> str:
    from graphs.estimation.soc.pareto import classify_front, union_of_regimes
    from graphs.estimation.soc.study import Study, load_study, run_study

    a = _analyzer()
    if args.get("study"):
        study = load_study(args["study"])
    elif args.get("designs"):
        study = Study(id="adhoc", name="ad-hoc sweep", workload=a.workload.version,
                      designs=args["designs"], nodes=args.get("nodes") or [None],
                      profiles=args.get("profiles", "regimes"),
                      efficiency=args.get("efficiency") or ["annex_v1"])
    else:
        raise ValueError("give either 'study' or 'designs'")
    rows = run_study(study, a)
    flat = [r.to_row() for r in rows]
    payload: Dict[str, Any] = {"study": study.id, "points": flat}
    pareto: Optional[List[str]] = args.get("pareto")
    if pareto:
        for row, (_, status) in zip(flat, classify_front(rows, pareto)):
            row["pareto"] = status
        payload["pareto_metrics"] = pareto
    if args.get("union"):
        payload["union_of_regimes"] = union_of_regimes(rows).to_dict()
    return json.dumps(payload, indent=2)


SOC_HANDLERS = {
    "list_soc_designs": _list_soc_designs,
    "analyze_soc": _analyze_soc,
    "sweep_soc_study": _sweep_soc_study,
}

__all__ = ["SOC_HANDLERS", "soc_tool_definitions"]
