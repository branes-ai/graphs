#!/usr/bin/env python
"""
Comprehensive Graph Analysis Tool (Refactored with Unified Framework)

All-in-one analysis tool for deep-dive neural network characterization.
Provides complete performance analysis using all Phase 3 analyzers:
- Roofline Model: Bottleneck analysis and latency estimation
- Energy Estimator: Power and energy consumption analysis
- Memory Estimator: Memory footprint and timeline analysis

Supports:
- Single model, single hardware (comprehensive analysis)
- Multiple output formats (text, JSON, markdown, CSV)
- Verdict-first output for agentic workflows (constraint checking)

Usage:
    # Comprehensive single-model analysis
    ./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-AGX

    # JSON output
    ./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-AGX \
        --output results.json

    # CSV output
    ./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-AGX \
        --output results.csv --format csv

    # Markdown report
    ./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-AGX \
        --output report.md --format markdown

    # Different precision
    ./cli/analyze_comprehensive.py --model resnet50 --hardware Jetson-Orin-AGX \
        --precision fp16 --batch-size 32

    # Verdict-first output with constraint checking
    ./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
        --check-latency 10.0 --format verdict

    # Check power budget
    ./cli/analyze_comprehensive.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
        --check-power 15.0 --format verdict

    # Check memory constraint
    ./cli/analyze_comprehensive.py --model resnet50 --hardware KPU-T256 \
        --check-memory 500 --format verdict
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision


def generate_verdict_output(result, constraint_metric=None, constraint_threshold=None):
    """Generate verdict-first JSON output for agentic workflows.

    Args:
        result: UnifiedAnalysisResult from analyzer
        constraint_metric: Optional metric to check ('latency', 'power', 'memory', 'energy')
        constraint_threshold: Required threshold for the constraint metric

    Returns:
        JSON string with verdict-first output
    """
    try:
        from graphs.adapters import convert_to_pydantic
        pydantic_result = convert_to_pydantic(
            result,
            constraint_metric=constraint_metric,
            constraint_threshold=constraint_threshold
        )
        return pydantic_result.model_dump_json(indent=2)
    except ImportError:
        # Fallback if embodied-schemas not installed
        return _generate_verdict_fallback(result, constraint_metric, constraint_threshold)


def _generate_verdict_fallback(result, constraint_metric=None, constraint_threshold=None):
    """Fallback verdict generation without embodied-schemas dependency."""
    verdict = "UNKNOWN"
    margin_pct = None
    constraint_actual = None
    summary = f"Analysis of {result.display_name} on {result.hardware_display_name}"
    suggestions = []

    if constraint_metric and constraint_threshold is not None:
        if constraint_metric == 'latency':
            constraint_actual = result.total_latency_ms
            if constraint_actual <= constraint_threshold:
                verdict = "PASS"
                margin_pct = ((constraint_threshold - constraint_actual) / constraint_threshold) * 100
                summary = f"Latency {constraint_actual:.2f}ms meets {constraint_threshold:.1f}ms target ({margin_pct:.0f}% headroom)"
            else:
                verdict = "FAIL"
                margin_pct = -((constraint_actual - constraint_threshold) / constraint_threshold) * 100
                summary = f"Latency {constraint_actual:.2f}ms exceeds {constraint_threshold:.1f}ms target by {abs(margin_pct):.0f}%"
                suggestions.append("Consider faster hardware or smaller model")
        elif constraint_metric == 'power':
            constraint_actual = result.energy_report.average_power_w
            if constraint_actual <= constraint_threshold:
                verdict = "PASS"
                margin_pct = ((constraint_threshold - constraint_actual) / constraint_threshold) * 100
                summary = f"Power {constraint_actual:.1f}W meets {constraint_threshold:.1f}W budget ({margin_pct:.0f}% headroom)"
            else:
                verdict = "FAIL"
                margin_pct = -((constraint_actual - constraint_threshold) / constraint_threshold) * 100
                summary = f"Power {constraint_actual:.1f}W exceeds {constraint_threshold:.1f}W budget by {abs(margin_pct):.0f}%"
                suggestions.append("Consider lower-power hardware or power gating")
        elif constraint_metric == 'memory':
            constraint_actual = result.peak_memory_mb
            if constraint_actual <= constraint_threshold:
                verdict = "PASS"
                margin_pct = ((constraint_threshold - constraint_actual) / constraint_threshold) * 100
                summary = f"Memory {constraint_actual:.1f}MB meets {constraint_threshold:.1f}MB limit ({margin_pct:.0f}% headroom)"
            else:
                verdict = "FAIL"
                margin_pct = -((constraint_actual - constraint_threshold) / constraint_threshold) * 100
                summary = f"Memory {constraint_actual:.1f}MB exceeds {constraint_threshold:.1f}MB limit by {abs(margin_pct):.0f}%"
                suggestions.append("Consider gradient checkpointing or smaller batch size")
        elif constraint_metric == 'energy':
            constraint_actual = result.energy_per_inference_mj
            if constraint_actual <= constraint_threshold:
                verdict = "PASS"
                margin_pct = ((constraint_threshold - constraint_actual) / constraint_threshold) * 100
                summary = f"Energy {constraint_actual:.1f}mJ meets {constraint_threshold:.1f}mJ limit ({margin_pct:.0f}% headroom)"
            else:
                verdict = "FAIL"
                margin_pct = -((constraint_actual - constraint_threshold) / constraint_threshold) * 100
                summary = f"Energy {constraint_actual:.1f}mJ exceeds {constraint_threshold:.1f}mJ limit by {abs(margin_pct):.0f}%"
                suggestions.append("Consider more efficient hardware or quantization")
    else:
        verdict = "PASS"
        summary = (
            f"{result.display_name} on {result.hardware_display_name}: "
            f"{result.total_latency_ms:.2f}ms latency, "
            f"{result.energy_per_inference_mj:.1f}mJ/inference"
        )

    output = {
        "verdict": verdict,
        "confidence": "medium",
        "summary": summary,
        "model_id": result.model_name,
        "hardware_id": result.hardware_name,
        "batch_size": result.batch_size,
        "precision": result.precision.name.lower(),
        "latency_ms": result.total_latency_ms,
        "throughput_fps": result.throughput_fps,
        "energy_per_inference_mj": result.energy_per_inference_mj,
        "peak_memory_mb": result.peak_memory_mb,
    }

    if constraint_metric:
        output["constraint"] = {
            "metric": constraint_metric,
            "threshold": constraint_threshold,
            "actual": constraint_actual,
            "margin_pct": margin_pct,
        }

    if suggestions:
        output["suggestions"] = suggestions

    return json.dumps(output, indent=2)


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive neural network analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis (text output)
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX

  # JSON output
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX --output results.json

  # CSV output
  %(prog)s --model mobilenet_v2 --hardware Jetson-Orin-Nano --output results.csv

  # Markdown report
  %(prog)s --model efficientnet_b0 --hardware KPU-T256 --output report.md

  # FP16 precision with batch size 32
  %(prog)s --model resnet50 --hardware Jetson-Orin-AGX --precision fp16 --batch-size 32

  # Enable power gating for accurate idle energy modeling
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX --power-gating

  # Compare with and without power gating
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX --output no_pg.json
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX --output with_pg.json --power-gating

  # Disable hardware mapping (fallback to thread-based estimation)
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX --no-hardware-mapping

  # Verdict-first output (for agentic workflows)
  %(prog)s --model resnet18 --hardware H100 --check-latency 10.0

  # Check power budget
  %(prog)s --model mobilenet_v2 --hardware Jetson-Orin-Nano --check-power 15.0

  # Check memory constraint
  %(prog)s --model resnet50 --hardware KPU-T256 --check-memory 500

  # Verdict format with explicit format flag
  %(prog)s --model resnet18 --hardware H100 --format verdict

Supported models:
  ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
  MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2
  VGG: vgg11, vgg16, vgg19
  ViT: vit_b_16

Supported hardware:
  GPUs: H100, A100, V100, Jetson-Orin-AGX, Jetson-Orin-Nano
  TPUs: TPU-v4, Coral-Edge-TPU
  KPUs: KPU-T768, KPU-T256, KPU-T64
  CPUs: EPYC, Xeon, Ampere-One, i7-12700K, Ryzen
  DSPs: QRB5165, TI-TDA4VM
  Accelerators: DPU, CGRA
        """
    )

    # Model and hardware
    parser.add_argument('--model', required=True,
                       help='Model to analyze (e.g., resnet18, mobilenet_v2)')
    parser.add_argument('--hardware', required=True,
                       choices=['H100', 'A100', 'V100', 'Jetson-Orin-AGX', 'Jetson-Orin-Nano',
                                'TPU-v4', 'Coral-Edge-TPU', 'KPU-T768', 'KPU-T256', 'KPU-T64',
                                'EPYC', 'Xeon', 'Ampere-One', 'i7-12700K', 'Ryzen',
                                'QRB5165', 'TI-TDA4VM', 'DPU', 'CGRA'],
                       help='Target hardware (e.g., KPU-T64, Jetson-Orin-AGX)')

    # Analysis configuration
    parser.add_argument('--precision', default='fp32',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision (default: fp32)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')

    # Analysis options
    parser.add_argument('--no-roofline', action='store_true',
                       help='Skip roofline analysis')
    parser.add_argument('--no-energy', action='store_true',
                       help='Skip energy analysis')
    parser.add_argument('--no-memory', action='store_true',
                       help='Skip memory analysis')
    parser.add_argument('--concurrency', action='store_true',
                       help='Run concurrency analysis (expensive)')

    # Power management options
    parser.add_argument('--power-gating', action='store_true',
                       help='Enable power gating modeling (unallocated units consume 0 idle power)')
    parser.add_argument('--no-hardware-mapping', action='store_true',
                       help='Disable hardware mapper integration (for comparison with old method)')

    # Constraint checking (verdict-first output)
    parser.add_argument('--check-latency', type=float, metavar='MS',
                       help='Check if latency is under target (in milliseconds)')
    parser.add_argument('--check-power', type=float, metavar='WATTS',
                       help='Check if average power is under budget (in watts)')
    parser.add_argument('--check-memory', type=float, metavar='MB',
                       help='Check if peak memory is under limit (in megabytes)')
    parser.add_argument('--check-energy', type=float, metavar='MJ',
                       help='Check if energy per inference is under limit (in millijoules)')

    # Output configuration
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (format auto-detected from extension)')
    parser.add_argument('--format', '-f',
                       choices=['text', 'json', 'csv', 'markdown', 'html', 'verdict'],
                       help='Output format (verdict: verdict-first JSON for agentic workflows)')

    # Report options
    parser.add_argument('--sections', nargs='+',
                       choices=['executive', 'performance', 'energy', 'memory', 'recommendations'],
                       help='Sections to include in text report (default: all)')
    parser.add_argument('--no-executive-summary', action='store_true',
                       help='Exclude executive summary from text report')
    parser.add_argument('--subgraph-details', action='store_true',
                       help='Include per-subgraph details in CSV output')

    # Diagram options (for markdown/HTML output)
    parser.add_argument('--include-diagrams', action='store_true',
                       help='Include Mermaid diagrams in markdown/HTML output')
    parser.add_argument('--diagram-types', nargs='+',
                       choices=['partitioned', 'bottleneck', 'hardware_mapping'],
                       help='Types of diagrams to include (default: partitioned bottleneck)')

    # Display options
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--style',
                       choices=['default', 'compact', 'detailed'],
                       default='default',
                       help='Report style (default: default)')

    args = parser.parse_args()

    # Parse precision
    precision_map = {
        'fp32': Precision.FP32,
        'fp16': Precision.FP16,
        'int8': Precision.INT8,
    }
    precision = precision_map[args.precision.lower()]

    # Create analysis configuration
    config = AnalysisConfig(
        run_roofline=not args.no_roofline,
        run_energy=not args.no_energy,
        run_memory=not args.no_memory,
        run_concurrency=args.concurrency,
        run_hardware_mapping=not args.no_hardware_mapping,  # NEW
        power_gating_enabled=args.power_gating,  # NEW
        use_fusion_partitioning=False,
        validate_consistency=True,
    )

    # Run analysis
    try:
        if not args.quiet:
            print(f"\n{'='*79}")
            print(f"{'COMPREHENSIVE ANALYSIS':^79}")
            print(f"{'='*79}")
            print(f"Model: {args.model}")
            print(f"Hardware: {args.hardware}")
            print(f"Precision: {precision.name}")
            print(f"Batch Size: {args.batch_size}")
            print(f"{'='*79}\n")

        analyzer = UnifiedAnalyzer(verbose=not args.quiet)
        result = analyzer.analyze_model(
            model_name=args.model,
            hardware_name=args.hardware,
            batch_size=args.batch_size,
            precision=precision,
            config=config
        )

    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Determine constraint (if any)
    constraint_metric = None
    constraint_threshold = None
    if args.check_latency is not None:
        constraint_metric = 'latency'
        constraint_threshold = args.check_latency
    elif args.check_power is not None:
        constraint_metric = 'power'
        constraint_threshold = args.check_power
    elif args.check_memory is not None:
        constraint_metric = 'memory'
        constraint_threshold = args.check_memory
    elif args.check_energy is not None:
        constraint_metric = 'energy'
        constraint_threshold = args.check_energy

    # Generate report
    try:
        generator = ReportGenerator(style=args.style)

        # Determine format
        if args.output:
            # Auto-detect or use explicit format
            if args.format:
                format_type = args.format
            else:
                # Auto-detect from extension
                ext = Path(args.output).suffix.lower()
                format_map = {
                    '.json': 'json',
                    '.csv': 'csv',
                    '.md': 'markdown',
                    '.txt': 'text',
                    '.html': 'html',
                }
                format_type = format_map.get(ext, 'text')

            # Generate and save report
            if format_type == 'verdict':
                content = generate_verdict_output(result, constraint_metric, constraint_threshold)
                with open(args.output, 'w') as f:
                    f.write(content)
            elif format_type == 'json':
                generator.save_report(result, args.output, format='json')
            elif format_type == 'csv':
                content = generator.generate_csv_report(
                    result,
                    include_subgraph_details=args.subgraph_details
                )
                with open(args.output, 'w') as f:
                    f.write(content)
            elif format_type == 'markdown':
                content = generator.generate_markdown_report(
                    result,
                    include_diagrams=args.include_diagrams,
                    diagram_types=args.diagram_types
                )
                with open(args.output, 'w') as f:
                    f.write(content)
            elif format_type == 'html':
                content = generator.generate_html_report(
                    result,
                    include_diagrams=args.include_diagrams or True,  # Default to True for HTML
                    diagram_types=args.diagram_types
                )
                with open(args.output, 'w') as f:
                    f.write(content)
            else:
                content = generator.generate_text_report(
                    result,
                    include_sections=args.sections,
                    show_executive_summary=not args.no_executive_summary
                )
                with open(args.output, 'w') as f:
                    f.write(content)

            if not args.quiet:
                print(f"\nReport saved to: {args.output}")

        else:
            # Print to stdout
            format_type = args.format or 'text'

            # Auto-switch to verdict format if constraint is specified
            if constraint_metric and format_type == 'text':
                format_type = 'verdict'

            if format_type == 'verdict':
                print(generate_verdict_output(result, constraint_metric, constraint_threshold))
            elif format_type == 'json':
                print(generator.generate_json_report(result))
            elif format_type == 'csv':
                print(generator.generate_csv_report(
                    result,
                    include_subgraph_details=args.subgraph_details
                ))
            elif format_type == 'markdown':
                print(generator.generate_markdown_report(
                    result,
                    include_diagrams=args.include_diagrams,
                    diagram_types=args.diagram_types
                ))
            elif format_type == 'html':
                print(generator.generate_html_report(
                    result,
                    include_diagrams=args.include_diagrams or True,  # Default to True for HTML
                    diagram_types=args.diagram_types
                ))
            else:
                print(generator.generate_text_report(
                    result,
                    include_sections=args.sections,
                    show_executive_summary=not args.no_executive_summary
                ))

    except Exception as e:
        print(f"\nError generating report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
