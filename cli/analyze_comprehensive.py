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
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision


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

    # Output configuration
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (format auto-detected from extension)')
    parser.add_argument('--format', '-f',
                       choices=['text', 'json', 'csv', 'markdown', 'html'],
                       help='Output format (auto-detected if --output provided)')

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
            if format_type == 'json':
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

            if format_type == 'json':
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
