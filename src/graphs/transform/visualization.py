"""
Visualization utilities for graph partitioning.

Provides:
- Color coding for bottleneck types
- ASCII/Unicode fallback for terminal compatibility
- DOT/Graphviz export
"""

import sys
import os
from typing import Optional, Dict
from enum import Enum


class TerminalCapability(Enum):
    """Terminal display capabilities"""
    BASIC = "basic"  # ASCII only
    UTF8 = "utf8"    # UTF-8 box drawing
    COLOR = "color"  # ANSI colors
    TRUECOLOR = "truecolor"  # 24-bit color


class ANSIColor:
    """ANSI color codes for terminal output"""

    # Reset
    RESET = "\033[0m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"


class BoxChars:
    """Box drawing characters with UTF-8 and ASCII fallback"""

    # UTF-8 box drawing
    UTF8 = {
        'top_left': '┌',
        'top_right': '┓',
        'bottom_left': '└',
        'bottom_right': '┘',
        'horizontal': '─',
        'vertical': '│',
        'vertical_right': '├',
        'vertical_left': '┤',
        'horizontal_down': '┬',
        'horizontal_up': '┴',
        'cross': '┼',
        'heavy_horizontal': '━',
        'heavy_vertical': '┃',
    }

    # ASCII fallback
    ASCII = {
        'top_left': '+',
        'top_right': '+',
        'bottom_left': '+',
        'bottom_right': '+',
        'horizontal': '-',
        'vertical': '|',
        'vertical_right': '+',
        'vertical_left': '+',
        'horizontal_down': '+',
        'horizontal_up': '+',
        'cross': '+',
        'heavy_horizontal': '=',
        'heavy_vertical': '|',
    }


def detect_terminal_capability() -> TerminalCapability:
    """
    Detect terminal capabilities for optimal rendering.

    Returns:
        TerminalCapability indicating what the terminal supports
    """
    # Check if output is being redirected
    if not sys.stdout.isatty():
        return TerminalCapability.BASIC

    # Check for NO_COLOR environment variable
    if os.environ.get('NO_COLOR'):
        return TerminalCapability.UTF8

    # Check TERM environment variable
    term = os.environ.get('TERM', '').lower()

    # Check for truecolor support
    if 'truecolor' in term or '24bit' in term:
        return TerminalCapability.TRUECOLOR

    # Check for color support
    if 'color' in term or 'ansi' in term or 'xterm' in term:
        return TerminalCapability.COLOR

    # Check encoding
    encoding = sys.stdout.encoding or ''
    if 'utf' in encoding.lower():
        return TerminalCapability.UTF8

    # Fallback to ASCII
    return TerminalCapability.BASIC


def get_box_chars(capability: TerminalCapability) -> Dict[str, str]:
    """Get appropriate box drawing characters for terminal"""
    if capability in [TerminalCapability.UTF8, TerminalCapability.COLOR, TerminalCapability.TRUECOLOR]:
        return BoxChars.UTF8
    return BoxChars.ASCII


def get_bottleneck_color(bottleneck_type: str, capability: TerminalCapability) -> tuple:
    """
    Get color codes for bottleneck type.

    Args:
        bottleneck_type: Type of bottleneck (compute_bound, memory_bound, etc.)
        capability: Terminal capability

    Returns:
        Tuple of (color_start, color_end) codes
    """
    if capability not in [TerminalCapability.COLOR, TerminalCapability.TRUECOLOR]:
        return ("", "")

    color_map = {
        'compute_bound': (ANSIColor.GREEN, ANSIColor.RESET),
        'memory_bound': (ANSIColor.YELLOW, ANSIColor.RESET),
        'bandwidth_bound': (ANSIColor.RED, ANSIColor.RESET),
        'balanced': (ANSIColor.CYAN, ANSIColor.RESET),
        'unknown': (ANSIColor.BRIGHT_BLACK, ANSIColor.RESET),
    }

    return color_map.get(bottleneck_type.lower(), ("", ""))


def colorize(text: str, color: str, capability: TerminalCapability) -> str:
    """
    Colorize text if terminal supports it.

    Args:
        text: Text to colorize
        color: ANSI color code
        capability: Terminal capability

    Returns:
        Colorized text or plain text if not supported
    """
    if capability not in [TerminalCapability.COLOR, TerminalCapability.TRUECOLOR]:
        return text

    return f"{color}{text}{ANSIColor.RESET}"


def create_legend(capability: TerminalCapability) -> str:
    """
    Create a color legend for bottleneck types.

    Args:
        capability: Terminal capability

    Returns:
        Formatted legend string
    """
    box = get_box_chars(capability)

    lines = []
    lines.append(box['horizontal'] * 60)
    lines.append("LEGEND: Bottleneck Types")
    lines.append(box['horizontal'] * 60)

    bottleneck_types = [
        ('compute_bound', 'Compute-bound (good for GPU/TPU)'),
        ('balanced', 'Balanced (good utilization)'),
        ('memory_bound', 'Memory-bound (may need fusion)'),
        ('bandwidth_bound', 'Bandwidth-bound (memory bandwidth limited)'),
    ]

    for btype, description in bottleneck_types:
        color_start, color_end = get_bottleneck_color(btype, capability)
        label = f"{color_start}{btype.upper()}{color_end}"
        lines.append(f"  {label:<30} - {description}")

    lines.append(box['horizontal'] * 60)

    return "\n".join(lines)


def export_to_dot(fused_subgraphs, fx_graph, output_file: str):
    """
    Export fusion graph to DOT format for Graphviz visualization.

    Args:
        fused_subgraphs: List of FusedSubgraph objects
        fx_graph: Original FX graph
        output_file: Output .dot file path

    Example:
        export_to_dot(report.fused_subgraphs, fx_graph, "fusion.dot")
        # Then: dot -Tpng fusion.dot -o fusion.png
    """
    lines = []

    # Graph header
    lines.append('digraph FusionGraph {')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box, style=rounded];')
    lines.append('    edge [color=gray];')
    lines.append('')

    # Define color scheme
    color_map = {
        'compute_bound': '#90EE90',      # Light green
        'memory_bound': '#FFD700',       # Gold
        'bandwidth_bound': '#FF6B6B',    # Light red
        'balanced': '#87CEEB',           # Sky blue
        'unknown': '#D3D3D3',            # Light gray
    }

    # Create nodes for each fused subgraph
    for sg in fused_subgraphs:
        sg_id = f"sg_{sg.subgraph_id}"

        # Choose color based on bottleneck
        bottleneck = sg.recommended_bottleneck.value
        color = color_map.get(bottleneck, color_map['unknown'])

        # Create label with metrics
        label_lines = [
            f"Subgraph {sg.subgraph_id}",
            f"Pattern: {sg.fusion_pattern[:30]}",
            f"Ops: {sg.num_operators}",
            f"FLOPs: {sg.total_flops / 1e9:.2f}G",
            f"AI: {sg.arithmetic_intensity:.1f}",
            f"Type: {bottleneck}",
        ]
        label = "\\n".join(label_lines)

        lines.append(f'    {sg_id} [label="{label}", fillcolor="{color}", style="filled,rounded"];')

    lines.append('')

    # Create edges based on data dependencies
    # Build node-to-subgraph mapping
    node_to_sg = {}
    for sg in fused_subgraphs:
        for node_id in sg.node_ids:
            node_to_sg[node_id] = sg

    # Track edges to avoid duplicates
    edges_seen = set()

    for node in fx_graph.graph.nodes:
        if node.name not in node_to_sg:
            continue

        current_sg = node_to_sg[node.name]

        # Check dependencies
        for input_node in node.all_input_nodes:
            if input_node.name in node_to_sg:
                producer_sg = node_to_sg[input_node.name]

                # Only create edge if different subgraphs
                if producer_sg.subgraph_id != current_sg.subgraph_id:
                    edge = (producer_sg.subgraph_id, current_sg.subgraph_id)

                    if edge not in edges_seen:
                        edges_seen.add(edge)
                        lines.append(f'    sg_{producer_sg.subgraph_id} -> sg_{current_sg.subgraph_id};')

    lines.append('')

    # Add legend as a subgraph
    lines.append('    subgraph cluster_legend {')
    lines.append('        label="Legend";')
    lines.append('        style=dashed;')
    lines.append('        legend [shape=none, label=<')
    lines.append('            <table border="0" cellborder="1" cellspacing="0">')
    lines.append('                <tr><td colspan="2"><b>Bottleneck Types</b></td></tr>')
    for btype, color in color_map.items():
        if btype != 'unknown':
            lines.append(f'                <tr><td bgcolor="{color}">{btype}</td><td align="left">   </td></tr>')
    lines.append('            </table>')
    lines.append('        >];')
    lines.append('    }')

    lines.append('}')

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Exported DOT graph to: {output_file}")
    print(f"To generate PNG: dot -Tpng {output_file} -o {output_file.replace('.dot', '.png')}")
    print(f"To generate SVG: dot -Tsvg {output_file} -o {output_file.replace('.dot', '.svg')}")


def format_metric_with_color(metric_name: str, value: float, threshold_good: float,
                             threshold_bad: float, capability: TerminalCapability,
                             unit: str = "", higher_is_better: bool = True) -> str:
    """
    Format a metric with color coding based on thresholds.

    Args:
        metric_name: Name of the metric
        value: Metric value
        threshold_good: Threshold for good performance
        threshold_bad: Threshold for bad performance
        capability: Terminal capability
        unit: Unit string (e.g., "GFLOPs", "%")
        higher_is_better: Whether higher values are better

    Returns:
        Formatted and colored metric string
    """
    if capability not in [TerminalCapability.COLOR, TerminalCapability.TRUECOLOR]:
        return f"{metric_name}: {value:.2f}{unit}"

    # Determine color based on thresholds
    if higher_is_better:
        if value >= threshold_good:
            color = ANSIColor.GREEN
        elif value >= threshold_bad:
            color = ANSIColor.YELLOW
        else:
            color = ANSIColor.RED
    else:
        if value <= threshold_good:
            color = ANSIColor.GREEN
        elif value <= threshold_bad:
            color = ANSIColor.YELLOW
        else:
            color = ANSIColor.RED

    return f"{metric_name}: {color}{value:.2f}{unit}{ANSIColor.RESET}"
