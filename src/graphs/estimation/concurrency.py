"""
Concurrency Analyzer: Analyzes available concurrency for performance validation.

This module analyzes concurrency at multiple levels:
- Graph-level: Which operations can run in parallel?
- Subgraph-level: How many threads within each operation?
- Critical path: What limits end-to-end latency?

The concurrency analysis enables validation of performance estimates.
"""

import networkx as nx
from typing import List, Dict
import numpy as np

from graphs.core.structures import (
    SubgraphDescriptor,
    SubgraphConcurrency,
    ConcurrencyDescriptor,
    PartitionReport,
    OperationType
)


class ConcurrencyAnalyzer:
    """Analyze available concurrency for performance validation"""

    def analyze(self, partition_report: PartitionReport) -> ConcurrencyDescriptor:
        """
        Analyze concurrency at both graph and subgraph levels

        Args:
            partition_report: Output from GraphPartitioner

        Returns:
            ConcurrencyDescriptor with detailed concurrency analysis
        """

        subgraphs = partition_report.subgraphs

        if not subgraphs:
            return ConcurrencyDescriptor(
                total_subgraphs=0,
                independent_subgraphs=0,
                sequential_subgraphs=0,
                critical_path_length=0,
                critical_path_flops=0,
                parallelizable_flops=0,
                batch_size=1,
                batch_parallelism=1,
                max_theoretical_speedup=1.0,
                num_stages=0,
                max_parallel_ops_per_stage=0
            )

        # Build dependency graph (should exist in partition report)
        dep_graph = self._build_dependency_graph(subgraphs)

        # Compute stages (groups of operations that can run in parallel)
        stages = self._compute_stages(dep_graph)

        # Find critical path
        critical_path_ids = self._find_critical_path(dep_graph, subgraphs)
        critical_path_subgraphs = [sg for sg in subgraphs if sg.node_id in critical_path_ids]

        # Analyze each subgraph's internal concurrency
        subgraph_concurrency = {}
        for sg in subgraphs:
            subgraph_concurrency[sg.node_id] = self._analyze_subgraph_concurrency(sg)

        # Graph-level statistics
        total_subgraphs = len(subgraphs)
        independent_count = sum(len(stage) for stage in stages if len(stage) > 1)
        sequential_count = sum(1 for stage in stages if len(stage) == 1)

        # Critical path stats
        critical_path_length = len(critical_path_subgraphs)
        critical_path_flops = sum(sg.flops for sg in critical_path_subgraphs)

        # Parallelizable FLOPs (sum of max FLOPs per stage)
        parallelizable_flops = sum(
            max((sg.flops for sg in subgraphs if sg.node_id in stage), default=0)
            for stage in stages
        )

        # Batch parallelism
        batch_size = subgraphs[0].parallelism.batch if subgraphs[0].parallelism else 1
        batch_parallelism = batch_size

        # Theoretical speedup
        max_parallel = max(len(stage) for stage in stages) if stages else 1
        max_theoretical_speedup = max_parallel * batch_size

        # Utilization metrics
        concurrency_utilization = self._compute_concurrency_utilization(stages, subgraphs)
        parallelism_efficiency = np.mean([sc.parallelism_efficiency
                                          for sc in subgraph_concurrency.values()])

        # Generate explanation
        explanation = self._explain_concurrency(stages, critical_path_subgraphs, subgraphs)

        return ConcurrencyDescriptor(
            total_subgraphs=total_subgraphs,
            independent_subgraphs=independent_count,
            sequential_subgraphs=sequential_count,
            critical_path_length=critical_path_length,
            critical_path_flops=critical_path_flops,
            parallelizable_flops=parallelizable_flops,
            batch_size=batch_size,
            batch_parallelism=batch_parallelism,
            max_theoretical_speedup=max_theoretical_speedup,
            num_stages=len(stages),
            max_parallel_ops_per_stage=max_parallel,
            stages=stages,
            subgraph_concurrency=subgraph_concurrency,
            concurrency_utilization=concurrency_utilization,
            parallelism_efficiency=parallelism_efficiency,
            explanation=explanation
        )

    def _build_dependency_graph(self, subgraphs: List[SubgraphDescriptor]) -> nx.DiGraph:
        """Build dependency graph from subgraphs"""
        G = nx.DiGraph()

        # Add nodes
        for sg in subgraphs:
            G.add_node(sg.node_id)

        # Add edges
        for sg in subgraphs:
            for dep_id in sg.depends_on:
                if dep_id in G.nodes:
                    G.add_edge(dep_id, sg.node_id)

        return G

    def _compute_stages(self, dep_graph: nx.DiGraph) -> List[List[str]]:
        """
        Compute stages: groups of nodes that can execute in parallel

        Uses topological sort to identify which nodes can run concurrently.
        """
        stages = []
        remaining_nodes = set(dep_graph.nodes())
        completed_nodes = set()

        while remaining_nodes:
            # Find nodes whose dependencies are all completed
            ready_nodes = []
            for node in remaining_nodes:
                deps = set(dep_graph.predecessors(node))
                if deps.issubset(completed_nodes):
                    ready_nodes.append(node)

            if not ready_nodes:
                # Circular dependency or isolated nodes
                ready_nodes = list(remaining_nodes)

            stages.append(ready_nodes)
            completed_nodes.update(ready_nodes)
            remaining_nodes -= set(ready_nodes)

        return stages

    def _find_critical_path(self, dep_graph: nx.DiGraph,
                           subgraphs: List[SubgraphDescriptor]) -> List[str]:
        """
        Find critical path (longest path by FLOPs)

        The critical path determines minimum latency.
        """
        # Create FLOP-weighted graph
        flop_map = {sg.node_id: sg.flops for sg in subgraphs}

        try:
            # Use longest path algorithm
            if nx.is_directed_acyclic_graph(dep_graph):
                path = nx.dag_longest_path(dep_graph, weight=lambda u, v, d: flop_map.get(v, 0))
                return path
            else:
                # Fallback: just return all nodes
                return list(dep_graph.nodes())
        except:
            # If algorithm fails, return all nodes
            return list(dep_graph.nodes())

    def _analyze_subgraph_concurrency(self, subgraph: SubgraphDescriptor) -> SubgraphConcurrency:
        """Analyze concurrency within a single subgraph"""

        if not subgraph.parallelism:
            return SubgraphConcurrency(
                total_threads=1,
                independent_threads=1,
                independent_operations=1,
                dependency_chains=1,
                min_hardware_units=1,
                optimal_hardware_units=1,
                max_hardware_units=1,
                parallelism_efficiency=1.0
            )

        p = subgraph.parallelism
        op_type = subgraph.operation_type

        if op_type in [OperationType.CONV2D, OperationType.CONV2D_POINTWISE]:
            # Standard convolution: highly parallel
            total_threads = p.total_threads
            independent_threads = total_threads  # each output pixel independent

            # Estimate optimal hardware units
            # Assume 256 threads per unit (typical for GPU blocks or CPU SIMD groups)
            optimal_units = max(1, min(128, total_threads // 256))

            return SubgraphConcurrency(
                total_threads=total_threads,
                independent_threads=independent_threads,
                independent_operations=total_threads,
                dependency_chains=1,  # no dependencies between outputs
                can_split_batch=True,
                can_split_spatial=True,
                can_split_channels=True,
                min_hardware_units=1,
                optimal_hardware_units=optimal_units,
                max_hardware_units=optimal_units * 2,
                parallelism_efficiency=0.9  # generally efficient
            )

        elif op_type == OperationType.CONV2D_DEPTHWISE:
            # Depthwise: limited channel parallelism
            total_threads = p.total_threads
            # Each channel processed separately, but spatial is parallel
            independent_threads = p.batch * p.spatial

            optimal_units = max(1, min(32, independent_threads // 256))

            return SubgraphConcurrency(
                total_threads=total_threads,
                independent_threads=independent_threads,
                independent_operations=independent_threads,
                dependency_chains=1,
                can_split_batch=True,
                can_split_spatial=True,
                can_split_channels=False,  # depthwise limitation
                min_hardware_units=1,
                optimal_hardware_units=optimal_units,
                max_hardware_units=optimal_units * 2,
                parallelism_efficiency=0.3  # limited by depthwise
            )

        elif op_type == OperationType.LINEAR:
            # Matrix multiplication: good parallelism
            total_threads = p.total_threads
            independent_threads = total_threads

            optimal_units = max(1, min(128, total_threads // 256))

            return SubgraphConcurrency(
                total_threads=total_threads,
                independent_threads=independent_threads,
                independent_operations=total_threads,
                dependency_chains=1,
                can_split_batch=True,
                can_split_spatial=False,
                can_split_channels=True,
                min_hardware_units=1,
                optimal_hardware_units=optimal_units,
                max_hardware_units=optimal_units * 2,
                parallelism_efficiency=0.85
            )

        elif op_type in [OperationType.RELU, OperationType.RELU6, OperationType.GELU,
                        OperationType.HARDSWISH]:
            # Elementwise: embarrassingly parallel
            total_threads = p.total_threads
            independent_threads = total_threads

            optimal_units = max(1, min(128, total_threads // 256))

            return SubgraphConcurrency(
                total_threads=total_threads,
                independent_threads=independent_threads,
                independent_operations=total_threads,
                dependency_chains=1,
                can_split_batch=True,
                can_split_spatial=True,
                can_split_channels=True,
                min_hardware_units=1,
                optimal_hardware_units=optimal_units,
                max_hardware_units=optimal_units * 2,
                parallelism_efficiency=0.95  # nearly perfect
            )

        else:
            # Default for unknown operations
            total_threads = p.total_threads
            optimal_units = max(1, min(32, total_threads // 256))

            return SubgraphConcurrency(
                total_threads=total_threads,
                independent_threads=total_threads // 2,  # conservative
                independent_operations=total_threads // 2,
                dependency_chains=2,
                min_hardware_units=1,
                optimal_hardware_units=optimal_units,
                max_hardware_units=optimal_units * 2,
                parallelism_efficiency=0.5  # conservative
            )

    def _compute_concurrency_utilization(self, stages: List[List[str]],
                                         subgraphs: List[SubgraphDescriptor]) -> float:
        """
        Compute how well we utilize available concurrency

        High utilization = many operations can run in parallel
        Low utilization = mostly sequential
        """
        if not stages:
            return 0.0

        total_ops = len(subgraphs)
        max_concurrent = max(len(stage) for stage in stages)

        # Average concurrent ops per stage
        avg_concurrent = sum(len(stage) for stage in stages) / len(stages)

        # Utilization = avg concurrent / max possible
        utilization = avg_concurrent / total_ops if total_ops > 0 else 0.0

        return min(1.0, utilization)

    def _explain_concurrency(self, stages: List[List[str]],
                            critical_path: List[SubgraphDescriptor],
                            all_subgraphs: List[SubgraphDescriptor]) -> str:
        """Generate human-readable concurrency explanation"""

        num_stages = len(stages)
        max_parallel = max(len(stage) for stage in stages) if stages else 0
        batch_size = critical_path[0].parallelism.batch if critical_path and critical_path[0].parallelism else 1

        # Thread-level parallelism range
        if all_subgraphs:
            thread_counts = [sg.parallelism.total_threads for sg in all_subgraphs
                           if sg.parallelism]
            if thread_counts:
                min_threads = min(thread_counts)
                max_threads = max(thread_counts)
                avg_threads = int(np.mean(thread_counts))
            else:
                min_threads = max_threads = avg_threads = 0
        else:
            min_threads = max_threads = avg_threads = 0

        explanation = f"""
Concurrency Analysis
====================

Graph-level Parallelism:
  Total stages: {num_stages}
  Max parallel ops per stage: {max_parallel}
  Critical path length: {len(critical_path)} sequential operations

Parallelism Potential:
  - Graph-level: {max_parallel}x (max ops that can run simultaneously)
  - Batch-level: {batch_size}x (independent samples in batch)
  - Thread-level: {min_threads:,} to {max_threads:,} threads per op (avg: {avg_threads:,})

Validation Checks:
  - With batch={batch_size}: Speedup limited to {max_parallel}x by graph structure
  - With batch=32: Could theoretically reach {max_parallel * 32}x speedup
  - Hardware with <{max_parallel} compute units will be under-utilized

Performance Implications:
  - Single-sample inference (batch=1) is inherently sequential
  - Batching is essential for high throughput on parallel hardware
  - {len([s for s in stages if len(s) == 1])} stages are sequential bottlenecks
"""

        return explanation
