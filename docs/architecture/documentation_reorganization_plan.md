# Documentation Reorganization Plan

> Plan to reorganize docs/ according to target architecture categories.

## Target Directory Structure

```
docs/
+-- architecture/           # System design (how it works internally)
+-- guides/                 # User-facing how-to documentation
+-- reference/              # API reference, technical specifications
+-- hardware/               # Hardware specifications and mappers
+-- validation/             # Accuracy reports and test results
+-- archive/
    +-- sessions/           # Development session logs
    +-- legacy/             # Superseded designs, completed work
```

## File Moves

### To architecture/

| Current Path | New Path |
|--------------|----------|
| characterization-architecture.md | architecture/characterization-architecture.md |
| computational-graphs.md | architecture/computational-graphs.md |
| operator_level_EDP.md | architecture/operator_level_EDP.md |
| performance_modeling_plan.md | architecture/performance_modeling_plan.md |
| embodied_ai_research_plan.md | architecture/embodied_ai_research_plan.md |
| enhanced_attention_fusion_plan.md | architecture/enhanced_attention_fusion_plan.md |
| analysis/bert-analysis.md | architecture/bert-analysis.md |
| analysis/infrastructure_analysis.md | architecture/infrastructure_analysis.md |
| analysis/kpu_energy_modeling_plan.md | architecture/kpu_energy_modeling_plan.md |
| analysis/subgraph_unification_analysis.md | architecture/subgraph_unification_analysis.md |
| analysis/tiled-matmul-analysis.md | architecture/tiled-matmul-analysis.md |
| analysis/transformer_attention_batching_analysis.md | architecture/transformer_attention_batching_analysis.md |
| designs/analyze_graph_mapping.md | architecture/analyze_graph_mapping.md |
| designs/bottleneck_operators.md | architecture/bottleneck_operators.md |
| designs/energy_model_architecture.md | architecture/energy_model_architecture.md |
| designs/functional_energy_composition.md | architecture/functional_energy_composition.md |
| designs/graph_partitioning_design.md | architecture/graph_partitioning_design.md |
| designs/hierarchical_edp_breakdown.md | architecture/hierarchical_edp_breakdown.md |
| hardware_db/cache_hierarchy_design.md | architecture/cache_hierarchy_design.md |
| hardware_db/gpu_core_clusters.md | architecture/gpu_core_clusters.md |
| hardware_db/heterogeneous_cores.md | architecture/heterogeneous_cores.md |
| hardware_db/memory_subsystem_design.md | architecture/memory_subsystem_design.md |
| hardware_db/phase1.md | architecture/hardware_db_phase1.md |
| hardware_db/unified_core_clusters_architecture.md | architecture/unified_core_clusters_architecture.md |
| plans/calibration_database.md | architecture/calibration_database_plan.md |
| plans/cuda_vs_tensor_core_separation.md | architecture/cuda_vs_tensor_core_separation.md |
| plans/edp_architectural_energy_plan.md | architecture/edp_architectural_energy_plan.md |
| plans/fusion_algorithm_proposal.md | architecture/fusion_algorithm_proposal.md |
| plans/fusion_viz_plan.md | architecture/fusion_viz_plan.md |
| plans/graph-analysis-tool-extraction.md | architecture/graph_analysis_tool_extraction.md |
| plans/multi_precision_calibration.md | architecture/multi_precision_calibration.md |
| plans/operand_fetch_energy_model.md | architecture/operand_fetch_energy_model.md |
| plans/realistic_tdp_model.md | architecture/realistic_tdp_model.md |
| plans/tpu_tile_energy_model_proposal.md | architecture/tpu_tile_energy_model_proposal.md |

### To guides/

| Current Path | New Path |
|--------------|----------|
| getting_started.md | guides/getting_started.md |
| guided_tour.md | guides/guided_tour.md |
| graph_partitioner_tutorial.md | guides/graph_partitioner_tutorial.md |
| graph_profiler_model_names_guide.md | guides/graph_profiler_model_names_guide.md |
| CHANGELOG_MANAGEMENT.md | guides/changelog_management.md |
| ci_workflow.md | guides/ci_workflow.md |
| FRAMEWORK_SEPARATED_CALIBRATION.md | guides/framework_separated_calibration.md |
| hardware_database_workflow.md | guides/hardware_database_workflow.md |
| mermaid_visualization_demo.md | guides/mermaid_visualization_demo.md |
| migration_guide_phase4_2.md | guides/migration_guide_phase4_2.md |
| mlir-serialization.md | guides/mlir_serialization.md |
| model_registry.md | guides/model_registry.md |
| post_silicon_improvement_tracking.md | guides/post_silicon_improvement_tracking.md |
| quick_start_diagramming.md | guides/quick_start_diagramming.md |
| tflite-converter-cli.md | guides/tflite_converter_cli.md |
| tflite-converter-guide.md | guides/tflite_converter_guide.md |
| transformer_support.md | guides/transformer_support.md |
| visualization_guide.md | guides/visualization_guide.md |
| YOLO_FX_COMMUNITY_OUTREACH_GUIDE.md | guides/yolo_fx_community_outreach.md |
| YOLO_FX_GITHUB_ISSUE_TEMPLATE.md | guides/yolo_fx_github_issue_template.md |
| calibration_framework_summary.md | guides/calibration_framework_summary.md |
| design/post_silicon_dynamics_tracking.md | guides/post_silicon_dynamics_tracking.md |
| designs/mermaid_visualization_design.md | guides/mermaid_visualization_design.md |
| hardware/README.md | guides/hardware_documentation_index.md |
| hardware_db/cache_levels_implementation_status.md | guides/cache_levels_implementation_status.md |
| hardware_db/implementation.md | guides/hardware_db_implementation.md |
| hardware_db/phase2_detection.md | guides/hardware_db_phase2_detection.md |
| hardware_db/phase3_management_tools.md | guides/hardware_db_phase3_management_tools.md |
| hardware_db/phase4_calibration_integration.md | guides/hardware_db_phase4_calibration_integration.md |
| sessions/README.md | guides/sessions_guide.md |
| sessions/template.md | guides/session_template.md |

### To reference/

| Current Path | New Path |
|--------------|----------|
| complexity.md | reference/complexity.md |
| domain-flow-vs-polyhedral.md | reference/domain_flow_vs_polyhedral.md |
| file_path_reference.md | reference/file_path_reference.md |
| graph_profiler.md | reference/graph_profiler.md |
| im2col.md | reference/im2col.md |
| matmul-tiling.md | reference/matmul_tiling.md |
| model_comparison_metrics.md | reference/model_comparison_metrics.md |
| operator_support.md | reference/operator_support.md |
| stablehlo-architecture.md | reference/stablehlo_architecture.md |
| summary.md | reference/summary.md |
| tensor_shape_analysis.md | reference/tensor_shape_analysis.md |
| tflite-matmul-concept.md | reference/tflite_matmul_concept.md |
| torch-compile-limitations.md | reference/torch_compile_limitations.md |
| unified_framework_api.md | reference/unified_framework_api.md |
| unified_profiler.md | reference/unified_profiler.md |
| YOLO_FX_TRACING_ISSUES.md | reference/yolo_fx_tracing_issues.md |
| chip_area_estimates.md | reference/chip_area_estimates.md |
| computational-spacetime-1994-Omtzigt-Physics-of-Computation.md | reference/computational_spacetime_1994.md |
| hardware/architecture_taxonomy.md | reference/architecture_taxonomy.md |
| hardware/efficacy_metrics.md | reference/efficacy_metrics.md |
| hardware/tensor_unit_analysis.md | reference/tensor_unit_analysis.md |
| hardware_db/memory_architecture_comparison.md | reference/memory_architecture_comparison.md |

### To hardware/

| Current Path | New Path |
|--------------|----------|
| ampere_ampereone_mapper.md | hardware/ampere_ampereone_mapper.md |
| datacenter_cpu_comparison.md | hardware/datacenter_cpu_comparison.md |
| dsp_npu_mappers.md | hardware/dsp_npu_mappers.md |
| edge_ai_categories.md | hardware/edge_ai_categories.md |
| hailo_edge_ai.md | hardware/hailo_edge_ai.md |
| i7_12700K_mapper_variants.md | hardware/i7_12700K_mapper_variants.md |
| ip_core_comparison.md | hardware/ip_core_comparison.md |
| kpu_architecture.md | hardware/kpu_architecture.md |
| TENSOR_CORE_PERFORMANCE.md | hardware/tensor_core_performance.md |
| Xilinx_Vitis_AI_spec.md | hardware/xilinx_vitis_ai_spec.md |
| analysis/dpu_and_cgra_analysis.md | hardware/dpu_and_cgra_analysis.md |
| analysis/embodied_ai_hardware_bom_summary.md | hardware/embodied_ai_hardware_bom_summary.md |
| hardware/architectural_energy.md | hardware/architectural_energy.md (keep) |
| hardware/jetson_specifications.md | hardware/jetson_specifications.md (keep) |
| hardware/microarchitectural_specs.md | hardware/microarchitectural_specs.md (keep) |
| hardware/nvidia_gpu_memory_hierarchy_official.md | hardware/nvidia_gpu_memory_hierarchy_official.md (keep) |

### To validation/

| Current Path | New Path |
|--------------|----------|
| graph_partitioner_validation.md | validation/graph_partitioner_validation.md |
| hardware_characterization_2025-10.md | validation/hardware_characterization_2025-10.md |
| investigation_summary.md | validation/investigation_summary.md |
| runtime-comparisons.md | validation/runtime_comparisons.md |
| test_architecture_comparison.md | validation/test_architecture_comparison.md |
| test_fx_graph.md | validation/test_fx_graph.md |
| test_hardware_mapping_h100.md | validation/test_hardware_mapping_h100.md |
| test_hardware_mapping_tpu.md | validation/test_hardware_mapping_tpu.md |
| test_partitioned_bottleneck.md | validation/test_partitioned_bottleneck.md |
| test_partitioned_optype.md | validation/test_partitioned_optype.md |
| analysis/tdp_model_vs_reality_rca.md | validation/tdp_model_vs_reality_rca.md |
| analysis/test_bottleneck_analysis.md | validation/test_bottleneck_analysis.md |
| bugs/bug_fixes_summary.md | validation/bug_fixes_summary.md |
| bugs/flop_mac_validation.md | validation/flop_mac_validation.md |
| bugs/tpu/* | validation/tpu/ (all TPU RCA files) |
| hardware/tensor_core_counting_bug_rca.md | validation/tensor_core_counting_bug_rca.md |
| hardware/tpu_bandwidth_fix.md | validation/tpu_bandwidth_fix.md |
| results/fusion_results.md | validation/fusion_results.md |
| results/fusion_test_results.md | validation/fusion_test_results.md |
| validation/* | validation/ (keep existing) |

### To archive/sessions/

All dated session files from sessions/ move to archive/sessions/:
- sessions/2025-10-*.md -> archive/sessions/
- sessions/2025-11-*.md -> archive/sessions/
- sessions/2025-12-*.md -> archive/sessions/

### To archive/legacy/

| Current Path | New Path |
|--------------|----------|
| efficientnet_fusion_improvements.md | archive/legacy/efficientnet_fusion_improvements.md |
| enhanced_attention_fusion_project_complete.md | archive/legacy/enhanced_attention_fusion_complete.md |
| fingerprint-design.md | archive/legacy/fingerprint_design.md |
| phase2_automatic_decomposition_complete.md | archive/legacy/phase2_automatic_decomposition_complete.md |
| phase3_attention_fusion_patterns_complete.md | archive/legacy/phase3_attention_fusion_patterns_complete.md |
| phase4_2_complete.md | archive/legacy/phase4_2_complete.md |
| phase4_2_unified_workflow_plan.md | archive/legacy/phase4_2_unified_workflow_plan.md |
| phase4_comprehensive_validation_complete.md | archive/legacy/phase4_comprehensive_validation_complete.md |
| phase4_integration_plan.md | archive/legacy/phase4_integration_plan.md |
| reorganization_2025-10-22.md | archive/legacy/reorganization_2025-10-22.md |
| SE_block_fusion_improvement.md | archive/legacy/SE_block_fusion_improvement.md |
| trusted_advisor_plan.md | archive/legacy/trusted_advisor_plan.md |
| bugs/efficiency_factor_not_visible.md | archive/legacy/efficiency_factor_not_visible.md |
| bugs/pytest_warnings_fix.md | archive/legacy/pytest_warnings_fix.md |
| designs/energy_model_complete.md | archive/legacy/energy_model_complete.md |
| hardware/gpu_energy_model_update_complete.md | archive/legacy/gpu_energy_model_update_complete.md |
| hardware/multifabric_migration_complete.md | archive/legacy/multifabric_migration_complete.md |
| analysis/embodied_ai_market_analysis.md | archive/legacy/embodied_ai_market_analysis.md |
| Hardware_Utilization_Metrics_Analysis.md | archive/legacy/Hardware_Utilization_Metrics_Analysis.md |

## Files to Delete

| File | Reason |
|------|--------|
| matmul-tiling.md | Empty file (0 bytes) |

## Directories to Remove (after moves)

After all files are moved, these directories will be empty and can be removed:
- analysis/
- bugs/tpu/ (after moving to validation/tpu/)
- bugs/
- design/
- designs/
- hardware_db/
- plans/
- results/
- sessions/ (after moving dated files to archive/sessions/)

## JSON Example Files

The JSON example files in hardware_db/ are reference data files, not documentation:
- core_clusters_arm_example.json
- core_clusters_example.json
- gpu_*.json
- memory_subsystem_*.json

These should be moved to a data directory or kept with the code:
- Consider moving to: src/graphs/hardware/registry/examples/

---

## Execution Order

1. Create target directories
2. Move files to architecture/
3. Move files to guides/
4. Move files to reference/
5. Move files to hardware/
6. Move files to validation/
7. Move files to archive/sessions/
8. Move files to archive/legacy/
9. Remove empty directories
10. Handle JSON example files
11. Delete empty file

---

*Plan created: 2025-01-16*
