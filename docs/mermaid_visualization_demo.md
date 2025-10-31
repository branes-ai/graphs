# Mermaid Visualization Demo: ResNet18

This document demonstrates the Mermaid visualization capabilities (Phases 1-3).

---

## 1. FX Graph Structure (Phase 1)

Shows the raw PyTorch FX graph structure with operation types.

```mermaid
graph TD
    N0[x]
    N1[conv1<br/>〈call_module〉]
    N2[bn1<br/>〈call_module〉]
    N3[relu<br/>〈call_module〉]
    N4[maxpool<br/>〈call_module〉]
    N5[layer1_0_conv1<br/>〈call_module〉]
    N6[layer1_0_bn1<br/>〈call_module〉]
    N7[layer1_0_relu<br/>〈call_module〉]
    N8[layer1_0_conv2<br/>〈call_module〉]
    N9[layer1_0_bn2<br/>〈call_module〉]
    N10[add<br/>〈call_function〉]
    N11[layer1_0_relu_1<br/>〈call_module〉]
    N12[layer1_1_conv1<br/>〈call_module〉]
    N13[layer1_1_bn1<br/>〈call_module〉]
    N14[layer1_1_relu<br/>〈call_module〉]
    Truncated[... 56 more nodes ...]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N4 --> N10
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
    N10 --> N11
    N11 --> N12
    N12 --> N13
    N13 --> N14
```

---

## 2. Partitioned Graph (Phases 1 & 2)

Shows fused subgraphs with bottleneck analysis.

```mermaid
graph TD

    Start([Input])
    style Start fill:#ADD8E6,stroke:#000080,stroke-width:3px

    subgraph SG0["Subgraph 0<br/>conv1 → bn1 → relu<br/>236.0M FLOPs, Compute-bound"]
        SG0_spacer[ ]
        SG0_spacer --> SG0_exec
        SG0_exec[236.0M FLOPs<br/>3.8 MB]
    end

    style SG0 fill:#228B22,stroke:#000000,stroke-width:2px
    style SG0_spacer fill:none,stroke:none
    subgraph SG1["Subgraph 1<br/>maxpool<br/>Memory-bound"]
        SG1_spacer[ ]
        SG1_spacer --> SG1_exec
        SG1_exec[4.0 MB]
    end

    style SG1 fill:#DC143C,stroke:#000000,stroke-width:2px
    style SG1_spacer fill:none,stroke:none
    subgraph SG2["Subgraph 2<br/>layer1_0_conv1 → layer1_0_bn1 → layer1_0_relu<br/>231.2M FLOPs, Balanced"]
        SG2_spacer[ ]
        SG2_spacer --> SG2_exec
        SG2_exec[231.2M FLOPs<br/>1.6 MB]
    end

    style SG2 fill:#FF8C00,stroke:#000000,stroke-width:2px
    style SG2_spacer fill:none,stroke:none
    subgraph SG3["Subgraph 3<br/>layer1_0_conv2 → layer1_0_bn2<br/>231.2M FLOPs, Compute-bound"]
        SG3_spacer[ ]
        SG3_spacer --> SG3_exec
        SG3_exec[231.2M FLOPs<br/>1.6 MB]
    end

    style SG3 fill:#228B22,stroke:#000000,stroke-width:2px
    style SG3_spacer fill:none,stroke:none
    subgraph SG4["Subgraph 4<br/>add → layer1_0_relu_1<br/>Memory-bound"]
        SG4_spacer[ ]
        SG4_spacer --> SG4_exec
        SG4_exec[2.4 MB]
    end

    style SG4 fill:#DC143C,stroke:#000000,stroke-width:2px
    style SG4_spacer fill:none,stroke:none
    subgraph SG5["Subgraph 5<br/>layer1_1_conv1 → layer1_1_bn1 → layer1_1_relu<br/>231.2M FLOPs, Balanced"]
        SG5_spacer[ ]
        SG5_spacer --> SG5_exec
        SG5_exec[231.2M FLOPs<br/>1.6 MB]
    end

    style SG5 fill:#FF8C00,stroke:#000000,stroke-width:2px
    style SG5_spacer fill:none,stroke:none
    subgraph SG6["Subgraph 6<br/>layer1_1_conv2 → layer1_1_bn2<br/>231.2M FLOPs, Compute-bound"]
        SG6_spacer[ ]
        SG6_spacer --> SG6_exec
        SG6_exec[231.2M FLOPs<br/>1.6 MB]
    end

    style SG6 fill:#228B22,stroke:#000000,stroke-width:2px
    style SG6_spacer fill:none,stroke:none
    subgraph SG7["Subgraph 7<br/>add_1 → layer1_1_relu_1<br/>Memory-bound"]
        SG7_spacer[ ]
        SG7_spacer --> SG7_exec
        SG7_exec[2.4 MB]
    end

    style SG7 fill:#DC143C,stroke:#000000,stroke-width:2px
    style SG7_spacer fill:none,stroke:none
    subgraph SG8["Subgraph 8<br/>layer2_0_conv1 → layer2_0_bn1 → layer2_0_relu<br/>115.6M FLOPs, Balanced"]
        SG8_spacer[ ]
        SG8_spacer --> SG8_exec
        SG8_exec[115.6M FLOPs<br/>1.2 MB]
    end

    style SG8 fill:#FF8C00,stroke:#000000,stroke-width:2px
    style SG8_spacer fill:none,stroke:none
    subgraph SG9["Subgraph 9<br/>layer2_0_conv2 → layer2_0_bn2<br/>231.2M FLOPs, Compute-bound"]
        SG9_spacer[ ]
        SG9_spacer --> SG9_exec
        SG9_exec[231.2M FLOPs<br/>802.8 KB]
    end

    style SG9 fill:#228B22,stroke:#000000,stroke-width:2px
    style SG9_spacer fill:none,stroke:none
    subgraph SG10["Subgraph 10<br/>layer2_0_downsample_0 → layer2_0_downsample_1<br/>12.8M FLOPs, Memory-bound"]
        SG10_spacer[ ]
        SG10_spacer --> SG10_exec
        SG10_exec[12.8M FLOPs<br/>1.2 MB]
    end

    style SG10 fill:#DC143C,stroke:#000000,stroke-width:2px
    style SG10_spacer fill:none,stroke:none
    subgraph SG11["Subgraph 11<br/>add_2 → layer2_0_relu_1<br/>Balanced"]
        SG11_spacer[ ]
        SG11_spacer --> SG11_exec
        SG11_exec[1.2 MB]
    end

    style SG11 fill:#FF8C00,stroke:#000000,stroke-width:2px
    style SG11_spacer fill:none,stroke:none
    Truncated[... 20 more subgraphs ...]
    style Truncated fill:#696969
    End([Output])
    style End fill:#228B22,stroke:#006400,stroke-width:3px

    Start --> SG0_exec
    SG0_exec --> SG1_exec
    SG1_exec --> SG2_exec
    SG2_exec --> SG3_exec
    SG3_exec --> SG4_exec
    SG4_exec --> SG5_exec
    SG5_exec --> SG6_exec
    SG6_exec --> SG7_exec
    SG7_exec --> SG8_exec
    SG8_exec --> SG9_exec
    SG9_exec --> SG10_exec
    SG10_exec --> SG11_exec
    SG11_exec --> End
```


**Legend** (High Contrast Colors):
- 🟢 **Forest Green**: Compute-bound (efficient use of compute resources)
- 🔴 **Crimson Red**: Memory-bound (bottlenecked by memory bandwidth)
- 🟠 **Dark Orange**: Balanced (mixed compute and memory bound)
- ⚫ **Dim Gray**: Unknown or idle

---

## 3. Hardware Mapping: H100 GPU (Phase 3)

Shows how subgraphs map to H100 GPU streaming multiprocessors.

```mermaid
graph TD

    HW[H100 GPU<br/>132 Compute Units]
    style HW fill:#87CEEB,stroke:#000080,stroke-width:4px

    SG0[Subgraph 0<br/>conv1 → bn1 → +1<br/>0.0% util<br/>0.100ms]
    style SG0 fill:#696969,stroke:#000000,stroke-width:2px
    SG1[Subgraph 1<br/>maxpool<br/>0.0% util<br/>0.200ms<br/>⚠️ Memory-bound]
    style SG1 fill:#696969,stroke:#000000,stroke-width:2px
    SG2[Subgraph 2<br/>layer1_0_conv1 → layer1_0_bn1 → +1<br/>0.0% util<br/>0.300ms]
    style SG2 fill:#696969,stroke:#000000,stroke-width:2px
    SG3[Subgraph 3<br/>layer1_0_conv2 → layer1_0_bn2<br/>0.0% util<br/>0.400ms]
    style SG3 fill:#696969,stroke:#000000,stroke-width:2px
    SG4[Subgraph 4<br/>add → layer1_0_relu_1<br/>0.0% util<br/>0.500ms<br/>⚠️ Memory-bound]
    style SG4 fill:#696969,stroke:#000000,stroke-width:2px
    SG5[Subgraph 5<br/>layer1_1_conv1 → layer1_1_bn1 → +1<br/>0.0% util<br/>0.060ms]
    style SG5 fill:#696969,stroke:#000000,stroke-width:2px
    SG6[Subgraph 6<br/>layer1_1_conv2 → layer1_1_bn2<br/>0.0% util<br/>0.070ms]
    style SG6 fill:#696969,stroke:#000000,stroke-width:2px
    SG7[Subgraph 7<br/>add_1 → layer1_1_relu_1<br/>0.0% util<br/>0.080ms<br/>⚠️ Memory-bound]
    style SG7 fill:#696969,stroke:#000000,stroke-width:2px
    SG8[Subgraph 8<br/>layer2_0_conv1 → layer2_0_bn1 → +1<br/>0.0% util<br/>0.090ms]
    style SG8 fill:#696969,stroke:#000000,stroke-width:2px
    SG9[Subgraph 9<br/>layer2_0_conv2 → layer2_0_bn2<br/>0.0% util<br/>0.100ms]
    style SG9 fill:#696969,stroke:#000000,stroke-width:2px
    SG10[Subgraph 10<br/>layer2_0_downsample_0 → layer2_0_downsample_1<br/>0.0% util<br/>0.110ms<br/>⚠️ Memory-bound]
    style SG10 fill:#696969,stroke:#000000,stroke-width:2px
    SG11[Subgraph 11<br/>add_2 → layer2_0_relu_1<br/>0.0% util<br/>0.120ms]
    style SG11 fill:#696969,stroke:#000000,stroke-width:2px

    Idle[IDLE RESOURCES<br/>132 units<br/>100.0% of hardware]
    style Idle fill:#DC143C,stroke:#8B0000,stroke-width:3px

    Truncated[... 20 more subgraphs ...]
    style Truncated fill:#696969

    HW --> SG0
    SG0 --> SG1
    SG1 --> SG2
    SG2 --> SG3
    SG3 --> SG4
    SG4 --> SG5
    SG5 --> SG6
    SG6 --> SG7
    SG7 --> SG8
    SG8 --> SG9
    SG9 --> SG10
    SG10 --> SG11
```


**Legend** (High Contrast Colors):
- 🟢 **Dark Green**: Very high utilization (>80%)
- 🟢 **Forest Green**: High utilization (60-80%)
- 🟠 **Dark Orange**: Medium utilization (40-60%)
- 🟠 **Orange**: Low utilization (20-40%)
- 🔴 **Crimson**: Very low utilization (<20%)
- ⚫ **Dim Gray**: Idle (0%)

---

## 4. Bottleneck Analysis (Phase 2)

Identifies operations that dominate execution time.

```mermaid
graph TD

    Start[Total Latency: 6.63ms]
    style Start fill:#87CEEB,stroke:#000080,stroke-width:3px

    SG0[Subgraph 0<br/>conv1<br/>0.10ms ~ 2%]
    style SG0 fill:#808080,stroke:#000000,stroke-width:1px
    SG1[Subgraph 1<br/>maxpool<br/>0.20ms ~ 3%]
    style SG1 fill:#808080,stroke:#000000,stroke-width:1px
    SG2[Subgraph 2<br/>layer1_0_conv1<br/>0.30ms ~ 5%]
    style SG2 fill:#808080,stroke:#000000,stroke-width:1px
    SG3[Subgraph 3<br/>layer1_0_conv2<br/>0.40ms ~ 6%]
    style SG3 fill:#808080,stroke:#000000,stroke-width:1px
    SG4[Subgraph 4<br/>add<br/>0.50ms ~ 8%]
    style SG4 fill:#808080,stroke:#000000,stroke-width:1px
    SG5[Subgraph 5<br/>layer1_1_conv1<br/>0.06ms ~ 1%]
    style SG5 fill:#808080,stroke:#000000,stroke-width:1px
    SG6[Subgraph 6<br/>layer1_1_conv2<br/>0.07ms ~ 1%]
    style SG6 fill:#808080,stroke:#000000,stroke-width:1px
    SG7[Subgraph 7<br/>add_1<br/>0.08ms ~ 1%]
    style SG7 fill:#808080,stroke:#000000,stroke-width:1px
    SG8[Subgraph 8<br/>layer2_0_conv1<br/>0.09ms ~ 1%]
    style SG8 fill:#808080,stroke:#000000,stroke-width:1px
    SG9[Subgraph 9<br/>layer2_0_conv2<br/>0.10ms ~ 2%]
    style SG9 fill:#808080,stroke:#000000,stroke-width:1px
    SG10[Subgraph 10<br/>layer2_0_downsample_0<br/>0.11ms ~ 2%]
    style SG10 fill:#808080,stroke:#000000,stroke-width:1px
    SG11[Subgraph 11<br/>add_2<br/>0.12ms ~ 2%]
    style SG11 fill:#808080,stroke:#000000,stroke-width:1px
    SG12[Subgraph 12<br/>layer2_1_conv1<br/>0.13ms ~ 2%]
    style SG12 fill:#808080,stroke:#000000,stroke-width:1px
    SG13[Subgraph 13<br/>layer2_1_conv2<br/>0.14ms ~ 2%]
    style SG13 fill:#808080,stroke:#000000,stroke-width:1px
    SG14[Subgraph 14<br/>add_3<br/>0.15ms ~ 2%]
    style SG14 fill:#808080,stroke:#000000,stroke-width:1px

    Start --> SG0
    SG0 --> SG1
    SG1 --> SG2
    SG2 --> SG3
    SG3 --> SG4
    SG4 --> SG5
    SG5 --> SG6
    SG6 --> SG7
    SG7 --> SG8
    SG8 --> SG9
    SG9 --> SG10
    SG10 --> SG11
    SG11 --> SG12
    SG12 --> SG13
    SG13 --> SG14
```


---

## Summary

This demo shows all visualization types implemented in Phases 1-3:

- ✅ **Phase 1**: FX graph and partitioned graph visualization
- ✅ **Phase 2**: Color schemes (bottleneck, utilization, op_type) and legends
- ✅ **Phase 3**: Hardware mapping with resource allocation

**Next Steps**: Test these visualizations with real analysis data from the unified analyzer.
