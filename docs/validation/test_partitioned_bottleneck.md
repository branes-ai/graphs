# ResNet18: Partitioned Graph (Bottleneck Analysis)

This diagram shows fused subgraphs colored by bottleneck type.

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
    subgraph SG4["Subgraph 4<br/>add → layer1_0_relu_1<br/>200.7K FLOPs, Memory-bound"]
        SG4_spacer[ ]
        SG4_spacer --> SG4_exec
        SG4_exec[200.7K FLOPs<br/>2.4 MB]
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
    subgraph SG7["Subgraph 7<br/>add_1 → layer1_1_relu_1<br/>200.7K FLOPs, Memory-bound"]
        SG7_spacer[ ]
        SG7_spacer --> SG7_exec
        SG7_exec[200.7K FLOPs<br/>2.4 MB]
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
    subgraph SG11["Subgraph 11<br/>add_2 → layer2_0_relu_1<br/>100.4K FLOPs, Balanced"]
        SG11_spacer[ ]
        SG11_spacer --> SG11_exec
        SG11_exec[100.4K FLOPs<br/>1.2 MB]
    end

    style SG11 fill:#FF8C00,stroke:#000000,stroke-width:2px
    style SG11_spacer fill:none,stroke:none
    subgraph SG12["Subgraph 12<br/>layer2_1_conv1 → layer2_1_bn1 → layer2_1_relu<br/>231.2M FLOPs, Compute-bound"]
        SG12_spacer[ ]
        SG12_spacer --> SG12_exec
        SG12_exec[231.2M FLOPs<br/>802.8 KB]
    end

    style SG12 fill:#228B22,stroke:#000000,stroke-width:2px
    style SG12_spacer fill:none,stroke:none
    subgraph SG13["Subgraph 13<br/>layer2_1_conv2 → layer2_1_bn2<br/>231.2M FLOPs, Memory-bound"]
        SG13_spacer[ ]
        SG13_spacer --> SG13_exec
        SG13_exec[231.2M FLOPs<br/>802.8 KB]
    end

    style SG13 fill:#DC143C,stroke:#000000,stroke-width:2px
    style SG13_spacer fill:none,stroke:none
    subgraph SG14["Subgraph 14<br/>add_3 → layer2_1_relu_1<br/>100.4K FLOPs, Balanced"]
        SG14_spacer[ ]
        SG14_spacer --> SG14_exec
        SG14_exec[100.4K FLOPs<br/>1.2 MB]
    end

    style SG14 fill:#FF8C00,stroke:#000000,stroke-width:2px
    style SG14_spacer fill:none,stroke:none
    Truncated[... 17 more subgraphs ...]
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
    SG11_exec --> SG12_exec
    SG12_exec --> SG13_exec
    SG13_exec --> SG14_exec
    SG14_exec --> End
```


**Legend** (High Contrast Colors):
- 🟢 **Forest Green**: Compute-bound (efficient use of compute resources)
- 🔴 **Crimson Red**: Memory-bound (bottlenecked by memory bandwidth)
- 🟠 **Dark Orange**: Balanced (mixed compute and memory bound)
- ⚫ **Dim Gray**: Unknown or idle
