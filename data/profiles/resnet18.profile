====================================================================================================
HIERARCHICAL MODULE TABLE: resnet18
====================================================================================================

[1/3] Tracing with PyTorch FX...
[2/3] Running graph partitioner...
[3/3] Formatting hierarchical table...

====================================================================================================
GRAPH PROFILE
====================================================================================================

| Module                         | #Parameters          | MACs         | FLOPs        | Memory       |
|:-------------------------------|:---------------------|:-------------|:-------------|:-------------|
| model                          | 11.690M              | 1.814G       | 3.644G       | 116.95MB     |
|  conv1                         | 9.408K               | 118.014M     | 236.028M     | 3.85MB       |
|  bn1                           | 128                  |              | 4.014M       | 6.42MB       |
|  relu                          |                      |              | 802.816K     | 6.42MB       |
|  maxpool                       |                      |              |              | 4.01MB       |
|  layer1                        | 147.968K             | 462.422M     | 929.661M     | 19.86MB      |
|   layer1.0                     | 73.984K              | 231.211M     | 464.830M     | 9.93MB       |
|    conv1                       | 36.864K              | 115.606M     | 231.211M     | 1.75MB       |
|    bn1                         | 128                  |              | 1.004M       | 1.61MB       |
|    relu                        |                      |              | 200.704K     | 1.61MB       |
|    conv2                       | 36.864K              | 115.606M     | 231.211M     | 1.75MB       |
|    bn2                         | 128                  |              | 1.004M       | 1.61MB       |
|    relu_1                      |                      |              | 200.704K     | 1.61MB       |
|   layer1.1                     | 73.984K              | 231.211M     | 464.830M     | 9.93MB       |
|    conv1                       | 36.864K              | 115.606M     | 231.211M     | 1.75MB       |
|    bn1                         | 128                  |              | 1.004M       | 1.61MB       |
|    relu                        |                      |              | 200.704K     | 1.61MB       |
|    conv2                       | 36.864K              | 115.606M     | 231.211M     | 1.75MB       |
|    bn2                         | 128                  |              | 1.004M       | 1.61MB       |
|    relu_1                      |                      |              | 200.704K     | 1.61MB       |
|  add                           |                      |              | 200.704K     | 2.41MB       |
|  add_1                         |                      |              | 200.704K     | 2.41MB       |
|  layer2                        | 525.568K             | 411.042M     | 824.994M     | 14.15MB      |
|   layer2.0                     | 230.144K             | 179.831M     | 361.368M     | 8.15MB       |
|    conv1                       | 73.728K              | 57.803M      | 115.606M     | 1.50MB       |
|    bn1                         | 256                  |              | 501.760K     | 804.86KB     |
|    relu                        |                      |              | 100.352K     | 802.82KB     |
|    conv2                       | 147.456K             | 115.606M     | 231.211M     | 1.39MB       |
|    bn2                         | 256                  |              | 501.760K     | 804.86KB     |
|    downsample                  | 8.448K               | 6.423M       | 13.347M      | 2.04MB       |
|     downsample.0               | 8.192K               | 6.423M       | 12.845M      | 1.24MB       |
|     downsample.1               | 256                  |              | 501.760K     | 804.86KB     |
|    relu_1                      |                      |              | 100.352K     | 802.82KB     |
|   layer2.1                     | 295.424K             | 231.211M     | 463.626M     | 6.00MB       |
|    conv1                       | 147.456K             | 115.606M     | 231.211M     | 1.39MB       |
|    bn1                         | 256                  |              | 501.760K     | 804.86KB     |
|    relu                        |                      |              | 100.352K     | 802.82KB     |
|    conv2                       | 147.456K             | 115.606M     | 231.211M     | 1.39MB       |
|    bn2                         | 256                  |              | 501.760K     | 804.86KB     |
|    relu_1                      |                      |              | 100.352K     | 802.82KB     |
|  add_2                         |                      |              | 100.352K     | 1.20MB       |
|  add_3                         |                      |              | 100.352K     | 1.20MB       |
|  layer3                        | 2.100M               | 411.042M     | 823.539M     | 14.43MB      |
|   layer3.0                     | 919.040K             | 179.831M     | 360.515M     | 7.29MB       |
|    conv1                       | 294.912K             | 57.803M      | 115.606M     | 1.78MB       |
|    bn1                         | 512                  |              | 250.880K     | 405.50KB     |
|    relu                        |                      |              | 50.176K      | 401.41KB     |
|    conv2                       | 589.824K             | 115.606M     | 231.211M     | 2.76MB       |
|    bn2                         | 512                  |              | 250.880K     | 405.50KB     |
|    downsample                  | 33.280K              | 6.423M       | 13.096M      | 1.14MB       |
|     downsample.0               | 32.768K              | 6.423M       | 12.845M      | 733.18KB     |
|     downsample.1               | 512                  |              | 250.880K     | 405.50KB     |
|    relu_1                      |                      |              | 50.176K      | 401.41KB     |
|   layer3.1                     | 1.181M               | 231.211M     | 463.024M     | 7.14MB       |
|    conv1                       | 589.824K             | 115.606M     | 231.211M     | 2.76MB       |
|    bn1                         | 512                  |              | 250.880K     | 405.50KB     |
|    relu                        |                      |              | 50.176K      | 401.41KB     |
|    conv2                       | 589.824K             | 115.606M     | 231.211M     | 2.76MB       |
|    bn2                         | 512                  |              | 250.880K     | 405.50KB     |
|    relu_1                      |                      |              | 50.176K      | 401.41KB     |
|  add_4                         |                      |              | 50.176K      | 602.11KB     |
|  add_5                         |                      |              | 50.176K      | 602.11KB     |
|  layer4                        | 8.394M               | 411.042M     | 822.811M     | 36.61MB      |
|   layer4.0                     | 3.673M               | 179.831M     | 360.088M     | 16.51MB      |
|    conv1                       | 1.180M               | 57.803M      | 115.606M     | 5.02MB       |
|    bn1                         | 1.024K               |              | 125.440K     | 208.90KB     |
|    relu                        |                      |              | 25.088K      | 200.70KB     |
|    conv2                       | 2.359M               | 115.606M     | 231.211M     | 9.64MB       |
|    bn2                         | 1.024K               |              | 125.440K     | 208.90KB     |
|    downsample                  | 132.096K             | 6.423M       | 12.970M      | 1.03MB       |
|     downsample.0               | 131.072K             | 6.423M       | 12.845M      | 825.34KB     |
|     downsample.1               | 1.024K               |              | 125.440K     | 208.90KB     |
|    relu_1                      |                      |              | 25.088K      | 200.70KB     |
|   layer4.1                     | 4.721M               | 231.211M     | 462.723M     | 20.09MB      |
|    conv1                       | 2.359M               | 115.606M     | 231.211M     | 9.64MB       |
|    bn1                         | 1.024K               |              | 125.440K     | 208.90KB     |
|    relu                        |                      |              | 25.088K      | 200.70KB     |
|    conv2                       | 2.359M               | 115.606M     | 231.211M     | 9.64MB       |
|    bn2                         | 1.024K               |              | 125.440K     | 208.90KB     |
|    relu_1                      |                      |              | 25.088K      | 200.70KB     |
|  add_6                         |                      |              | 25.088K      | 301.06KB     |
|  add_7                         |                      |              | 25.088K      | 301.06KB     |
|  avgpool                       |                      |              |              | 102.40KB     |
|  fc                            | 513.000K             | 512.000K     | 1.024M       | 2.06MB       |

Compute Metrics:
  - Conv2d/Linear: MACs (multiply-accumulate operations)
  - BatchNorm: 5 FLOPs/element (normalize + scale + shift)
  - ReLU: 1 FLOP/element (max(0,x) comparison)
  - Add/Mul/Sub/Div: 1 FLOP/element (elementwise operation)
  - MaxPool/AdaptiveAvgPool: 0 FLOPs (comparison-based, matches fvcore)

Shape Information (shown with --showshape):
  - Parameters: Learnable weights/biases (e.g., conv.weight shape)
  - Tensor Shape: Output tensor dimensions during forward pass (e.g., [1, 64, 56, 56])
  - Operations without parameters (ReLU, MaxPool) only show Tensor Shape

====================================================================================================
SUMMARY
====================================================================================================

Total parameters: 11.69M (11,689,512)
Total FLOPs: 3.644 GFLOPs (3,643,625,984)
Total MACs: 1.814 GMACs (1,814,073,344)
Total memory: 116.95 MB (116,950,592 bytes)

Subgraphs: 68
Average AI: 24.76 FLOPs/byte
