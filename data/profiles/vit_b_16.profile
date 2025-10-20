====================================================================================================
HIERARCHICAL MODULE TABLE: vit_b_16
====================================================================================================

[1/3] Tracing with PyTorch FX...
[2/3] Running graph partitioner...
WARNING: Conv2d produced invalid output dimensions:
  Node: conv_proj
  Input shape: torch.Size([1, 768, 14, 14])
  H=14, W=14, K_h=16, K_w=16, S_h=16, S_w=16, P=0
  H_out=0, W_out=0
  Using H_out=W_out=1 as fallback
[3/3] Formatting hierarchical table...

====================================================================================================
GRAPH PROFILE
====================================================================================================

| Module                              | #Parameters          | Tensor Shape         | MACs         | FLOPs        | Memory       |
|:------------------------------------|:---------------------|:---------------------|:-------------|:-------------|:-------------|
| model                               | 86.568M              |                      | 11.271G      | 22.553G      | 549.84MB     |
|  conv_proj                          | 590.592K             | (1, 768, 14, 14)     | 115.606M     | 231.211M     | 3.57MB       |
|   conv_proj.weight                  |                      | (768, 3, 16, 16)     |              |              |              |
|   conv_proj.bias                    |                      | (768,)               |              |              |              |
|  add                                |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  encoder                            | 85.056M              |                      | 11.155G      | 22.317G      | 497.80MB     |
|   dropout                           |                      | (1, 197, 768)        |              |              | 1.21MB       |
|   layers                            | 85.054M              |                      | 11.155G      | 22.317G      | 495.38MB     |
|    encoder_layer_0                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_1                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_2                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_3                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_4                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_5                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_6                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_7                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_8                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_9                  | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_10                 | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|    encoder_layer_11                 | 7.088M               |                      | 929.563M     | 1.860G       | 41.28MB      |
|     ln_1                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_1.weight                    |                      | (768,)               |              |              |              |
|      ln_1.bias                      |                      | (768,)               |              |              |              |
|     self_attention                  | 2.362M               |                      |              |              | 1.82MB       |
|      self_attention.in_proj_weight  |                      | (2304, 768)          |              |              |              |
|      self_attention.in_proj_bias    |                      | (2304,)              |              |              |              |
|      self_attention.out_proj.weight |                      | (768, 768)           |              |              |              |
|      self_attention.out_proj.bias   |                      | (768,)               |              |              |              |
|     dropout                         |                      | (1, 197, 768)        |              |              | 1.21MB       |
|     ln_2                            | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|      ln_2.weight                    |                      | (768,)               |              |              |              |
|      ln_2.bias                      |                      | (768,)               |              |              |              |
|     mlp                             | 4.722M               |                      | 929.563M     | 1.860G       | 35.83MB      |
|      mlp.0                          | 2.362M               | (1, 197, 3072)       | 464.781M     | 929.563M     | 12.48MB      |
|       mlp.0.weight                  |                      | (3072, 768)          |              |              |              |
|       mlp.0.bias                    |                      | (3072,)              |              |              |              |
|      mlp.1                          |                      | (1, 197, 3072)       |              | 605.184K     | 4.84MB       |
|      mlp.2                          |                      | (1, 197, 3072)       |              |              | 4.84MB       |
|      mlp.3                          | 2.360M               | (1, 197, 768)        | 464.781M     | 929.563M     | 12.47MB      |
|       mlp.3.weight                  |                      | (768, 3072)          |              |              |              |
|       mlp.3.bias                    |                      | (768,)               |              |              |              |
|      mlp.4                          |                      | (1, 197, 768)        |              |              | 1.21MB       |
|   ln                                | 1.536K               | (1, 197, 768)        |              |              | 1.21MB       |
|    ln.weight                        |                      | (768,)               |              |              |              |
|    ln.bias                          |                      | (768,)               |              |              |              |
|  add_1                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_2                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_3                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_4                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_5                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_6                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_7                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_8                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_9                              |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_10                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_11                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_12                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_13                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_14                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_15                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_16                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_17                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_18                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_19                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_20                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_21                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_22                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_23                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  add_24                             |                      | (1, 197, 768)        |              | 151.296K     | 1.82MB       |
|  heads                              | 769.000K             |                      | 768.000K     | 1.536M       | 3.08MB       |
|   head                              | 769.000K             | (1, 1000)            | 768.000K     | 1.536M       | 3.08MB       |
|    head.weight                      |                      | (1000, 768)          |              |              |              |
|    head.bias                        |                      | (1000,)              |              |              |              |

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

Total parameters: 86.57M (86,567,656)
Total FLOPs: 22.553 GFLOPs (22,553,294,592)
Total MACs: 11.271 GMACs (11,271,124,992)
Total memory: 549.84 MB (549,837,680 bytes)

Subgraphs: 137
Average AI: 29.49 FLOPs/byte
