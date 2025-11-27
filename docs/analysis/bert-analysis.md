# Bidirectional Encoder Representations from Transformers (BERT)

The BERT acronym reflects the model’s architecture and training approach:

- Bidirectional: Unlike traditional language models that read text left-to-right or right-to-left, BERT reads in both directions simultaneously. This allows it to understand the full context of a word based on all surrounding words.

- Encoder Representations: BERT uses only the encoder part of the Transformer architecture to generate deep contextual representations of words.

- from Transformers: It’s built on the Transformer model, a neural network architecture introduced in the paper “Attention is All You Need”.

It's a deep learning model developed by Google in 2018 that revolutionized natural language processing by enabling models to understand the context of a word based on both its left and right surroundings. This bidirectional approach, combined with the Transformer architecture, allows BERT to achieve state-of-the-art performance on a wide range of NLP tasks like question answering, sentiment analysis, and language inference.

![Transformers](./img/transformer.png)

![multi-head](./img/BERT-multi-head-attention-detail.png)

![BERT Base vs Large](./img/BERT-base-vs-large.png)


```bash
(p311) stillwater@sw-21:~/dev/branes/clones/graphs$ python cli/profile_graph.py --model bert-base-uncased

Input type: Tokens (batch_size=1, seq_len=128, with attention_mask) [BERT-style]
[1/4] Warming up model (initializing any lazy modules)...
[2/4] Attempting standard FX symbolic_trace...
  ✗ Standard FX trace failed: ValueError
  Falling back to Dynamo export...
[2/4] Using PyTorch Dynamo export (more robust)...
  ✓ Dynamo export successful
[3/4] Propagating tensor shapes through graph...
[4/4] Running graph partitioner...

====================================================================================================
HIERARCHICAL GRAPH PROFILE
====================================================================================================

| Module                                   | #Parameters          | MACs         | FLOPs        | Memory       |
|:-----------------------------------------|:---------------------|:-------------|:-------------|:-------------|
|  model                                   | 109.482M             | 10.872G      | 21.744G      | 467.88MB     |
|   model.embeddings                       | 23.837M              |              |              | 2.75MB       |
|    model.embeddings.word                 | 23.441M              |              |              | 393.73KB     |
|     ...word.embeddings                   | 23.441M              |              |              | 393.73KB     |
|    model.embeddings.token                | 1.536K               |              |              | 393.73KB     |
|     model.embeddings.token.type          | 1.536K               |              |              | 393.73KB     |
|      ...type.embeddings                  | 1.536K               |              |              | 393.73KB     |
|    model.embeddings.position             | 393.216K             |              |              | 393.73KB     |
|     ...ition.embeddings                  | 393.216K             |              |              | 393.73KB     |
|    model.embeddings.LayerNorm            | 1.536K               |              |              | 786.43KB     |
|    model.embeddings.dropout              |                      |              |              | 786.43KB     |
|   model.encoder                          | 85.054M              | 10.872G      | 21.743G      | 462.75MB     |
|    model.encoder.layer                   | 85.054M              | 10.872G      | 21.743G      | 462.75MB     |
|     model.encoder.layer.0                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...0.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...0.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.0.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.1                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...1.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...1.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.1.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.2                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...2.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...2.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.2.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.3                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...3.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...3.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.3.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.4                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...4.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...4.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.4.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.5                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...5.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...5.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.5.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.6                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...6.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...6.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.6.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.7                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...7.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...7.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.7.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.8                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...8.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...8.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.8.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.9                | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...9.attention                      | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...9.intermediate                   | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.9.output        | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.10               | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...10.attention                     | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...10.intermediate                  | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.10.output       | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|     model.encoder.layer.11               | 7.088M               | 905.970M     | 1.812G       | 38.56MB      |
|      ...11.attention                     | 2.364M               | 301.990M     | 603.980M     | 14.17MB      |
|       ...attention.self                  | 1.772M               | 226.492M     | 452.985M     | 9.45MB       |
|        ...self.query                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.key                       | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...self.value                     | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|       ...attention.output                | 592.128K             | 75.497M      | 150.995M     | 4.72MB       |
|        ...output.dense                   | 590.592K             | 75.497M      | 150.995M     | 3.15MB       |
|        ...output.dropout                 |                      |              |              | 786.43KB     |
|        ...output.LayerNorm               | 1.536K               |              |              | 786.43KB     |
|      ...11.intermediate                  | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|       ...termediate.dense                | 2.362M               | 301.990M     | 603.980M     | 11.42MB      |
|      model.encoder.layer.11.output       | 2.362M               | 301.990M     | 603.980M     | 12.98MB      |
|       ...output.dense                    | 2.360M               | 301.990M     | 603.980M     | 11.41MB      |
|       ...output.dropout                  |                      |              |              | 786.43KB     |
|       ...output.LayerNorm                | 1.536K               |              |              | 786.43KB     |
|   model.pooler                           | 590.592K             | 589.824K     | 1.180M       | 2.37MB       |
|    model.pooler.dense                    | 590.592K             | 589.824K     | 1.180M       | 2.37MB       |
|    model.pooler.activation               |                      |              |              | 6.14KB       |
|  embeddings                              |                      |              | 98.304K      | 1.18MB       |
|  inverted                                |                      |              | 16.384K      | 131.07KB     |
|   inverted.mask                          |                      |              | 16.384K      | 131.07KB     |
|  add                                     |                      |              | 2.359M       | 28.31MB      |
|   add.2                                  |                      |              | 98.304K      | 1.18MB       |
|   add.3                                  |                      |              | 98.304K      | 1.18MB       |
|   add.4                                  |                      |              | 98.304K      | 1.18MB       |
|   add.5                                  |                      |              | 98.304K      | 1.18MB       |
|   add.6                                  |                      |              | 98.304K      | 1.18MB       |
|   add.7                                  |                      |              | 98.304K      | 1.18MB       |
|   add.8                                  |                      |              | 98.304K      | 1.18MB       |
|   add.9                                  |                      |              | 98.304K      | 1.18MB       |
|   add.10                                 |                      |              | 98.304K      | 1.18MB       |
|   add.11                                 |                      |              | 98.304K      | 1.18MB       |
|   add.12                                 |                      |              | 98.304K      | 1.18MB       |
|   add.13                                 |                      |              | 98.304K      | 1.18MB       |
|   add.14                                 |                      |              | 98.304K      | 1.18MB       |
|   add.15                                 |                      |              | 98.304K      | 1.18MB       |
|   add.16                                 |                      |              | 98.304K      | 1.18MB       |
|   add.17                                 |                      |              | 98.304K      | 1.18MB       |
|   add.18                                 |                      |              | 98.304K      | 1.18MB       |
|   add.19                                 |                      |              | 98.304K      | 1.18MB       |
|   add.20                                 |                      |              | 98.304K      | 1.18MB       |
|   add.21                                 |                      |              | 98.304K      | 1.18MB       |
|   add.22                                 |                      |              | 98.304K      | 1.18MB       |
|   add.23                                 |                      |              | 98.304K      | 1.18MB       |
|   add.24                                 |                      |              | 98.304K      | 1.18MB       |
|   add.25                                 |                      |              | 98.304K      | 1.18MB       |

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
MODEL SUMMARY
====================================================================================================

Model: bert-base-uncased
Input: Tokens (batch_size=1, seq_len=128)
Tracing method: dynamo_export

Total parameters: 109.48M (109,482,240)
Total FLOPs: 21.747 GFLOPs (21,746,925,568)
Total MACs: 10.872 GMACs (10,872,225,792)

Memory breakdown:
  Input tensors:  81.86 MB
  Output tensors: 73.21 MB
  Weights:        342.43 MB
  Total:          497.50 MB

Graph structure:
  Subgraphs (fused ops): 153
  Average arithmetic intensity: 36.11 FLOPs/byte

Bottleneck analysis:
  Compute-bound ops: 72 (47.1%)
  Memory-bound ops:  81 (52.9%)
  (Threshold: AI > 10 FLOPs/byte)
```
**The many tensor additions at the bottom of BERT stem from residual connections, layer outputs, and embedding summations—all crucial for preserving information and enabling deep learning stability.**

Here's a breakdown of why these tensor additions are so prevalent:

---

### 🧠 Key Reasons for Tensor Adds in BERT's Lower Layers

- **Residual Connections (Skip Connections):**
  - Every transformer layer in BERT includes residual connections around both the multi-head attention and feed-forward sublayers.
  - These are implemented as tensor additions: the input to a sublayer is added to its output before normalization.
  - *Purpose:* Helps preserve gradient flow, stabilizes training, and allows deeper networks to learn effectively.

- **Embedding Summation:**
  - The initial input representation is formed by summing three tensors:
    - *Token embeddings* (word identity)
    - *Segment embeddings* (sentence A/B distinction)
    - *Position embeddings* (token order)
  - This composite tensor is the first input to the transformer stack.

- **Layer Output Aggregation:**
  - When `output_hidden_states=True`, BERT returns a tuple of hidden states from each layer.
  - These are often aggregated (e.g., summed, averaged, or concatenated) for downstream tasks like classification or question answering.
  - Tensor adds may appear here if you're pooling across layers.

- **Multi-Head Attention Output:**
  - Each attention head produces a tensor, and these are concatenated and linearly transformed.
  - The result is added back to the input via a residual connection.

---

### 🔍 Why It Matters

- These additions are not redundant—they’re *architectural necessities* for BERT’s deep learning capabilities.
- They allow BERT to maintain context, stabilize gradients, and integrate multiple sources of information (e.g., position, token, segment).
- If you're profiling or optimizing BERT, these tensor adds are hotspots for memory movement and compute cost, especially in large-scale deployments.

Would you like a breakdown of how these tensor adds affect energy-delay or memory movement in your KPU pipeline modeling? I can synthesize a Gantt-style view or annotate the loop nest implications.