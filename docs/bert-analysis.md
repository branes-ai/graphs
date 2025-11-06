# BERT graph

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