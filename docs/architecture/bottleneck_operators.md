# Bottleneck operators in SoTA DNNs


1. Transformer Models (General)

    Primary Bottleneck Operators:

    Multi-Head Attention (MHA)
    - Issue: Quadratic time complexity O(n²) with sequence length
    - Impact: Dominates computational cost for long sequences
    - Status: Most critical bottleneck across all transformer variants

    Scaled Dot Product Attention (SDPA)
    - Impact: Primary computational bottleneck in production video generation models
    - Real-world cost: Millions of dollars in latency overhead (NVIDIA 2024 study)

    KV Cache Management
    - Issue: Linear growth with context length → memory bottleneck
    - Impact: Limits long-context applications
    - Bandwidth: Major attention bandwidth consumption in decode phase

    Gather_ND Operation
    - Issue: Random-access memory pattern, hard to vectorize/parallelize
    - Impact: Significant bottleneck in sparse attention patterns

    Normalization Layers
    - Types: LayerNorm, BatchNorm
    - Issue: Computational inefficiency on resource-constrained devices

    Quantified Performance Data:

    Matrix Multiplication (GEMM)
    - Prefill phase: 87.6% of execution time
    - Decode phase: 76.2% of execution time
    - Conclusion: Dominant computational bottleneck

---
2. JEPA (Joint Embedding Predictive Architecture)

    Architecture Overview:

    - Yann LeCun's vision for world-modeling AI
    - Encoder-predictor architecture avoiding pixel-level reconstruction

    Identified Bottlenecks:

    EMA (Exponential Moving Average) Mechanism
    - Issue: Instability in I-JEPA framework
    - Impact: Limits convergence quality

    Prediction Mechanism
    - Issue: Struggles to learn mean of patch representations accurately
    - Impact: Performance degradation, limited applicability

    Solutions (2024 Research):
    - C-JEPA: Synergizes with VICReg to address EMA limitations
    - Results: Faster convergence, improved stability on ImageNet-1K

    Variants and Performance:

    - A-JEPA (Audio): Highly scalable with Vision Transformer structure
    - Point-JEPA: Faster pre-training than alternatives
    - 3D-JEPA: 88.65% accuracy with fewer epochs (150 vs standard)

---
3. Recursive Reasoning Models (o1, o3, DeepSeek-R1)

    New Paradigm: Inference-Time Compute

    Key Bottleneck: Time Per Output Token (TPOT)
    - Issue: Models generate millions of reasoning tokens
    - Impact: Takes seconds to minutes vs instant responses
    - Trade-off: Accuracy vs responsiveness

    Computational Architecture:

    DeepSeek-R1 Specifications:
    - Architecture: Mixture of Experts (MoE)
    - Total parameters: 671 billion
    - Active per forward pass: 37 billion (5.5% activation)
    - Efficiency gain: 18× fewer active parameters

    Chain-of-Thought (CoT) Generation:
    - Length: Potentially millions of tokens
    - Bottleneck: Sequential token generation can't be parallelized
    - Impact: Linear scaling with reasoning depth

    Performance Benchmarks (2025):

    | Model       | Codeforces | AIME 2024 | MATH-500 |
    |-------------|------------|-----------|----------|
    | o3          | 2727       | -         | -        |
    | DeepSeek-R1 | 2029       | 79.8%     | 97.3%    |
    | o1          | 1891       | 79.2%     | 96.4%    |

    Key Insight: Better reasoning requires more compute → TPOT becomes the bottleneck

---
4. Small Language Models (SLMs) - Phi-3, Gemma

    Primary Bottleneck: Matrix Multiplication

    GGML_OP_MUL_MAT Dominance:
    - Prefill phase: 87.6% of execution time
    - Decode phase: 76.2% of execution time
    - Confirmed: GEMM is THE key bottleneck in LLaMA inference

    Architecture Impact Beyond Size:

    Performance factors:
    - Number of layers
    - Vocabulary size
    - Attention head configuration
    - Architectural design > parameter count

    Top SLMs Performance (2024):

    Phi-3 Family:
    - Most capable and cost-effective SLMs
    - Phi-3-mini: Leading accuracy (September 2024)
    - Key: Data engineering and tuning techniques

    Gemma Models:
    - Gemma 2B and 7B from Google (Gemini technology)
    - Competitive but Llama 8B beats Gemma 9B on many benchmarks

    Models Analyzed (<3B parameters):
    - Olmo 1B
    - Qwen1.5 1.8B
    - Gemma 2B
    - Phi2 2.7B

---
5. KV Cache Optimization Research (2024)

    Major Advances:

    Quantization Techniques:
    - KVQuant: Targeting 10M context length
    - FlashAttention + KV quantization: Combined memory reduction

    Compression Methods:
    - Eviction: StreamingLLM, H2O, SnapKV, PyramidKV
    - Compression: KIVI, KVQuant, Palu (quantization + low-rank)

    Novel Approaches:
    - FourierAttention: Fixed-length state compression via Fourier transform
    - PagedAttention: Memory-efficient KV cache management

    Performance Gains:
    - Combined techniques: 3-8× speedup
    - FlashAttention inference optimization: Separate kernel for KV cache loading

---
Critical Bottleneck Operator Rankings by Architecture

Transformers (General):

1. Multi-Head Attention (O(n²) complexity)
2. GEMM operations (87% of compute)
3. KV cache memory bandwidth
4. Normalization layers
5. Gather_ND (sparse patterns)

JEPA:

1. Prediction mechanism accuracy
2. EMA stability
3. Encoder computational cost

Reasoning Models (o1, o3, R1):

1. Time Per Output Token (TPOT) - Critical UX bottleneck
2. Sequential CoT generation (can't parallelize)
3. MoE routing overhead
4. Active parameter memory footprint

SLMs (Phi-3, Gemma):

1. Matrix multiplication (87.6% prefill, 76.2% decode)
2. Vocabulary embedding lookups
3. Layer normalization
4. Small batch inefficiency

---
Optimization Opportunities for the KPU

Here are the key operators to characterize and optimize:

Tier 1 - Highest Impact:

1. Multi-Head Attention (all models)
    - SDPA kernel optimization
    - FlashAttention integration
    - KV cache compression
2. GEMM Operations (87% of compute)
    - Tiling strategies
    - Tensor Core utilization
    - Mixed precision optimization
3. KV Cache Management (memory bottleneck)
    - Quantization (INT8, INT4)
    - Eviction policies
    - PagedAttention

Tier 2 - Important:

4. Layer Normalization
    - Fused kernels with activation
    - SLAB optimizations
5. MoE Routing (reasoning models)
    - Expert selection overhead
    - Load balancing
6. Token Generation Loop (reasoning)
    - TPOT optimization
    - Speculative decoding

Tier 3 - Specialized:

7. Gather_ND (sparse attention)
8. JEPA Prediction Head
9. Vocabulary Embeddings (SLMs)

---
Recommended Research Papers (2024-2025)

1. "Breaking the Bottleneck: Advances in Efficient Transformer Design" (Feb 2025)
2. "SLAB: Efficient Transformers with Simplified Linear Attention" (ICML 2024)
3. "Inference-Time Scaling for Complex Tasks" (Microsoft 2025)
4. "DeepSeek-R1: Incentivizing Reasoning Capability" (Jan 2025)
5. "Small Language Models: A Comprehensive Survey" (2024)
6. "KVQuant: Towards 10M Context Length" (2024)
