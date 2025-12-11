# DNN Tensor Shape Heatmap Analysis

This document provides interpretation of the (M, N), (M, K), and (K, N) heatmaps
generated from 2,262 matrix multiply operations across 30 DNN models.

## Matrix Multiply Notation

For matrix multiply `C = A @ B`:
- **A** is the input activation matrix with shape (M, K)
- **B** is the weight matrix with shape (K, N)
- **C** is the output activation matrix with shape (M, N)

Where:
- **M** = output rows = batch_size x spatial_elements (e.g., H_out x W_out for conv)
- **K** = reduction dimension = input_channels x kernel_elements (e.g., C_in x kH x kW)
- **N** = output columns = output_channels

---

## (M, N) Heatmap: Output Tensor Dimensions

**What it shows**: The frequency of output activation tensor shapes across all operations.

### Why This Matters for Systolic Arrays

A systolic array is a 2D grid of processing elements (PEs) that computes matrix multiply
by flowing data through the array. For a `rows x cols` array computing `C = A @ B`:

```
                    B (weights) flows down
                    [K x N matrix]
                         |
                         v
    A (input) -->  +-----------+  --> C (output)
    [M x K]        | PE  PE  PE|      [M x N]
    flows right    | PE  PE  PE|
                   | PE  PE  PE|
                   +-----------+
                   [rows x cols]
```

The M dimension maps to how many rows of output are computed in parallel (array rows),
and N maps to how many columns of output are computed in parallel (array columns).

**Utilization follows directly from this mapping:**
- If M < array_rows: some PE rows sit idle (row underutilization)
- If N < array_cols: some PE columns sit idle (column underutilization)
- Spatial utilization = min(M/rows, 1) x min(N/cols, 1)

For example, a 128x128 array processing a layer with M=64, N=256:
- Row utilization: 64/128 = 50%
- Column utilization: min(256/128, 1) = 100%
- Spatial utilization: 50% x 100% = 50%

This is why the (M, N) heatmap directly predicts which array sizes will be efficient:
operations clustering at small M or N values will underutilize large arrays.

### Key Observations

1. **Dominant Hotspot at M~196, N~256-2048**
   - The darkest region centers around M=196 with N spanning 256-2048
   - M=196 corresponds to 14x14 spatial feature maps (196 = 14x14)
   - This is the output resolution of layer3/layer4 in ResNets and similar depths in other CNNs
   - 42.3% of all operations have M in the range [150-250]

2. **Vertical Bands at Powers of 2**
   - Clear vertical concentration at N=128, 256, 512, 1024
   - These are standard channel counts in CNN architectures
   - Top N values: 256 (182 ops), 512 (143 ops), 1024 (134 ops)

3. **M=1 Column (Left Edge)**
   - 15.9% of operations have M=1
   - These are fully-connected layers with batch_size=1
   - Causes severe row underutilization on systolic arrays

4. **Large M Values (Right Side)**
   - Sparse operations with M > 10,000
   - Early CNN layers: M=50,176 corresponds to 224x224 input images
   - M=12,544 corresponds to 112x112 feature maps (first conv layers)
   - M=3,136 corresponds to 56x56 feature maps (7.0% of ops)

### Systolic Array Implications

| Array Size | Ops with M < size | Ops with N < size | Severe Underutil |
|------------|-------------------|-------------------|------------------|
| 16x16      | 15.9%             | 1.8%              | 1.8%             |
| 32x32      | 15.9%             | 5.2%              | 3.7%             |
| 64x64      | 28.7%             | 16.1%             | 7.4%             |
| 128x128    | 31.6%             | 27.1%             | 10.0%            |

The M dimension is the primary utilization bottleneck. Even at 128x128, nearly a third
of operations cannot fully utilize the array rows.

---

## (M, K) Heatmap: Input Activation Matrix Dimensions

**What it shows**: The relationship between output spatial size (M) and reduction
dimension (K), which together define the input activation matrix A with shape (M, K).

### Why This Matters for Data Movement

The (M, K) relationship determines how input activations flow through the system and
how much data reuse is possible. Consider the matrix multiply `C = A @ B`:

```
    Input A [M x K]          Weight B [K x N]         Output C [M x N]
    +-------------+          +-------------+          +-------------+
    | row 0       | -------> |             | -------> | row 0       |
    | row 1       | -------> |   K x N     | -------> | row 1       |
    | ...         |          |   weights   |          | ...         |
    | row M-1     | -------> |             | -------> | row M-1     |
    +-------------+          +-------------+          +-------------+
```

**Data reuse mechanics:**
- Each row of A (length K) is multiplied with the entire weight matrix B
- If we tile across N (output columns), the same input row is reused N/Tn times
- Larger M means more input rows, each needing K elements loaded
- Larger K means longer dot products, more arithmetic per input element

**Memory bandwidth implications:**
- Input bandwidth = M x K x element_size (total input data)
- For weight-stationary dataflow: inputs stream through while weights stay fixed
- Input reuse factor = N_tiles (each input tile reused across all output column tiles)

**Why the (M, K) pattern matters:**
- Large M, small K (upper-left region): Many short rows, high bandwidth pressure per output
- Small M, large K (lower-right region): Few long rows, better arithmetic intensity
- The heatmap shows where most operations fall, indicating typical bandwidth requirements

For a layer with M=196, K=2304 on a 128x128 array:
- Input tile: 128 x 2304 = 576 KB per tile
- Number of M tiles: ceil(196/128) = 2
- Each input tile reused across N dimension (weight reuse opportunity)

### Key Observations

1. **Vertical Striping Pattern**
   - Strong vertical bands at specific M values: ~49, ~196, ~784, ~3136
   - These correspond to standard feature map sizes:
     - 49 = 7x7 (final conv layers)
     - 196 = 14x14 (mid-network)
     - 784 = 28x28 (early-mid network)
     - 3136 = 56x56 (early layers)

2. **K Spans Wide Range at Each M**
   - At M=196, K ranges from ~27 to ~5000+
   - Small K (~9-27): Depthwise convolutions or 3x3 convs with few input channels
   - Large K (~1000-5000): Later layers with many input channels

3. **Inverse Relationship Trend**
   - General pattern: larger M tends to have smaller K
   - Early layers: large spatial (M) but few channels (small K)
   - Late layers: small spatial (M) but many channels (large K)
   - This is the standard CNN "funnel" architecture

4. **Horizontal Band at K~9**
   - Visible low-K band across multiple M values
   - K=9 corresponds to 3x3 kernels with 1 input channel (depthwise conv)
   - 89 operations have K=9

### Data Reuse Implications

- **Large M, Small K**: High input reuse potential (each input row used many times)
- **Small M, Large K**: Lower input reuse, but potentially high weight reuse
- The hotspot at (M~196, K~1000-2000) represents good balance for weight-stationary

---

## (K, N) Heatmap: Weight Matrix Dimensions

**What it shows**: The shape distribution of weight tensors B with shape (K, N).
This is the most important heatmap for weight-stationary accelerators like TPUs.

### Why This Matters for Weight-Stationary Dataflows

In weight-stationary dataflow, weights are loaded into the systolic array once and
held stationary while multiple batches of input activations stream through:

```
    Weight loading phase:              Compute phase (repeated for each batch):

    DRAM -> Array                      Input streams through, weights fixed
    +-------------+                    +-------------+
    | w00 w01 w02 |                    | w00 w01 w02 | <-- A[batch_i] streams in
    | w10 w11 w12 |  (load once)       | w10 w11 w12 |     from left
    | w20 w21 w22 |                    | w20 w21 w22 | --> C[batch_i] streams out
    +-------------+                    +-------------+     to right
    Weight B [K x N]
```

**Why weight dimensions matter:**

1. **On-chip storage requirements**: The weight matrix must fit in on-chip buffers
   - Weight size = K x N x element_size
   - K=1024, N=1024 with BF16 = 2 MB
   - Larger weights require tiling, reducing reuse benefits

2. **Weight reuse opportunity**: Each weight element is used M times (once per input row)
   - Weight reuse factor = M_tiles (number of input row tiles)
   - Larger M means more reuse of each weight load
   - This is why batch size increases efficiency: larger M = more weight reuse

3. **DRAM bandwidth amortization**: Weight loads are amortized across all input rows
   - Energy to load weights from DRAM: K x N x 200 pJ/byte (typical)
   - This cost is paid once, then reused M times
   - Operations with small M relative to K x N waste weight bandwidth

**The (K, N) heatmap reveals:**
- How large weights typically are (memory sizing)
- Whether weights fit on-chip without tiling (K x N vs buffer size)
- The correlation between input and output channel counts (architecture patterns)

For a transformer MLP layer with K=768, N=3072:
- Weight size: 768 x 3072 x 2 = 4.7 MB (BF16)
- Requires tiling on most accelerators (TPU v4 has 32 MB unified buffer)
- Each weight tile reused across sequence length (M = seq_len x batch)

### Key Observations

1. **Diagonal Correlation**
   - Clear positive correlation between K and N
   - Larger networks have both more input channels (K) and output channels (N)
   - This reflects the scaling behavior of modern architectures

2. **Dense Region at K~64-512, N~128-512**
   - Highest concentration in the "medium-sized" weight region
   - These are typical mid-network layer dimensions
   - Weights in this range: 64x128 to 512x512 = 8KB to 512KB (BF16)

3. **Transformer Signature (Upper Right)**
   - Cluster at K~768-1024, N~768-3072
   - These are attention projection and MLP weights in transformers
   - BERT/ViT hidden_dim=768, MLP expansion=4x gives N=3072

4. **Small Weight Matrices (Lower Left)**
   - Sparse region with K < 32, N < 64
   - Early conv layers or specialized operations
   - High risk of weight underutilization on large arrays

5. **Asymmetric Weights**
   - Some operations have K >> N (tall matrices): channel reduction layers
   - Others have N >> K (wide matrices): channel expansion layers
   - MLP layers in transformers: K=768, N=3072 (expansion) then K=3072, N=768 (projection)

### Weight Memory Analysis

| K x N Range       | Approx Weight Size (BF16) | Count | Typical Layer Type |
|-------------------|---------------------------|-------|-------------------|
| < 64 x 64         | < 8 KB                    | ~15%  | Early conv, depthwise |
| 64-256 x 64-256   | 8-128 KB                  | ~30%  | Mid conv layers |
| 256-1024 x 256-1024 | 128 KB - 2 MB           | ~35%  | Late conv, attention |
| > 1024 x 1024     | > 2 MB                    | ~20%  | Large MLP, classifiers |

---

## DNN Class Differences

### CNNs (1,923 operations)
- **M**: median=196, highly variable (1 to 50,176)
- **K**: median=288, moderate range
- **N**: median=240, powers-of-2 dominant
- Characterized by the spatial dimension pyramid (large M early, small M late)

### Encoders - ViT/BERT (285 operations)
- **M**: median=196 (sequence length x batch or patch count)
- **K**: median=1024, consistently large (hidden dimensions)
- **N**: median=1024, consistently large
- More uniform dimensions due to fixed hidden_dim throughout

### Full Transformers - T5/BART (54 operations)
- **M**: median=196
- **K**: median=512
- **N**: median=256
- Mix of encoder and decoder patterns

---

## Recommendations for Systolic Array Design

### Array Size Selection

Based on the dimension distributions:

1. **16x16 arrays** achieve 97.5% weighted utilization
   - Only 1.8% of ops severely underutilize both dimensions
   - Good for edge deployment where area/power is critical

2. **32x32 arrays** achieve 95.5% weighted utilization
   - Sweet spot for CNN inference
   - Most operations (M~196, N~256) map efficiently

3. **64x64 arrays** achieve 90.0% weighted utilization
   - 28.7% of ops underutilize rows (M < 64)
   - Better for transformer workloads with larger dimensions

4. **128x128 arrays** achieve 80.1% weighted utilization
   - Significant underutilization for many operations
   - Best suited for large batch sizes or training workloads

### Dataflow Recommendations

1. **Weight-Stationary** (TPU-style)
   - Best when K is large relative to M and N
   - Good for transformer MLP layers (K=3072, N=768)
   - Maximizes weight reuse across batch dimension

2. **Output-Stationary**
   - Best when M and N are large, K is moderate
   - Reduces partial sum traffic for large outputs
   - Good for early CNN layers with large feature maps

3. **Row-Stationary** (Eyeriss-style)
   - Best for balanced dimensions
   - Good default for mixed workloads
   - Particularly effective for depthwise convolutions

### Memory Hierarchy Sizing

Based on weight matrix dimensions:
- **L1/Unified Buffer**: 2-4 MB to hold full weight tiles for most operations
- **Accumulator storage**: Size for (M_tile x N_tile) partial sums
- **Input buffer**: Size for (M_tile x K_tile) activation tiles

The (K, N) distribution suggests most weights fit in 2MB, with outliers up to 8MB
for the largest transformer layers.
