# CUDA for Jetson

## Grids, Blocks, and Threads

## The Execution Hierarchy

CUDA has a three-level hierarchy that maps your logical parallelism onto hardware:

```
Grid (1 kernel launch)
 └── Blocks (scheduled onto SMs)
      └── Threads (executed in warps of 32)
```

### Threads

The finest unit of execution. Each thread runs your kernel code with its own registers and can compute a unique index:

```cpp
__global__ void add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}
```

Each thread has access to `threadIdx.x/y/z` (its position within its block). 
Each Ampere SM has 128 CUDA cores and 65536 registers.

### Blocks

A group of threads that:
- Execute on a **single SM** (cannot span SMs)
- Can synchronize with each other via `__syncthreads()`
- Share a fast scratchpad memory called **shared memory**
- Are limited in size (1024 threads max on the Jetson Orin)

Blocks are configured through the block dimension parameter at launch:

```cpp
dim3 blockDim(256);        // 1D: 256 threads
dim3 blockDim(16, 16);     // 2D: 256 threads (16×16)
dim3 blockDim(8, 8, 4);    // 3D: 256 threads (8×8×4)
```

### Grid

The collection of all blocks for a kernel launch. Blocks in a grid:
- **Cannot synchronize** with each other (within a single kernel)
- Are distributed across SMs by the hardware scheduler
- Can number in the thousands

```cpp
int n = 1000000;
int threadsPerBlock = 256;
int blocksInGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // ceiling division

add<<<blocksInGrid, threadsPerBlock>>>(a, b, c, n);
//     ^^^^^^^^^     ^^^^^^^^^^^^^^^
//     grid dim      block dim
```

## Visual Mapping to Hardware

```
Your Orin Nano (8 SMs, 128 cores each)
┌─────────────────────────────────────────────────────────────────┐
│    SM 0          SM 1          SM 2         ...     SM 7        │
│ ┌───────────┐ ┌───────────┐ ┌───────────┐         ┌───────────┐ │
│ │ Block 0   │ │ Block 1   │ │ Block  2  │   ...   │ Block  7  │ │
│ │ Block 8   │ │ Block 9   │ │ Block 10  │         │ Block 15  │ │
│ │ ...       │ │ ...       │ │ ...       │         │ ...       │ │
│ └───────────┘ └───────────┘ └───────────┘         └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

Multiple blocks can reside on one SM simultaneously (limited by resources). The scheduler interleaves their execution.

## Warps: Yes, You Need to Worry About Them

A **warp** is 32 threads that execute in lockstep (SIMT model). This is the actual unit of execution on the hardware.

### Why It Matters

**1. Divergence Penalty**

```cpp
// Bad: threads in same warp take different paths
if (threadIdx.x % 2 == 0) {
    doExpensiveThingA();  // Half the warp waits
} else {
    doExpensiveThingB();  // Other half waits
}
// Effective throughput: 50%
```

Both paths execute serially—the warp runs path A while path-B threads are masked off, then vice versa.

```cpp
// Better: divergence at warp boundaries
if (threadIdx.x < 32) {
    doExpensiveThingA();  // Warp 0: all threads take this
} else {
    doExpensiveThingB();  // Warp 1: all threads take this
}
// Full throughput: both warps run independently
```

**2. Memory Coalescing**

Warps issue memory requests together. If threads 0–31 access consecutive addresses, it's one transaction. If they access scattered addresses, it's up to 32 transactions:

```cpp
// Good: coalesced (threads access consecutive elements)
float val = input[idx];  // where idx = blockIdx.x * blockDim.x + threadIdx.x

// Bad: strided access
float val = input[idx * stride];  // If stride > 1, uncoalesced

// Terrible: random access
float val = input[randomIndex[idx]];  // Likely 32 separate transactions per warp
```

On your memory-bandwidth-limited Orin Nano, uncoalesced access can destroy performance.

**3. Warp-Level Primitives**

Modern CUDA exposes fast warp-level operations:

```cpp
// Warp reduction (no shared memory needed)
float sum = val;
for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
// Thread 0 now has sum of all 32 values

// Warp vote
bool allTrue = __all_sync(0xffffffff, predicate);
bool anyTrue = __any_sync(0xffffffff, predicate);
unsigned ballot = __ballot_sync(0xffffffff, predicate);  // Bitmask of which threads have true
```

These are faster than shared memory alternatives.

## Practical Guidelines for Your Hardware

| Concern | Guidance |
|---------|----------|
| Block size | Use multiples of 32 (warp size). 128 or 256 are safe defaults. |
| Grid size | At minimum 8 blocks (one per SM), ideally many more for latency hiding. |
| Divergence | Structure conditionals so threads within a warp take the same path when possible. |
| Memory access | Ensure threads in a warp access consecutive addresses. |
| Shared memory | Use it to stage data when you need non-coalesced access patterns—load coalesced into shared, then access arbitrarily. |

## Block Size Selection Heuristic

```cpp
// Query and use recommended configuration
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

// Or just use 256 as a reasonable default
// Then verify with profiler that occupancy is acceptable
```

For the Jetson Nano 8 SMs with 128 cores each: 128 cores ÷ 32 threads/warp = 4 warps can execute simultaneously per SM. But more warps (blocks/threads) can be resident to hide memory latency through interleaving—this is the occupancy game.

## Optimize for Jetson

This is a nuanced problem. With only 8 SMs, the Nano has limited parallelism at the grid level, so the strategy differs significantly from desktop/datacenter GPUs. Here's the core framework:

## 1. Understand the Bottleneck First

For each kernel, profile whether it's:
- **Memory-bound**: Arithmetic intensity < ~10 FLOPs/byte (for Orin's bandwidth/compute ratio)
- **Compute-bound**: Arithmetic intensity higher

Use `ncu` (Nsight Compute) or at minimum `nvprof` to get `sm__throughput.avg.pct_of_peak_sustained` and `dram__throughput.avg.pct_of_peak_sustained`.

## 2. Concurrent Kernel Execution via Streams

With 8 SMs, you can run multiple small kernels concurrently if they don't individually saturate the device:

```cpp
cudaStream_t streams[4];
for (int i = 0; i < 4; ++i)
    cudaStreamCreate(&streams[i]);

// Launch kernels that collectively fill SMs
kernelA<<<gridA, blockA, 0, streams[0]>>>(...);
kernelB<<<gridB, blockB, 0, streams[1]>>>(...);
kernelC<<<gridC, blockC, 0, streams[2]>>>(...);
```

**Critical caveat**: Concurrent execution only happens if:
- Kernels are in different streams
- Each kernel uses fewer resources than available (registers, shared memory, blocks)
- Hardware scheduler decides it's beneficial

On Orin Nano with 8 SMs, a kernel launching 8+ blocks of high occupancy will monopolize the device regardless of streams.

## 3. Occupancy Considerations

Each SM can run multiple blocks concurrently, limited by:
- Registers per SM (65536 on Ampere-derived arch)
- Shared memory per SM (typically 48-164 KB configurable)
- Max blocks per SM (16-32 depending on architecture)
- Max threads per SM (1536-2048)

Use the occupancy API to guide launch config:

```cpp
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

// Or query for a specific block size
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, 256, sharedMemBytes);
```

## 4. Kernel Fusion for Memory-Bound Operators

This is often the highest-impact optimization. If you have a graph like:

```
A → B → C (each memory-bound)
```

Each kernel round-trips through DRAM. Fuse them:

```cpp
// Instead of 3 kernels with 3 global memory round-trips
__global__ void fused_ABC(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        val = opA(val);  // Was kernel A
        val = opB(val);  // Was kernel B
        val = opC(val);  // Was kernel C
        out[idx] = val;
    }
}
```

This transforms 3× memory bandwidth usage into 1×.

## 5. Overlap Compute and Memory Transfers

If you're streaming data from host or doing multi-stage pipelines:

```cpp
for (int i = 0; i < numChunks; ++i) {
    int curr = i % 2;
    int prev = (i + 1) % 2;
    
    // Async copy for current chunk
    cudaMemcpyAsync(d_in[curr], h_in + i*chunkSize, bytes, 
                    cudaMemcpyHostToDevice, copyStream);
    
    // Process previous chunk (overlapped)
    if (i > 0)
        processKernel<<<grid, block, 0, computeStream>>>(d_in[prev], d_out[prev]);
    
    cudaStreamSynchronize(computeStream);  // Ensure previous compute done before overwriting
}
```

## 6. Graph-Level Strategy: Operator Clustering

For your heterogeneous compute graph, consider:

```
[Memory-bound ops] → fuse aggressively
[Compute-bound ops] → ensure high occupancy, can run independently
[Mixed dependencies] → use CUDA Graphs to reduce launch overhead
```

CUDA Graphs eliminate kernel launch overhead (which matters when you have many small kernels):

```cpp
cudaGraph_t graph;
cudaGraphExec_t graphExec;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// Launch your entire compute graph here
kernelA<<<...>>>();
kernelB<<<...>>>();
// ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Execute many times with minimal overhead
for (int iter = 0; iter < 1000; ++iter)
    cudaGraphLaunch(graphExec, stream);
```

## 7. Practical Prioritization for 8 SMs

1. **Fuse memory-bound operators first** — biggest wins on bandwidth-limited Jetson
2. **Use CUDA Graphs** if you're iterating the same graph repeatedly
3. **Concurrent kernels** only help when individual kernels are small; profile to verify actual concurrency
4. **Don't over-optimize occupancy** — 50% occupancy is often sufficient if memory access patterns are good
5. **Consider Tensor Cores** for any matmul-like operations (even small ones), but note the data layout constraints (typically require dimensions divisible by 8/16)

## What Not to Do

- Don't assume more streams = more parallelism. The HW scheduler is the arbiter.
- Don't micro-optimize block sizes before profiling. The difference between 128 and 256 threads is often noise.
- Don't ignore memory coalescing—on a bandwidth-starved device like Orin Nano, uncoalesced access is catastrophic.

