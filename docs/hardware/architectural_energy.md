# Architectural energy

✦ Here are the summaries of the energy models for the four architectures, based on the analysis of `./cli/compare_architectures_energy.py`

  ## CPU (Stored-Program) Energy Model

  The CPU's energy model is characterized by the overheads inherent in a sequential, instruction-driven architecture. Its primary energy
  events are:

   * Instruction Pipeline: A significant portion of energy is consumed by the frontend of the processor, which includes fetching instructions
     from the I-cache, decoding them into micro-operations, and dispatching them to execution units. This overhead is paid for every
     instruction.
   * Register File Operations: Accessing the register file for reading operands and writing back results is a high-frequency event that
     consumes energy comparable to the arithmetic operations themselves.
   * Memory Hierarchy: Energy is spent moving data through the multi-level cache hierarchy (L1, L2, L3) and to/from main DRAM. Cache misses
     that require fetching from DRAM are particularly expensive.
   * ALU Operations: This represents the "useful" work, where energy is consumed to perform the actual floating-point calculations.
   * Branch Prediction: The energy cost of predicting the direction of control flow branches and the penalty for mispredictions contribute to
     the control overhead.

  ## GPU (Data-Parallel) Energy Model

  The GPU's energy model is defined by its massively parallel SIMT (Single Instruction, Multiple Thread) architecture. While powerful, this
  parallelism introduces substantial energy overheads for coordination and data management:

   * SIMT Control Overheads: This is the largest source of architectural overhead.
       * Coherence Machinery: The dominant energy consumer, responsible for keeping the caches of thousands of cores consistent.
       * Thread Scheduling: Warp schedulers consume energy to manage the execution of a massive number of threads to hide latency.
       * Memory Coalescing: Hardware units use energy to bundle memory requests from multiple threads into fewer transactions.
   * Compute Units: Energy is consumed by both the general-purpose CUDA Cores and the specialized Tensor Cores, which are highly efficient
     for matrix arithmetic.
   * Large Register File: A very large register file is required to service thousands of in-flight threads, and the energy to access it is
     substantial.
   * Memory Hierarchy: Energy is consumed accessing the GPU's memory system, which includes a fast on-chip Shared Memory/L1 cache, a larger
     L2 cache, and high-bandwidth DRAM.

  ## TPU (Systolic-Array) Energy Model

  The TPU's energy model reflects its highly specialized, dataflow-driven design, which minimizes the control overhead seen in CPUs and
  GPUs. Its core energy events are tied to the efficient movement of data through the systolic array:

   * Systolic Array Operations: The primary energy consumer is the array of MAC (Multiply-Accumulate) units performing computations. Because
     data is reused extensively within the array, this is extremely efficient.
   * On-Chip Buffers: Significant energy is used to access the large on-chip memory buffers that hold activations and weights, reducing the
     need for expensive off-chip memory access.
   * Weight-Stationary Dataflow Control: Instead of fetching instructions, the TPU's energy is spent on orchestrating the dataflow. This
     includes:
       * DMA Controller: For loading data blocks from DRAM.
       * Weight Loading Sequencer: For shifting weights into the array.
       * Accumulator Control: For managing partial sums.
   * Minimal Instruction Overhead: A single instruction can initiate millions of operations, making the energy cost of instruction decoding
     negligible on a per-MAC basis.

  ## KPU (Domain-Flow) Energy Model

  The KPU's energy model is based on a programmable distributed dataflow architecture that eliminates traditional instruction processing in favor of token-based execution. This results in a unique energy profile:

   * Token-Based Execution: The core of the architecture's efficiency.
       * Data Token Signature Matching: Energy is spent in hardware to match incoming data tokens (operands) with the instruction tokens to which they belong. When all operands of the instruction token are available, the instruction token is ready to be dispatched to the target ALU.
       * Result Data Token Routing: A mesh network consumes energy to route tokens between processing elements, emulating data dependencies specified in the Domain Flow Program.
       * The compute fabric operated purely on pushed-based execution, removing the need for instruction pointers, fetch and decode, or any request/reply cycles to gather data.
   * Specialized Data Movement Engines: The architecture uses three distinct engines to optimize data movement, each with its own energy cost: 
       * a DMA Engine (DRAM to L3)
       * a BlockMover (L3 to L2)
       * a Streamer (L2 to L1).
   * Distributed Memory Hierarchy: Energy is consumed across a 4-stage memory hierarchy (DRAM -> L3 -> L2 -> L1) designed to keep data as local to the processing elements as possible.
   * Computation: Energy is consumed by the processing elements performing the actual MAC operations, triggered by the arrival of data tokens. There is no energy spent on instruction fetching or decoding during computation.
