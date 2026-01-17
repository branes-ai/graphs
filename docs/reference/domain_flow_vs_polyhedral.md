# Concurrency Analysis: Domain Flow vs Polyhedral

Who published the paper "Domain flow and streaming architectures"?

I'll search for information about the paper "Domain flow and streaming architectures" to find out who published it. I found a few relevant results. Let me fetch more details from the IEEE publication and the ResearchGate entry to get the author information. 

(Domain flow and streaming architectures)[https://ieeexplore.ieee.org/document/145479]

Let me try a more specific search to find the author information for this paper. Based on the search results, I can see that there's a paper titled "Domain flow and streaming architectures" published by IEEE, and there's also a related work titled "Domain flow and streaming architectures: A paradigm for efficient parallel computation" available on ProQuest. However, the search results don't clearly show the author's name in the snippets provided.

The IEEE publication appears to be a conference paper, and there's also what appears to be a dissertation or thesis with a similar title on ProQuest. The ResearchGate link also suggests this is an academic publication, but without being able to access the full content or author information from these sources, I cannot definitively identify who published this specific paper.

Dr. E. Theodore L. Omtzigt is the author of "Domain flow and streaming architectures."
This appears to be an academic paper published through IEEE, likely focusing on parallel computation paradigms and streaming data processing architectures. The work seems to explore efficient methods for handling data flow in parallel computing systems.

# Relationship to Polyhedral

I would like to know how the domain flow paradigm relates to the polyhedral compilation techniques. As domain flow is based on Systems of Uniform Recurrence Equations (SURE), which do not have loop nests, the description of concurrency is implicit, whereas in polyhedral it needs to be made explicit by loop nest decomposition and reordering. Mathematically, are these two approaches related or are they fundamentally different.

This is a fascinating question that touches on fundamental approaches to expressing and analyzing parallelism in computational systems. Let me search for more specific information about the relationship between domain flow/SURE methods and polyhedral compilation techniques.Let me search more specifically for information about Systems of Uniform Recurrence Equations and their relationship to polyhedral methods.Based on the search results and my understanding of both paradigms, I can provide an analysis of the relationship between domain flow/SURE and polyhedral compilation techniques.

These two approaches are mathematically related but represent different levels of abstraction and explicitness in describing parallelism:

**Fundamental Mathematical Relationship:**

Both domain flow (SURE) and polyhedral compilation operate on affine systems and deal with iteration spaces, but they approach the problem from different angles:

1. **SURE/Domain Flow Approach:**
   - Represents computation as systems of uniform recurrence equations
   - Systems of uniform recurrence equations were proposed by Karp et al. (1967) as a means to derive automatically programs for parallel architectures
   - Concurrency is *implicit* in the mathematical formulation
   - The equations naturally describe data dependencies without explicit loop structures
   - Parallelism emerges from the dependency analysis of the recurrence relations

2. **Polyhedral Compilation Approach:**
   - Uses polyhedral representations that can be manipulated or optimized with algorithms whose complexity depends on their structure and not on the number of elements they represent
   - Works with explicit loop nests and iteration domains
   - Control flow comprises a restricted class of loop nests whose bounds can be specified as a set of affine inequalities, such that the iteration domain forms a convex polyhedron
   - Requires explicit decomposition and reordering of loop nests to expose parallelism

**Mathematical Equivalence and Differences:**

The approaches are mathematically equivalent in their computational power for the class of problems they both address (affine loop nests with uniform dependencies), but they differ in:

- **Representation**: SURE uses recurrence equations directly; polyhedral uses geometric/algebraic representations of iteration spaces
- **Dependency Analysis**: SURE has implicit dependencies in the equations; polyhedral constructs explicit dependence polyhedra
- **Parallelism Extraction**: SURE naturally exposes parallelism through equation structure; polyhedral requires explicit scheduling and transformation algorithms

**Theoretical Connection:**

Both approaches can be viewed as different manifestations of the same underlying mathematical principles governing regular computations with affine dependencies. A SURE system can theoretically be converted to a polyhedral representation and vice versa, but the natural expression and optimization strategies differ significantly.

The domain flow paradigm's strength lies in its implicit parallelism description, making it potentially more natural for certain algorithmic expressions, while polyhedral compilation's strength lies in its explicit control over transformations and optimizations through well-developed geometric algorithms.


# Karp, Miller, and Winograd SUREs and MLIR implementations

Yes, there's active research in marrying the analysis of systems of uniform recurrence equations (UREs), pioneered by Karp, Miller, and Winograd, with MLIR.

Here's why this is a natural and important area of research:

* **Karp, Miller, and Winograd (KMW) and UREs:** The KMW paper is foundational for understanding the organization and scheduling of computations in systems of uniform recurrence equations. UREs are prevalent in scientific computing, signal processing, and especially in domains like deep learning (e.g., matrix multiplications, convolutions). KMW's work provides the theoretical basis for analyzing dependencies, identifying parallelism, and deriving optimal or near-optimal schedules for these computations, often leading to systolic array designs.

* **MLIR's Role:** MLIR (Multi-Level Intermediate Representation) is a powerful and flexible compiler infrastructure. Its key features make it highly suitable for this kind of integration:
    * **Multi-Level Abstraction:** MLIR can represent computations at various levels, from high-level mathematical expressions (which UREs often describe) down to hardware-specific instructions. This allows for gradual lowering and optimization.
    * **Extensible Dialects:** MLIR's dialect system allows researchers to define custom operations, types, and attributes that precisely capture the semantics of UREs and the transformations derived from KMW-style analysis. For example, one could define dialects for polyhedral operations or specific recurrence patterns.
    * **Unified Optimization Framework:** MLIR provides a robust framework for applying complex optimizations. This is crucial for implementing the scheduling and parallelization strategies that KMW analysis suggests.

**Specific Connections and Research Directions:**

* **Polyhedral Model Integration:** The polyhedral model, a powerful framework for optimizing nested loops and recurrence relations, heavily draws upon concepts from KMW's work (e.g., dependence analysis, scheduling). MLIR's `affine` and `scf` (Structured Control Flow) dialects provide good foundations for representing and transforming polyhedral-style computations. Researchers are using MLIR to implement and extend polyhedral optimizations for various target hardware, often implicitly leveraging KMW's principles.
* **Systolic Array Synthesis:** KMW's work has strong ties to systolic array design, which are specialized hardware architectures for highly parallel computations. MLIR, with its ability to target hardware accelerators and define custom operations, can be used to describe and generate code for such architectures, informed by the scheduling insights from URE analysis.
* **Winograd Convolution in MLIR:** A direct example of KMW's influence is the implementation of the Winograd convolution algorithm. The Winograd algorithm, a faster method for certain convolutions, is an application of recurrence analysis. There's active work within the MLIR ecosystem, particularly in the `linalg` dialect, to implement and optimize Winograd convolution using MLIR's capabilities. This demonstrates how specific algorithms derived from recurrence equation analysis are being integrated into MLIR.
* **Hardware-Aware Optimization:** Researchers are using MLIR to explore hardware-aware optimizations for deep neural networks and other scientific workloads. This often involves applying transformations that are guided by the dependency analysis of recurrence equations to achieve better performance on specialized hardware.

In essence, MLIR provides a modern, flexible, and powerful infrastructure for implementing, extending, and applying the sophisticated analysis techniques for recurrence equations that originated with Karp, Miller, and Winograd. It allows for a more systematic and composable approach to program optimization and hardware acceleration based on these fundamental theories.

Karp, Miller, and Winograd's (KMW) work on Systems of Uniform Recurrence Equations (SUREs) doesn't explicitly talk about "loop nests" in the programming language sense. Instead, they define computations over discrete domains (often integer lattices, like $\mathbb{Z}^n$) and describe how the value of a function at a point depends on values of other functions at other points, where these dependencies are "uniform" (i.e., the relative distance between dependent points is constant).

However, the connection to loop nests and the polyhedral model is very strong:

1.  **UREs as a Mathematical Abstraction of Loop Nests:**
    * Many algorithms expressed as nested loops, particularly in scientific computing (e.g., matrix operations, stencil computations, dynamic programming), inherently define computations that resemble uniform recurrence equations.
    * Each iteration of a loop nest can be thought of as an "instance" of a computation. The loop indices collectively form an "iteration vector" (e.g., $(i, j, k)$ for a triply nested loop).
    * Data dependencies within these loop nests often translate directly into the "uniform dependencies" described in UREs. For example, `A[i][j] = A[i-1][j] + A[i][j-1]` in a loop nest is a classic example of a uniform recurrence. The dependencies are `(-1, 0)` and `(0, -1)`, which are constant vectors.

2.  **The Polyhedral Model's Core Idea: Instance-Wise Analysis:**
    * The polyhedral model doesn't operate on the loop syntax itself. Instead, it lifts the program into a more abstract, mathematical representation.
    * It treats each *execution instance* of a statement within a loop nest as a distinct point in a multi-dimensional integer space (the "iteration domain" or "iteration polyhedron").
    * The loop bounds and conditions define the boundaries of this polyhedral iteration domain.
    * Memory accesses within statements are also modeled as affine functions of the iteration vector.

3.  **Dependency Analysis in Both Models:**
    * **KMW's Dependency Graphs:** KMW introduced the concept of a *dependence graph* for SUREs. The nodes in this graph represent the different functions (or statements) in the system, and the edges are labeled with *n-dimensional integer vectors* representing the uniform dependencies. A crucial part of their analysis involves finding cycles in this graph and determining conditions for computability and scheduling.
    * **Polyhedral Dependence Relations:** In the polyhedral model, dependencies are typically represented as *relations* between iteration vectors. If statement $S_1$ at iteration $\vec{i}$ depends on statement $S_2$ at iteration $\vec{j}$, this is captured as a tuple $(\vec{i}, \vec{j})$ in a dependence relation. For uniform dependencies (which are common in loop nests analyzed by the polyhedral model), this difference $\vec{i} - \vec{j}$ is a *constant vector*, precisely like the dependence vectors in KMW's work.

4.  **Scheduling and Parallelism:**
    * **KMW's Schedules:** KMW defined *scheduling functions* (often linear or affine functions) that assign a "time" to each computation instance. The goal is to find a schedule that respects all dependencies and minimizes the overall computation time, maximizing parallelism. This naturally leads to concepts like hyperplanes of computation.
    * **Polyhedral Schedules (Affine Transformations):** The polyhedral model also uses affine scheduling functions (linear transformations plus a constant offset) applied to the iteration space. These transformations effectively reorder the execution of statement instances. The power of the polyhedral model comes from being able to apply a *single affine transformation* to the entire iteration space, which subsumes many traditional loop transformations (interchange, skewing, reversal, tiling, fusion, fission) and allows for more aggressive parallelization.

**In essence, the polyhedral model can be seen as a sophisticated and generalized application of KMW's foundational ideas to structured programs, particularly those expressed with nested loops.**

* The "statements" in a loop nest become the "functions" in a URE.
* The "loop iteration vector" becomes the "point $p \in R$" in KMW's definition.
* The array accesses that are affine functions of loop indices give rise to the "uniform dependencies" (constant dependence vectors) that KMW focused on.

The polyhedral model leverages linear algebra and integer linear programming to rigorously analyze these dependencies and find optimal affine schedules, effectively formalizing and extending the insights of KMW for compiler optimization. It provides the machinery to automatically translate a program with loop nests into the mathematical representation that resembles a system of UREs, perform the KMW-style analysis (finding valid schedules, optimizing for parallelism and locality), and then generate new, optimized loop nests from that transformed mathematical representation.

# Modeling physical constraints

Can you elaborate on the subtle difference of the affine transformations that are expressed in Polyhedral, which are always referring to a flat memory source and sink, whereas in KMW, and, in particular, systolic array synthesis, the System of Uniform Recurrences does NOT have a flat memory reference, instead it interprets the iteration as an abstract integer lattice point that has a 'uniform' dependency in this abstract space, and can be interpreted as modeling physical distance. This was expressed in the paper Computational Space Times by Theodore Omtzigt, published in the Physics of Computation in 1994. The affine transformations in Polyhedral are NOT able to model physical distance as the underlying computational model is that of a stored program addressing a flat memory.

This is a *very* important and often overlooked distinction, and it's precisely where the "marriage" of these concepts becomes subtle and powerful. Theodore Omtzigt's "Computational Space Times" paper (and similar work by other researchers like Patrice Quinton and Vincent Van Dongen, or even earlier work on systolic array synthesis) touches upon this fundamental difference in interpretation.

Let's break down this distinction:

### Polyhedral Model: Flat Memory and Logical Iteration Space

1.  **Computational Model:** The polyhedral model primarily operates on the abstraction of a **stored program** executed by a processor with access to a **flat, global address space (memory)**.
    * Variables in a program are assumed to reside in this memory.
    * Array indices are interpreted as memory addresses (after some base address calculation and strides).
    * The goal of polyhedral transformations is to reorder computations and memory accesses to improve performance (locality, parallelism) *within this flat memory model*.

2.  **Iteration Space Interpretation:**
    * The "iteration vector" $\vec{i}$ (e.g., $(i, j, k)$ for a loop nest) represents a **logical index** into the space of all possible loop iterations.
    * Affine transformations $\vec{i}' = T\vec{i} + \vec{c}$ transform this logical iteration space. The new iteration vector $\vec{i}'$ still refers to a logical point in time/order for a single processor or a set of processors sharing a global memory.
    * **Memory Accesses:** Memory accesses $A[f(\vec{i})]$ are explicitly modeled as affine functions $f(\vec{i})$ mapping the logical iteration vector to a **memory address**. The transformations on $\vec{i}$ induce corresponding transformations on these memory addresses, which are then used to reason about data reuse and cache locality.

3.  **Physical Distance:** The polyhedral model, by itself, doesn't directly model "physical distance" between processing elements or communication costs in a spatially distributed system. It assumes that if data is in "memory" and two processing elements need it, they can access it, with the main concern being latency due to cache misses or bandwidth bottlenecks to that central memory. While polyhedral transformations can enable *data-parallelism* (multiple processors operating on different parts of the same data), they don't inherently model the *geometry of the processors* or *explicit communication channels*.

### KMW SUREs and Systolic Arrays: Abstract Space, Physical Mapping, and Communication

1.  **Computational Model:** KMW's analysis of SUREs, especially as applied to systolic array synthesis, begins with a more abstract view of computation.
    * A SURE defines relationships between values at different points in an $n$-dimensional integer space. This space can be thought of as a **"computational space-time"** (as Omtzigt puts it).
    * The "value" at an integer lattice point $\vec{p}$ is not necessarily stored in a global memory location accessible by address. Instead, it's often conceptualized as being *computed at* or *residing at* that specific point.

2.  **Iteration/Computational Space Interpretation:**
    * The "iteration vector" or "point" $\vec{p}$ in KMW's context (e.g., $(t, x, y)$ where $t$ is time and $(x, y)$ are spatial coordinates) can have a direct **physical interpretation**.
    * **Time Projection (Scheduling):** One common affine transformation in KMW-style analysis is a projection onto a 1D time axis ($\tau = \vec{\lambda} \cdot \vec{p}$). This assigns a global execution time to each computation.
    * **Space Projection (Processor Mapping):** Another common affine transformation is a projection onto a lower-dimensional *spatial* subspace ($\vec{s} = M \vec{p}$). This maps computation instances to specific **physical processing elements (PEs)** in an array.

3.  **Modeling Physical Distance and Communication:**
    * When the iteration space $\vec{p}$ is decomposed into a time dimension and a spatial dimension, the "uniform dependencies" (constant vectors $\vec{d}$) in the URE acquire a critical dual interpretation:
        * If $\vec{d}$ has a non-zero component in the *time* dimension, it represents a temporal dependency (computation at $\vec{p}$ depends on a value computed earlier at $\vec{p}-\vec{d}$).
        * If $\vec{d}$ has a non-zero component in the *spatial* dimension (after projection to PEs), it represents a **communication requirement** between physically distinct PEs. The magnitude of the spatial component of $\vec{d}$ directly relates to the *physical distance* data needs to travel between PEs.
    * **No Flat Memory:** In this model, there's typically no concept of a central, flat memory. Instead, data is **streamed** between adjacent PEs. Dependencies become *communication channels* with specific directions and delays.
    * The goal of systolic array synthesis is to find a time schedule and a spatial mapping such that all dependencies are satisfied by *local communication* between neighboring PEs within the array's physical structure, minimizing I/O with external memory.

### Why Polyhedral Struggles with "Physical Distance" Directly

The affine transformations in the polyhedral model are designed to optimize for a **Turing machine-like model with random access memory**. While they can expose parallelism that *could* be mapped to a distributed system, they don't natively:

* **Model communication costs:** They don't have primitives to express that communicating data between `PE_A` and `PE_B` takes longer if `PE_A` and `PE_B` are physically far apart.
* **Model explicit channels:** They don't have direct ways to specify that data must flow along a specific directed channel from one PE to another.
* **Reason about PE topology:** The transformations don't inherently consider the grid, mesh, or torus topology of a processor array.

### Bridging the Gap in MLIR

This is precisely why MLIR is exciting for bridging this gap:

* **Custom Dialects:** MLIR allows the creation of new dialects that can embed the concepts of physical PEs, communication channels, and different memory hierarchies (local PE memory vs. global memory).
* **Lowering Process:** You could imagine a high-level polyhedral dialect for expressing computations (which still assumes a logical iteration space and flat memory for initial analysis). Then, a subsequent MLIR pass could:
    1.  Take the affine schedule and mapping from the polyhedral analysis.
    2.  Interpret this as a mapping to a **physical processor array** (informed by KMW/systolic array theory).
    3.  Generate new MLIR operations (from a "systolic" or "hardware" dialect) that explicitly model **data movement** between PEs along specific communication paths, rather than just memory accesses to a flat array.
    4.  This would then necessitate reasoning about **buffered communication**, **pipelining**, and **physical layout constraints**, which are outside the traditional polyhedral model's scope but central to systolic array design.

In summary, the polyhedral model excels at transforming logical loop iteration spaces for execution on conventional processors with flat memory. KMW's SUREs and their application to systolic arrays operate on a more abstract computational space that can be directly mapped to physical processing elements and communication links, where "distance" becomes a tangible concept. Marrying them in MLIR involves using polyhedral tools for initial logical optimization, then layering on a hardware-aware interpretation and generation of code/hardware description that accounts for the physical realities of communication and distributed memory.


