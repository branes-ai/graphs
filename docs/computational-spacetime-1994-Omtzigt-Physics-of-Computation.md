# Computational SpaceTime

Theodore Omtzigt's 1994 paper "Computational Space Times," published in the *Physics of Computation* (as part of a broader body of work on "Domain Flow" and streaming architectures), proposes a fundamental shift in how we model and design parallel computations, particularly in light of evolving hardware constraints.

Here's a summary of its key ideas:

1.  **The Crisis of the Von Neumann Model:** Omtzigt argues that the traditional von Neumann computational model (with its central processor and a single, flat, infinite memory) is increasingly inefficient. As integrated circuit technology miniaturizes, computation delays are shrinking faster than communication delays. This means that moving data becomes the dominant cost, both in terms of time and energy, making the "flat memory" abstraction increasingly misleading for high-performance parallel systems.

2.  **Introducing Computational Spacetime:**
    * Inspired by Minkowski spacetime in physics, Omtzigt proposes a **"computational spacetime"** as a more accurate model for physical machines.
    * In this model, computational events are not just points in abstract time, but occupy specific **spatial locations** as well.
    * The crucial insight is that there's a **limit on the speed of signal propagation** within a computational system (analogous to the speed of light). This limit means that a computational event can only influence other events within its "cone of influence" (similar to a light cone).
    * This makes explicit the **spatial and temporal constraints** on computation and communication.

3.  **Spatial Single-Assignment Form:**
    * To leverage this spacetime model, Omtzigt advocates for algorithms to be specified in a **"spatial single-assignment form."** This is a form of dataflow or functional programming where every variable is assigned a value only once, and crucially, all assignments are **spatially explicit**.
    * This explicitly exposes the data dependencies and their associated spatial and temporal relationships, which are critical for efficient parallel execution.

4.  **Physical Distance as a Primary Metric:**
    * Unlike traditional models where memory access is often considered uniform or only distinguished by cache levels, Computational Spacetime inherently incorporates **physical distance** as a key factor.
    * Minimizing data movement and localizing computation becomes paramount because distance is directly proportional to communication delay and energy consumption.
    * This leads to algorithms where data is streamed between neighboring processing elements, rather than being fetched from a global memory.

5.  **Reinterpreting Uniform Recurrence Equations (UREs):**
    * While not explicitly stated as "KMW," Omtzigt's framework directly builds on the mathematical foundations of UREs (or dependence graphs of uniform recurrences).
    * He interprets the *iteration space* of a URE as a multi-dimensional integer lattice where each point represents a distinct computational event.
    * The "uniform dependencies" (constant vectors) in a URE are then mapped directly to **physical communication paths** between processing elements in the computational spacetime. The vector's components define both the temporal delay and the spatial distance/direction of data flow.

6.  **Domain Flow Paradigm:**
    * This spacetime model is part of a larger **"Domain Flow"** paradigm. This paradigm aims to provide a path from sequential computation to efficient high-speed parallel computation by:
        * Shielding casual programmers from parallel complexities.
        * Facilitating libraries of parallel subroutines.
        * Integrating the spacetime model of spatial execution with more established sequential models.
    * It treats computations as "domains" of data flow graphs that have both temporal and spatial extent, and these domains can flow through the system.

7.  **Implications for Architecture and Algorithms:**
    * The paper argues that computer architectures need to quantify time and distance explicitly, rather than hiding them behind a flat memory abstraction.
    * Optimal algorithms for these architectures will be those that minimize data movement and maximize local communication, often resembling systolic or array-processor designs.
    * The goal is to design "uniform architectures" (like orthonormal lattices of PEs) where computation and communication delays remain constant across scaling, allowing algorithms to remain optimal across technology generations.

In essence, Omtzigt's "Computational Space Times" paper is a call to action for computer architects and algorithm designers to break free from the limitations of the von Neumann model and to explicitly account for the physical reality of communication and spatial distribution in the design of high-performance parallel systems, using a computational spacetime model and favoring spatially explicit, dataflow-oriented algorithmic descriptions derived from the principles of uniform recurrence equations.