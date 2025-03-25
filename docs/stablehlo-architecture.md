# StableHLO Workflow

The StableHLO project and the role of `stablehlo-opt`.

**StableHLO: The Specification**

* **Purpose:**
    * StableHLO is primarily a specification for a portable, high-level intermediate representation (IR) for machine learning (ML) models.
    * It aims to provide a stable interface between ML frameworks (like TensorFlow, JAX, PyTorch) and ML hardware backends (like GPUs, TPUs, CPUs).
    * This allows framework developers to target a single, well-defined IR, and hardware vendors to implement a single backend that works with various frameworks.
* **Key Aspects:**
    * It defines a set of operations (ops) that represent common ML computations.
    * It specifies the data types and semantics of these ops.
    * It is designed to be extensible and adaptable to new ML hardware and software.
    * The specification is the documentation that you see on the openxla.org website.
* **Role in the ML Pipeline:**
    * Frameworks serialize their model representations into StableHLO.
    * Hardware backends consume StableHLO and execute the model on the target hardware.

**`stablehlo-opt`: The Tool**

* **Purpose:**
    * `stablehlo-opt` is a command-line tool within the StableHLO source tree that acts as a pass runner.
    * It's built using the MLIR (Multi-Level Intermediate Representation) infrastructure.
    * It's used to perform transformations and optimizations on StableHLO programs.
* **Functionality:**
    * It applies a sequence of passes to a StableHLO program, such as:
        * Simplifying operations.
        * Fusing operations.
        * Lowering high-level ops to lower-level ops.
        * Performing target-specific optimizations.
    * These passes are essential for preparing StableHLO programs for efficient execution on hardware backends.
* **Relationship to the Specification:**
    * `stablehlo-opt` operates on StableHLO programs that adhere to the specification.
    * It's a tool for manipulating and optimizing these programs, not the specification itself.
    * Think of the specification as the rules of a game, and `stablehlo-opt` as a tool that helps you play the game more efficiently.
* **Role in Development:**
    * Developers use `stablehlo-opt` to test and refine StableHLO programs.
    * It's also used to develop and test new StableHLO passes.

**How the StableHLO Source Tree is Used**

The StableHLO source tree, primarily within its GitHub repository, serves several key functions:

1.  **Implementation of the Specification:**
    * It contains the MLIR dialect that defines the StableHLO operations.
    * This dialect is the concrete implementation of the abstract specification.
2.  **Development of Tools:**
    * It provides tools like `stablehlo-opt` for working with StableHLO programs.
    * It also includes other utilities for parsing, printing, and manipulating StableHLO.
3.  **Development of Passes:**
    * It's the place where developers implement and contribute new optimization and transformation passes.
4.  **Testing and Validation:**
    * It provides tests to ensure that StableHLO programs are handled correctly.
    * It also tests that the passes within `stablehlo-opt` function as intended.
5.  **Example and Reference:**
    * The source code serves as a reference implementation for how to work with StableHLO.

**In Summary**

* StableHLO is a specification for a portable ML IR.
* `stablehlo-opt` is a tool for optimizing and transforming StableHLO programs.
* The StableHLO source tree contains the implementation, tools, and tests related to the StableHLO specification.

## Why does the StableHLO repo generate a `opt` tool?

If there is the StableHLO specification, why is there a `stablehlo-opt` tool? This gets to the heart of StableHLO's role in the ML ecosystem. Here's a breakdown of why `stablehlo-opt` exists, even with dedicated compilers like XLA and IREE:

**1. StableHLO as a Portable IR Needs Optimization:**

* **Framework Agnostic:**
    * StableHLO is designed to be a common IR for *multiple* ML frameworks. This means it needs to handle a wide range of model representations and optimizations that might not be specific to any single framework.
    * Frameworks produce StableHLO, but they may not perform all optimizations that a hardware backend needs. `stablehlo-opt` fills this gap.
* **Hardware Agnostic (to a Point):**
    * While backends like XLA and IREE handle hardware-specific lowering, StableHLO itself benefits from optimizations that are more general.
    * For example, operation fusion, constant folding, and shape inference can be performed at the StableHLO level, reducing the burden on individual backends.
* **A "Middle Ground" Optimization Layer:**
    * `stablehlo-opt` provides a layer of optimization that sits between the framework's output and the backend's input.
    * This allows for optimizations that are not tied to a specific framework or hardware target.

**2. Facilitating Backend Development and Interoperability:**

* **Standardized Optimization Passes:**
    * By providing a set of standard optimization passes within `stablehlo-opt`, StableHLO ensures that all backends receive a reasonably well-optimized input.
    * This helps to ensure consistent performance across different hardware platforms.
* **Simplifying Backend Complexity:**
    * Hardware vendors can focus on implementing hardware-specific optimizations within their backends, rather than having to deal with a wide range of framework-specific variations.
    * `stablehlo-opt` handles many of the general optimizations.
* **Enabling Research and Development:**
    * Having a common set of optimization tools enables research into optimizations that are applicable to a broad range of ML hardware.
    * It also allows for the easy testing and comparison of different optimization strategies.

**3. MLIR Infrastructure and Extensibility:**

* **MLIR's Pass Pipeline:**
    * MLIR's pass pipeline is a powerful and flexible mechanism for performing IR transformations.
    * `stablehlo-opt` leverages this infrastructure to provide a rich set of optimization passes.
* **Extensibility:**
    * The MLIR-based design of StableHLO and `stablehlo-opt` allows for easy extension and customization.
    * New optimization passes can be added to `stablehlo-opt` as needed.

**In essence:**

* While XLA and IREE are crucial for hardware-specific lowering and optimization, `stablehlo-opt` provides a necessary layer of general-purpose optimization that enhances the portability and interoperability of StableHLO.
* It is a tool that helps to bridge the gap between ML frameworks and ML hardware backends, enabling a more standardized and efficient ML compilation pipeline.

Therefore, `stablehlo-opt` is not redundant; it plays a complementary role to dedicated DL compilers, ensuring that StableHLO achieves its goal of being a truly portable and efficient ML IR.


## Understand that StableHLO is under construction: many developers, many claims, many opinions

**1. StableHLO's Focus on Specification Over Optimization:**

* StableHLO's primary goal is to be a *specification*. The emphasis is on defining a stable, portable IR, not on providing a complete, prescriptive optimization pipeline.
* The developers might have intended for hardware backends and compiler frameworks to provide their own optimization passes tailored to their specific needs.
* This aligns with the idea of StableHLO being a "contract" between frameworks and backends, where the contract specifies the IR, but the implementation of optimizations is left to the individual parties.

**2. MLIR's Flexibility and Implicit Pipelines:**

* MLIR's pass pipeline is very flexible. It allows for a wide range of pass combinations and ordering.
* The developers might have assumed that users would leverage MLIR's flexibility to build their own pipelines, rather than relying on a predefined, documented one.
* Many MLIR based projects rely on the developers of the compiler or backend to create the pass pipeline that best fits their needs.
* The MLIR community often emphasizes "composition over configuration," which means that users are expected to assemble passes as needed.

**3. Evolving Nature of StableHLO:**

* StableHLO is a relatively new project, and it's still evolving.
* The optimization landscape for ML hardware is constantly changing, and the developers might be hesitant to commit to a specific pass pipeline that could become obsolete quickly.
* The focus has been on getting the specification solidified, and optimization pipelines may be an area of future development.

**4. The "Opt" Tool As a Testbed:**

* The `stablehlo-opt` is being used as a test bed for the StableHLO dialect, not as a full-fledged optimization tool.
* It is used to test passes as they are developed, and to make sure that the dialect itself is working correctly.
* Production level optimization would be expected to happen in the compilers that are consuming the StableHLO IR, such as XLA and IREE.

Pass documentation can be found at [stablehlo/docs/generated](https://github.com/openxla/stablehlo/tree/main/docs/generated) 
or at the website [stablehlo_passes](https://openxla.org/stablehlo/generated/stablehlo_passes).

Also, [iree-opt](https://iree.dev/developers/general/developer-overview/#iree-opt) and [stablehlo APIs](https://openxla.org/stablehlo/compatibility#apis) may be useful too. 
Each project has its own *-opt tool that is mostly used for testing individual passes. 
Passes are assembled into pipelines in *-translate tools or higher level / more user-friendly (vs compiler-developer-friendly) tools like IREE's iree-compile or Python/C/etc. APIs



