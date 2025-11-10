# Project Overview

This repository contains "graphs," a Python-based toolkit for analyzing and characterizing the performance of neural networks on a variety of hardware platforms. The project provides a suite of command-line interface (CLI) tools to analyze PyTorch models, understand their hardware mapping, and predict performance without relying on purely theoretical estimates.

The core of the project involves:
- **Graph Partitioning:** Decomposing PyTorch FX graphs into smaller, analyzable subgraphs.
- **Hardware Mapping:** Modeling the performance of these subgraphs on over 32 different hardware targets, including CPUs, GPUs, TPUs, and specialized AI accelerators.
- **Performance Analysis:** Providing detailed insights into metrics like FLOPs, memory bandwidth, arithmetic intensity, and potential bottlenecks.

The project is structured around a core `src/graphs` library and a rich set of `cli/` tools for user interaction.

# Building and Running

The project is Python-based and requires `torch`, `torchvision`, and `pandas`.

## Installation

The primary dependencies can be installed via pip:

```bash
pip install torch torchvision pandas
```

Some tools may have additional, optional dependencies (e.g., `fvcore`).

## Running Tools

The main way to interact with this project is through its command-line tools located in the `cli/` directory. All commands should be run from the root of the repository.

### Key Commands:

*   **List available hardware mappers:**
    ```bash
    python cli/list_hardware_mappers.py
    ```

*   **Analyze a model on a specific hardware target:**
    ```bash
    python cli/analyze_graph_mapping.py resnet18 --hardware H100
    ```

*   **Compare multiple hardware targets for a given model:**
    ```bash
    python cli/analyze_graph_mapping.py resnet18 --compare "H100,A100,Jetson-Orin-AGX"
    ```

*   **Profile a model's computational graph:**
    ```bash
    python cli/profile_graph.py resnet18
    ```

*   **Run a comprehensive analysis (roofline, energy, memory):**
    ```bash
    python cli/analyze_comprehensive.py --model resnet18 --hardware H100
    ```

*   **Analyze the impact of batch size:**
    ```bash
    python cli/analyze_batch.py --model resnet18 --hardware H100 --batch-size 1 2 4 8 16 32
    ```

# Development Conventions

*   **Primary Interface:** The project is primarily CLI-driven. New functionality is often exposed as a new script in the `cli/` directory.
*   **Documentation:** There is a strong emphasis on documentation. The `docs/` directory contains detailed guides, and each CLI tool has its own documentation in `cli/docs/`. The `README.md` and `cli/README.md` files serve as excellent starting points.
*   **Testing:** The `tests/` directory suggests a structured testing approach, with subdirectories for different components of the project (hardware, analysis, IR, etc.). `pytest.ini` indicates that `pytest` is the testing framework.
*   **Modularity:** The code is organized into modules within the `src/graphs` directory, covering areas like intermediate representation (`ir`), transformations (`transform`), analysis (`analysis`), and hardware modeling (`hardware`).
*   **Examples:** The `examples/` directory provides scripts that demonstrate how to use the core functionalities of the library.
