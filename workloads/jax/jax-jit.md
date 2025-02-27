# JAX programs

To compile a JAX program with `iree-compiler` to an MLIR bytecode file, you'll need to follow a series of steps that involve exporting the JAX program to a format that IREE can understand (ideally, MLIR), and then using `iree-compiler` to process that MLIR.

Here's a breakdown of the process:

**1. Exporting JAX to MLIR (or a Compatible Format):**

* **JAX to StableHLO:**
    * The most common and recommended approach is to use JAX's ability to export to StableHLO. StableHLO is a stable and portable representation of HLO (High-Level Optimizer), which is the intermediate representation used by XLA (Accelerated Linear Algebra).
    * StableHLO has excellent MLIR support, making it ideal for IREE.
    * You can use the `jax.jit` decorator along with `jax.experimental.mlir.mlir_module` to get a StableHLO MLIR module.
* **Example Code (JAX to StableHLO):**

```python
    import jax
    import jax.numpy as jnp
    from jax.experimental import mlir

    def my_jax_function(x, y):
        return jnp.sin(x) + y

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    jitted_func = jax.jit(my_jax_function)
    mlir_module = mlir.mlir_module(jitted_func.lower(x, y).compiler_data(dialect='stablehlo'))

    with open("jax_module.mlir", "w") as f:
        f.write(str(mlir_module))
```

* **Saving the MLIR:**
    * The `mlir_module` object now contains the StableHLO MLIR representation of your JAX function.
    * Save this MLIR to a `.mlir` file (e.g., `jax_module.mlir`).

**2. Compiling with `iree-compiler`:**

* **Using `iree-compile`:**
    * Now, use the `iree-compile` command-line tool to compile the saved MLIR file.
    * You'll need to specify the target backend (e.g., `local-task`, `vulkan`, `vmvx`).
    * Example command for cpu target:

```bash
iree-compile jax_module.mlir --output=jax_module.vmfb --iree-hal-target-backends=local-task
```

* **Target Backend:**
    * Replace `local-task` with the desired backend.
    * For GPU execution, use `vulkan`.
    * For the software VM, use vmvx.
* **Output File:**
    * The `--output` flag specifies the name of the output VMFB file (e.g., `jax_module.vmfb`).

**3. Running the VMFB:**

* **Using `iree-run-module`:**
    * To run the compiled VMFB, use the `iree-run-module` command-line tool.
    * You will need to pass the input data to the module.
    * This step is very dependant on the inputs and outputs of the MLIR module that was created.

**Important Considerations:**

* **IREE Installation:**
    * Ensure that you have IREE installed and that `iree-compile` and `iree-run-module` are in your PATH.
* **Dependencies:**
    * Make sure you have JAX and its dependencies installed.
* **Target Compatibility:**
    * The JAX program must be compatible with the target backend that you choose.
    * Not all JAX operations may be supported by all IREE backends.
* **Data Types:**
    * Ensure that the data types used in your JAX program are compatible with IREE.
* **StableHLO Version:**
    * Keep your StableHLO version up to date. IREE is designed to work with the latest versions.

By following these steps, you should be able to compile your JAX program with `iree-compiler` to an MLIR bytecode file (VMFB).

