# MLIR Reader and Writer

In the MLIR repository, serialization is primarily handled in the following directories and files:

1. `mlir/lib/Support/StorageUniquer.cpp` - This file contains core serialization infrastructure
2. `mlir/include/mlir/Support/StorageUniquer.h` - Header file for serialization support
3. `mlir/lib/Serialization/` - A dedicated directory for serialization-related implementations

More specifically, for reading and writing MLIR format, you'll want to look at:

- `mlir/lib/Serialization/MLIRReader.cpp`
- `mlir/lib/Serialization/MLIRWriter.cpp`

These files contain the core logic for parsing and generating MLIR's text and binary formats. The reader and writer implementations should give you insights into how MLIR handles graph serialization.

If you're specifically interested in graph serialization for deep learning, you might also want to examine:

- `mlir/lib/Dialect/Func/IR/` for function-level serialization
- `mlir/lib/Dialect/Tensor/IR/` for tensor-related serialization

## Libraries and Executables for MLIR serialization

Key Libraries:
1. `libMLIRSupport.a/so` - Contains core serialization infrastructure
2. `libMLIRSerialization.a/so` - Handles MLIR reading and writing
3. `libMLIRFuncDialect.a/so` - Used for function-level serialization
4. `libMLIRTensorDialect.a/so` - Manages tensor serialization

Key Executables that use these serialization mechanisms:
1. `mlir-opt` - MLIR optimization tool that uses serialization
2. `mlir-translate` - Converts between MLIR and other representation formats
3. `mlir-reduce` - Uses serialization for test case reduction
4. `mlir-pdll` - Pattern Descriptor Language tool that relies on serialization

These libraries and executables are typically built as part of the LLVM/MLIR build system using CMake. 
They're compiled into the `build/lib/` and `build/bin/` directories respectively when you build the MLIR project.

