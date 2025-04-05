from mlir.ir import Context, Module, Operation, Value
import sys

def walk_mlir_graph(mlir_input):
    # Initialize the MLIR context
    with Context() as ctx:
        # Allow unregistered dialects (like TOSA) for simplicity
        ctx.allow_unregistered_dialects = True

        # Parse the MLIR input
        module = Module.parse(mlir_input)
        
        # Traverse the module to find the function
        for func in module.body:
            if func.is_declaration():  # Skip declarations
                continue
            print(f"Walking function: {func.name.value}")
            print("Graph Node Analysis:")
            print("-" * 50)

            # Walk through all operations in the function
            def walk_operations(op, indent=0):
                # Get operation name and result(s)
                op_name = op.operation.name
                results = op.operation.results
                result_str = ", ".join(str(r) for r in results) if results else "None"

                # Get operands (inputs) to this operation
                operands = op.operation.operands
                inputs_str = ", ".join(str(o) for o in operands) if operands else "None"

                # Print node details
                indent_str = "  " * indent
                print(f"{indent_str}Node: {op_name}")
                print(f"{indent_str}  Inputs: {inputs_str}")
                print(f"{indent_str}  Outputs: {result_str}")

                # Recursively walk nested operations (if any, though not in this example)
                for region in op.regions:
                    for block in region:
                        for nested_op in block:
                            walk_operations(nested_op, indent + 1)

            # Start walking from the function's entry block
            for block in func.blocks():
                for op in block:
                    walk_operations(op)

            print("-" * 50)
            print("Summary of Node Communications:")
            print("-" * 50)

            # Analyze dependencies by tracking uses of each result
            value_to_op = {}  # Map Value to its producing operation
            for block in func.keys():
                for op in block:
                    for result in op.operation.results:
                        value_to_op[result] = op.operation.name

            for block in func.keys():
                for op in block:
                    op_name = op.operation.name
                    for operand in op.operation.operands:
                        if operand in value_to_op:
                            producer = value_to_op[operand]
                            print(f"{producer} -> {op_name} (via {operand})")
                    if op.operation.results:
                        result = op.operation.results[0]
                        for use in result.uses:
                            consumer = use.owner.name
                            print(f"{op_name} -> {consumer} (via {result})")

# Example MLIR input (your provided snippet)
mlir_input = """
module { 
  func.func @simple_function(%arg0: tensor<1x1x4xf32>, %arg1: tensor<1x4x4xf32>) -> tensor<1x1x4xf32> { 
    %0 = "tosa.const"() {value = dense<1.0> : tensor<1x1x4xf32>} : () -> tensor<1x1x4xf32> 
    %1 = "tosa.add"(%arg0, %0) : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4xf32> 
    %2 = "tosa.matmul"(%1, %arg1) : (tensor<1x1x4xf32>, tensor<1x4x4xf32>) -> tensor<1x1x4xf32> 
    return %2 : tensor<1x1x4xf32> 
  } 
}
"""

if __name__ == "__main__":
    walk_mlir_graph(mlir_input)