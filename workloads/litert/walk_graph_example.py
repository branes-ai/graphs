from iree.compiler.ir import Context, Module, Operation, Value
import sys

def walk_mlir_graph(mlir_input):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(mlir_input)
        
        for func in module.body:
            if not func.body:  # Skip if the function has no body (i.e., a declaration)
                continue
            print(f"Walking function: {func.name.value}")
            print("Graph Node Analysis:")
            print("-" * 50)

            def walk_operations(op, indent=0):
                op_name = op.operation.name
                results = op.operation.results
                result_str = ", ".join(str(r) for r in results) if results else "None"
                operands = op.operation.operands
                inputs_str = ", ".join(str(o) for o in operands) if operands else "None"
                indent_str = "  " * indent
                print(f"{indent_str}Node: {op_name}")
                print(f"{indent_str}  Inputs: {inputs_str}")
                print(f"{indent_str}  Outputs: {result_str}")
                for region in op.regions:
                    for block in region:
                        for nested_op in block:
                            walk_operations(nested_op, indent + 1)

            block = func.body.blocks[0]  # Get the first block
            for op in block:
                walk_operations(op)

            print("-" * 50)
            print("Summary of Node Communications:")
            print("-" * 50)

            value_to_op = {}
            block = func.body.blocks[0]  # Get the first block
            for op in block:
                for result in op.operation.results:
                    value_to_op[result] = op.operation.name

            block = func.body.blocks[0]  # Get the first block
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

mlir_input = """
module { 
  func.func @simple_function(%arg0: tensor<1x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<1x4xf32> { 
    %0 = "tosa.const"() {values = dense<1.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> 
    %1 = "tosa.add"(%arg0, %0) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32> 
    %init = "tosa.const"() {values = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> 
    %2 = "linalg.matmul"(%1, %arg1, %init) <{operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%a: f32, %b: f32, %c: f32):
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        linalg.yield %add : f32
    }) : (tensor<1x4xf32>, tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32> 
    return %2 : tensor<1x4xf32> 
  } 
}
"""

if __name__ == "__main__":
    walk_mlir_graph(mlir_input)