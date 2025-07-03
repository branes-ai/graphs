\# Torch Compile Limitations



You can execute data-dependent PyTorch programs with an IREE compiler/runtime setup, but there are important considerations and nuances.



\## How IREE Handles PyTorch and Data Dependency:



1\. \*\*PyTorch Integration:\*\* IREE works with PyTorch through `torch.compile` (using TorchDynamo and iree-turbine) or by exporting models ahead-of-time (AOT) using `iree-turbine.aot.export()`. This process translates PyTorch operations into MLIR (Multi-Level Intermediate Representation), which IREE then compiles.

2\. \*\*Graph-Based Compilation:\*\* Compilers like IREE generally prefer static, well-defined computation graphs. This is because they perform extensive optimizations (fusion, hoisting, constant evaluation, etc.) that benefit from knowing the entire flow of data and operations upfront.

3\. \*\*Data-Dependent Control Flow:\*\* This is where things get tricky. "Data-dependent" often refers to control flow (e.g., `if`/`else` statements, loops) where the path of execution depends on the \*values\* of tensors at runtime, rather than just their shapes.

4\. \*\*`torch.export` and `cond`:\*\* PyTorch's `torch.export` (which IREE leverages) \*does\* support data-dependent control flow, but it needs to be expressed using explicit control flow operators like `torch.cond`. If your PyTorch code uses standard Python `if`/`else` statements that operate on tensor values, `torch.compile` (and thus IREE) might either "graph break" (fall back to eager PyTorch execution for that part) or require you to restructure your code to use these explicit control flow ops.

5\. \*\*Dynamic Shapes:\*\* IREE, along with `torch.export`, also supports dynamic shapes. This means that the \*size\* of your tensors can vary at runtime, as long as the operations themselves are amenable to such dynamicism. You can specify dynamic dimensions using `Dim.AUTO` in `dynamic\_shapes` when exporting.

6\. \*\*IREE's Internal Handling:\*\* Internally, IREE's MLIR dialects (like `flow` and `hal`) have mechanisms to represent and optimize control flow, including conditional dispatches (`flow.cond\_dispatch`) which allow execution to depend on prior computation results. However, getting your PyTorch code to translate into these optimized IREE constructs is the key.



\## Challenges and Solutions:



1\.  \*\*Implicit Python Control Flow:\*\* If your PyTorch program has `if`/`else` or `for` loops whose conditions depend on tensor values and are written as standard Python control flow, `torch.compile` might encounter a "graph break."

&nbsp;   \* \*\*Solution:\*\* Refactor your code to use PyTorch's functional control flow primitives where possible (e.g., `torch.cond`, `torch.while\_loop`, `torch.map`). This allows the entire computation, including the conditional logic, to be captured in the graph for IREE to compile.

2\.  \*\*Unsupported Operations:\*\* Not all PyTorch operations are directly supported or have optimized lowering paths in IREE. Data-dependent operations that involve complex Python logic might fall into this category.

&nbsp;   \* \*\*Solution:\*\* Consult IREE's documentation and PyTorch's `torch.compile` FAQs for known limitations and supported operations. You might need to use "escape hatches" (`torch.\_dynamo.disable`, `torch.\_dynamo.disallow\_in\_graph`) to prevent compilation of unsupported parts, leading to a hybrid execution where some parts run in IREE and others in eager PyTorch.

3\.  \*\*Dynamic Tensor Data in Operations:\*\* While IREE supports dynamic shapes, operations that truly depend on the \*values\* of tensors for their structure (e.g., creating a tensor of a size determined by a runtime tensor value, or indexing into an array with a value-dependent index) can be more challenging.

&nbsp;   \* \*\*Solution:\*\* This is an area of ongoing development in compilers. For many common neural network patterns, this is handled. For highly custom or unusual data-dependent logic, you might hit limitations that require careful restructuring or a deeper understanding of MLIR and IREE's capabilities.



\## Summary:



You absolutely can execute data-dependent PyTorch programs with IREE. The critical factor is how those data dependencies are expressed in your PyTorch code. By leveraging `torch.compile` and its underlying `torch.export` functionality, and by structuring your PyTorch code to use explicit functional control flow operators when dealing with data-dependent logic, you can enable IREE to compile and efficiently execute your programs.



For the best results, aim for a clear, traceable computation graph, and be aware of how PyTorch's compilation tools (Dynamo, AOTAutograd) handle dynamic behavior.



\# How to check if a PyTorch program is traceable



There are several ways to interrogate a PyTorch program to see if it adheres to a "traceable computation graph," especially in the context of tools like `torch.compile` (which `iree-turbine` leverages) and `torch.export`. The key concept here is identifying \*\*graph breaks\*\*.



A truly "traceable" computation graph means that the entire computation can be represented as a static sequence of operations, without needing to fall back to Python eager mode for certain parts.



Here's how you can check:



\## Using `torch.compile` for Graph Break Analysis



The most direct and modern way to evaluate traceability for IREE (and other PyTorch compilation backends) is through `torch.compile`.



1\.  \*\*`TORCH\_LOGS="graph\_breaks"` Environment Variable:\*\*

&nbsp;   This is often the first step. By setting this environment variable before running your PyTorch program, `torch.compile` will log detailed information about any graph breaks it encounters, including the reason for the break and the line of code that caused it.



&nbsp;   ```bash

&nbsp;   TORCH\_LOGS="graph\_breaks" python your\_script.py

&nbsp;   ```



&nbsp;   You'll see output like:

&nbsp;   `Graph break in user code at /path/to/your\_script.py:XX Reason: Unsupported: builtin: ...`



2\.  \*\*`torch.\_dynamo.explain()`:\*\*

&nbsp;   For a more programmatic way to get insights into graph breaks without changing environment variables, `torch.\_dynamo.explain()` is very useful. It provides a structured output of the compilation process, including reasons for graph breaks.



&nbsp;   ```python

&nbsp;   import torch

&nbsp;   import torch.\_dynamo as dynamo



&nbsp;   def my\_model\_forward(x):

&nbsp;       if x.sum() > 0: # This is a common source of graph breaks

&nbsp;           return x.sin()

&nbsp;       else:

&nbsp;           return x.cos()



&nbsp;   # Get explanation

&nbsp;   explanation = dynamo.explain(my\_model\_forward)(torch.randn(5))

&nbsp;   print(explanation.graph\_break\_reasons)

&nbsp;   ```



&nbsp;   This will explicitly tell you if and why a graph break occurred.



3\.  \*\*`fullgraph=True` in `torch.compile`:\*\*

&nbsp;   When you apply `torch.compile` to your model or function, you can set `fullgraph=True`. If `torch.compile` fails to produce a single, unbroken graph, it will raise an error. This is a strict check to ensure complete graph capture.



&nbsp;   ```python

&nbsp;   import torch



&nbsp;   class MyModel(torch.nn.Module):

&nbsp;       def forward(self, x):

&nbsp;           if x.mean() > 0: # Data-dependent control flow

&nbsp;               return x + 1

&nbsp;           else:

&nbsp;               return x \* 2



&nbsp;   model = MyModel()



&nbsp;   try:

&nbsp;       compiled\_model = torch.compile(model, fullgraph=True)

&nbsp;       # If execution reaches here, it means a full graph was compiled

&nbsp;       print("Model compiled to a full graph!")

&nbsp;       output = compiled\_model(torch.randn(10))

&nbsp;   except torch.\_dynamo.exc.BackendCompilerFailed as e:

&nbsp;       print(f"Failed to compile to a full graph: {e}")

&nbsp;       # You can inspect e for more details, though logs are often more verbose

&nbsp;   ```



4\.  \*\*Inspecting the `ExportedProgram` (for `torch.export`):\*\*

&nbsp;   If you're directly using `torch.export` (which is what IREE ultimately consumes), the `ExportedProgram` object itself offers some insights. A key limitation of `torch.export` (compared to `torch.compile`'s fallback mechanism) is that it \*does not support graph breaks\*. If your code has graph breaks, `torch.export` will typically raise an error directly.



&nbsp;   You can try to export and see if it succeeds:



&nbsp;   ```python

&nbsp;   import torch

&nbsp;   from torch.export import export, ExportedProgram, Dim



&nbsp;   class MyExportableModel(torch.nn.Module):

&nbsp;       def forward(self, x, y):

&nbsp;           return torch.matmul(x, y)



&nbsp;   # Example with dynamic shapes

&nbsp;   model = MyExportableModel()

&nbsp;   example\_args = (torch.randn(10, 5), torch.randn(5, 20))

&nbsp;   dynamic\_shapes = {"x": {0: Dim("batch\_size\_x")}, "y": {1: Dim("output\_dim\_y")}}



&nbsp;   try:

&nbsp;       ep = export(model, example\_args, dynamic\_shapes=dynamic\_shapes)

&nbsp;       print("Model successfully exported as a single graph!")

&nbsp;       # You can inspect ep.graph for the FX graph

&nbsp;       # print(ep.graph)

&nbsp;   except Exception as e:

&nbsp;       print(f"Model export failed: {e}")

&nbsp;       print("This usually indicates untraceable code or unsupported operations.")



&nbsp;   # Example with data-dependent control flow that would typically break export

&nbsp;   class MyProblematicModel(torch.nn.Module):

&nbsp;       def forward(self, x):

&nbsp;           if x.sum() > 0:

&nbsp;               return x + 1

&nbsp;           else:

&nbsp;               return x \* 2



&nbsp;   model\_problematic = MyProblematicModel()

&nbsp;   try:

&nbsp;       ep\_problematic = export(model\_problematic, (torch.randn(5),))

&nbsp;   except Exception as e:

&nbsp;       print(f"\\nProblematic model export failed (as expected): {e}")

&nbsp;       print("This is because torch.export does not handle general Python data-dependent control flow.")



&nbsp;   # How to fix problematic model for export using torch.cond

&nbsp;   class MyFixedModel(torch.nn.Module):

&nbsp;       def forward(self, x):

&nbsp;           return torch.cond(x.sum() > 0, lambda: x + 1, lambda: x \* 2)



&nbsp;   model\_fixed = MyFixedModel()

&nbsp;   try:

&nbsp;       ep\_fixed = export(model\_fixed, (torch.randn(5),))

&nbsp;       print("\\nFixed model successfully exported with torch.cond!")

&nbsp;   except Exception as e:

&nbsp;       print(f"Fixed model export failed unexpectedly: {e}")

&nbsp;   ```



\## What causes "non-traceable" behavior (graph breaks)?



The main culprits for graph breaks are:



&nbsp; \* \*\*Data-dependent control flow:\*\* `if`/`else` statements, `for`/`while` loops where the condition depends on the \*values\* of tensors.

&nbsp;     \* \*\*Solution:\*\* Use `torch.cond`, `torch.while\_loop`, or refactor to avoid value-dependent branching in Python.

&nbsp; \* \*\*Operations on non-tensor Python objects:\*\* Using standard Python lists, dictionaries, or built-in functions on non-tensor data that then influences tensor operations.

&nbsp; \* \*\*Unsupported PyTorch operations:\*\* Some rare or very new PyTorch ops might not have tracing support yet.

&nbsp; \* \*\*Operations that implicitly change graph structure:\*\* Certain operations might not be easily represented in a static graph.

&nbsp; \* \*\*`item()` calls:\*\* Extracting scalar values from tensors can cause a graph break because the compiler needs to materialize the tensor and then convert it to a Python scalar. You can configure `torch.\_dynamo.config.capture\_scalar\_outputs = True` to mitigate this in some cases.

&nbsp; \* \*\*Modifying model `nn.Module` attributes directly:\*\* Modifying `self.param` or `self.buffer` in ways that aren't clear to the tracer can cause issues.



By using the methods above, especially `TORCH\_LOGS="graph\_breaks"` and `torch.\_dynamo.explain()`, you can pinpoint exactly where your PyTorch program deviates from a fully traceable graph and then work to refactor those sections using PyTorch's graph-friendly constructs. This will improve your success rate with IREE.







\# Examples of graph breaks and their refactorizations to resolve the graph breaks



Understanding and fixing graph breaks is crucial for getting good performance with compilers like IREE. Here are two common examples of PyTorch programs that cause graph breaks, and how to refactor them using proper PyTorch control flow primitives (`torch.cond` and `torch.while\_loop`).



\*\*Key Concept: Graph Breaks\*\*



A "graph break" occurs when `torch.compile` (or `torch.export`) encounters Python control flow or operations on non-tensor data that it cannot convert into a static, "tracable" computation graph. When this happens, PyTorch falls back to eager execution for that segment, hurting performance and preventing full compilation by backends like IREE.



-----



\### Example 1: Data-Dependent Conditional (`if/else`)



This is perhaps the most common source of graph breaks. When a Python `if`/`else` statement's condition depends on the \*value\* of a tensor (not just its shape or existence), `torch.compile` will often break.



\*\*Problematic Code:\*\*



```python

import torch



class DynamicActivation(torch.nn.Module):

&nbsp;   def forward(self, x):

&nbsp;       # The condition (x.mean() > 0) depends on the \*value\* of x

&nbsp;       if x.mean() > 0:

&nbsp;           return torch.relu(x)

&nbsp;       else:

&nbsp;           return torch.sigmoid(x)



\# Test with torch.compile and fullgraph=True to see the error

model\_problematic = DynamicActivation()



\# You'll likely see a GraphBreak error if you try to compile this with fullgraph=True

\# Or, if not using fullgraph=True, you'll see a log message about the graph break.

\# Example: TORCH\_LOGS="graph\_breaks" python your\_script.py

\# If compiled without fullgraph=True, it will silently fall back to eager for the if/else.

try:

&nbsp;   compiled\_model = torch.compile(model\_problematic, fullgraph=True)

&nbsp;   \_ = compiled\_model(torch.randn(5))

except Exception as e:

&nbsp;   print(f"Problematic model compile failed (as expected): {e}")

```



\*\*Why it breaks:\*\* The Python interpreter needs to evaluate `x.mean() > 0` at runtime to decide which branch to take. This decision point cannot be statically compiled into the graph by default.



\*\*Refactored Code (using `torch.cond`):\*\*



`torch.cond` (conditional) is a functional primitive that allows you to express data-dependent `if`/`else` logic directly within the computation graph.



```python

import torch



class DynamicActivationRefactored(torch.nn.Module):

&nbsp;   def forward(self, x):

&nbsp;       # Use torch.cond for data-dependent conditional logic

&nbsp;       # The first argument is the predicate (must be a boolean tensor)

&nbsp;       # The second argument is a callable (lambda) for the 'true' branch

&nbsp;       # The third argument is a callable (lambda) for the 'false' branch

&nbsp;       return torch.cond(x.mean() > 0,

&nbsp;                         lambda: torch.relu(x),

&nbsp;                         lambda: torch.sigmoid(x))



\# Test with torch.compile and fullgraph=True

model\_refactored = DynamicActivationRefactored()



try:

&nbsp;   compiled\_model\_refactored = torch.compile(model\_refactored, fullgraph=True)

&nbsp;   output\_true\_branch = compiled\_model\_refactored(torch.tensor(\[1.0, 2.0, 3.0])) # Mean > 0

&nbsp;   output\_false\_branch = compiled\_model\_refactored(torch.tensor(\[-1.0, -2.0, -3.0])) # Mean < 0

&nbsp;   print("Refactored model compiled and ran successfully!")

&nbsp;   print(f"Output (true branch): {output\_true\_branch}")

&nbsp;   print(f"Output (false branch): {output\_false\_branch}")



except Exception as e:

&nbsp;   print(f"Refactored model compile failed unexpectedly: {e}")



\# Verification of behavior

expected\_true = torch.relu(torch.tensor(\[1.0, 2.0, 3.0]))

expected\_false = torch.sigmoid(torch.tensor(\[-1.0, -2.0, -3.0]))

assert torch.allclose(output\_true\_branch, expected\_true)

assert torch.allclose(output\_false\_branch, expected\_false)

print("Outputs match expected behavior.")

```



-----



\### Example 2: Data-Dependent Loop (`while`)



Similar to conditionals, Python `while` loops whose termination condition or iteration count depends on tensor values will cause graph breaks.



\*\*Problematic Code:\*\*



```python

import torch



class NewtonRaphsonSolver(torch.nn.Module):

&nbsp;   def \_\_init\_\_(self, tol=1e-6, max\_iter=100):

&nbsp;       super().\_\_init\_\_()

&nbsp;       self.tol = tol

&nbsp;       self.max\_iter = max\_iter



&nbsp;   def forward(self, x\_init):

&nbsp;       # This is a simplified example for illustration

&nbsp;       # Imagine a real Newton-Raphson where iteration depends on a condition

&nbsp;       x = x\_init

&nbsp;       diff = torch.tensor(1.0) # Placeholder for difference

&nbsp;       iterations = 0



&nbsp;       # Loop condition depends on 'diff' (a tensor value)

&nbsp;       while diff.abs().max() > self.tol and iterations < self.max\_iter:

&nbsp;           # Simulate a step, e.g., x\_new = x - f(x)/f'(x)

&nbsp;           # For simplicity, let's just make 'x' change and 'diff' decrease

&nbsp;           x\_new = x - 0.1 \* x # Dummy update

&nbsp;           diff = x\_new - x

&nbsp;           x = x\_new

&nbsp;           iterations += 1

&nbsp;           # print(f"  Iteration {iterations}: x={x.item():.4f}, diff={diff.item():.4f}")



&nbsp;       return x



\# Test with torch.compile and fullgraph=True

model\_problematic\_loop = NewtonRaphsonSolver()



try:

&nbsp;   compiled\_model\_loop = torch.compile(model\_problematic\_loop, fullgraph=True)

&nbsp;   \_ = compiled\_model\_loop(torch.tensor(\[5.0]))

except Exception as e:

&nbsp;   print(f"\\nProblematic loop model compile failed (as expected): {e}")

```



\*\*Why it breaks:\*\* The `while` loop's condition `diff.abs().max() > self.tol` depends on a tensor's value. The Python interpreter needs to re-evaluate this condition in each iteration, which prevents it from being compiled into a static graph.



\*\*Refactored Code (using `torch.while\_loop`):\*\*



`torch.while\_loop` provides a graph-friendly way to express loops. It takes a condition callable and a body callable.



```python

import torch



class NewtonRaphsonSolverRefactored(torch.nn.Module):

&nbsp;   def \_\_init\_\_(self, tol=1e-6, max\_iter=100):

&nbsp;       super().\_\_init\_\_()

&nbsp;       self.tol = tol

&nbsp;       self.max\_iter = max\_iter



&nbsp;   def forward(self, x\_init):

&nbsp;       # Initial state for the loop: (current\_x, current\_diff, iterations)

&nbsp;       initial\_loop\_vars = (x\_init, torch.tensor(1.0), torch.tensor(0))



&nbsp;       def loop\_condition(current\_x, current\_diff, iterations):

&nbsp;           # The condition must return a boolean tensor

&nbsp;           return (current\_diff.abs().max() > self.tol) \& (iterations < self.max\_iter)



&nbsp;       def loop\_body(current\_x, current\_diff, iterations):

&nbsp;           # This function returns the next state of the loop variables

&nbsp;           x\_new = current\_x - 0.1 \* current\_x # Dummy update

&nbsp;           diff\_new = x\_new - current\_x

&nbsp;           iterations\_new = iterations + 1

&nbsp;           return (x\_new, diff\_new, iterations\_new)



&nbsp;       # Call torch.while\_loop

&nbsp;       # It takes the condition, body, and initial state of loop variables

&nbsp;       final\_x, final\_diff, final\_iterations = torch.while\_loop(

&nbsp;           loop\_condition,

&nbsp;           loop\_body,

&nbsp;           initial\_loop\_vars

&nbsp;       )

&nbsp;       return final\_x



\# Test with torch.compile and fullgraph=True

model\_refactored\_loop = NewtonRaphsonSolverRefactored()



try:

&nbsp;   compiled\_model\_refactored\_loop = torch.compile(model\_refactored\_loop, fullgraph=True)

&nbsp;   output = compiled\_model\_refactored\_loop(torch.tensor(\[5.0]))

&nbsp;   print("\\nRefactored loop model compiled and ran successfully!")

&nbsp;   print(f"Final output: {output}")

except Exception as e:

&nbsp;   print(f"Refactored loop model compile failed unexpectedly: {e}")



\# Manual verification (for comparison)

x\_val = torch.tensor(\[5.0])

diff\_val = torch.tensor(1.0)

iter\_val = 0

tol = 1e-6

max\_iter = 100



while diff\_val.abs().max() > tol and iter\_val < max\_iter:

&nbsp;   x\_new\_val = x\_val - 0.1 \* x\_val

&nbsp;   diff\_val = x\_new\_val - x\_val

&nbsp;   x\_val = x\_new\_val

&nbsp;   iter\_val += 1



print(f"Manual computation result: {x\_val}")

assert torch.allclose(output, x\_val)

print("Outputs match expected behavior.")

```



-----



\### General Refactoring Principles:



1\.  \*\*Prefer `torch.cond` for `if`/`else`:\*\* If your conditional logic depends on tensor values, always use `torch.cond`.

2\.  \*\*Prefer `torch.while\_loop` for `while` loops:\*\* If your loop termination or iteration count depends on tensor values, use `torch.while\_loop`.

3\.  \*\*Avoid `item()` and Python scalar conversions:\*\* Operations like `tensor.item()` or implicitly converting a 1-element tensor to a Python scalar can cause graph breaks. If you need a scalar, try to keep it as a 0-D tensor. If you absolutely need a Python scalar for external logic, ensure it's outside the compiled region or acknowledge the graph break.

4\.  \*\*Keep tensor operations pure:\*\* Try to avoid mixing Python list/dict manipulations with tensor operations that are intended to be compiled.

5\.  \*\*Use `torch.\_dynamo.explain()` and `TORCH\_LOGS="graph\_breaks"`:\*\* These are invaluable tools for identifying \*why\* graph breaks are happening in your actual code.



By following these patterns, you can significantly increase the "traceability" of your PyTorch programs, making them much more amenable to compilation with IREE and other backends.

