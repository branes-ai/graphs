# torch.fx

## Overview

FX is a toolkit for developers to use to transform nn.Module instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation. 

The symbolic tracer performs “symbolic execution” of the Python code. It feeds fake values, called Proxies, through the code. Operations on theses Proxies are recorded. More information about symbolic tracing can be found in the symbolic_trace() and Tracer documentation.

The intermediate representation is the container for the operations that were recorded during symbolic tracing. It consists of a list of Nodes that represent function inputs, callsites (to functions, methods, or torch.nn.Module instances), and return values. More information about the IR can be found in the documentation for Graph. The IR is the format on which transformations are applied.

Python code generation is what makes FX a Python-to-Python (or Module-to-Module) transformation toolkit. For each Graph IR, we can create valid Python code matching the Graph’s semantics. This functionality is wrapped up in GraphModule, which is a torch.nn.Module instance that holds a Graph as well as a forward method generated from the Graph.

Taken together, this pipeline of components (symbolic tracing -> intermediate representation -> transforms -> Python code generation) constitutes the Python-to-Python transformation pipeline of FX. In addition, these components can be used separately. 

For example, symbolic tracing can be used in isolation to capture a form of the code for analysis (and not transformation) purposes. 

Code generation can be used for programmatically generating models, for example from a config file. 


## Writing Transformations

What is an FX transform? Essentially, it’s a function that looks like this.

```python
import torch
import torch.fx

def transform(m: nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    # Step 1: Acquire a Graph representing the code in `m`

    # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule. We'll
    # split that out in our transform to allow the caller to
    # customize tracing behavior.
    graph : torch.fx.Graph = tracer_class().trace(m)

    # Step 2: Modify this Graph or create a new one
    graph = ...

    # Step 3: Construct a Module to return
    return torch.fx.GraphModule(m, graph)
```

Your transform will take in a torch.nn.Module, acquire a Graph from it, do some modifications, and return a new torch.nn.Module. You should think of the torch.nn.Module that your FX transform returns as identical to a regular torch.nn.Module – you can pass it to another FX transform, you can pass it to TorchScript, or you can run it. Ensuring that the inputs and outputs of your FX transform are a torch.nn.Module will allow for composability.

Note

It is also possible to modify an existing GraphModule instead of creating a new one, like so:

```python
import torch
import torch.fx

def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # Modify gm.graph
    # <...>

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

    return gm
```

Note that you MUST call GraphModule.recompile() to bring the generated forward() method on the GraphModule in sync with the modified Graph.

Given that you’ve passed in a torch.nn.Module that has been traced into a Graph, there are now two primary approaches you can take to building a new Graph.

## A Quick Primer on Graphs

Full treatment of the semantics of graphs can be found in the Graph documentation, but we are going to cover the basics here. A Graph is a data structure that represents a method on a GraphModule. The information that this requires is:

 - What are the inputs to the method?
 - What are the operations that run inside the method?
 - What is the output (i.e. return) value from the method?

All three of these concepts are represented with Node instances. Let’s see what we mean by that with a short example:

```python
import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.topk(torch.sum(
            self.linear(x + self.linear.weight).relu(), dim=-1), 3)

m = MyModule()
gm = torch.fx.symbolic_trace(m)

gm.graph.print_tabular()
```

Here we define a module MyModule for demonstration purposes, instantiate it, symbolically trace it, then call the Graph.print_tabular() method to print out a table showing the nodes of this Graph:

opcode        | name          | target                                                      | args               | kwargs      |
------------- | ------------- | ----------------------------------------------------------- | ------------------ | ----------- |
placeholder   | x             | x                                                           | ()                 | {}          |
get_attr      | linear_weight | linear.weight                                               | ()                 | {}          |
call_function | add           | <built-in function add>                                     | (x, linear_weight) | {}          |
call_module   | linear        | linear                                                      | (add,)             | {}          |
call_method   | relu          | relu                                                        | (linear,)          | {}          |
call_function | sum_1         | <built-in method sum of type object at 0x00007FF8B8E35450>  | (relu,)            | {'dim': -1} |
call_function | topk          | <built-in method topk of type object at 0x00007FF8B8E35450> | (sum_1, 3)         | {}          |
output        | output        | output                                                      | (topk,)            | {}          |

We can use this information to answer the questions we posed above.

 - What are the inputs to the method? In FX, method inputs are specified via special placeholder nodes. In this case, we have a single placeholder node with a target of x, meaning we have a single (non-self) argument named x.

 - What are the operations within the method? The get_attr, call_function, call_module, and call_method nodes represent the operations in the method. A full treatment of the semantics of all of these can be found in the Node documentation.

 - What is the return value of the method? The return value in a Graph is specified by a special output node.

Given that we now know the basics of how code is represented in FX, we can now explore how we would edit a Graph.

## Graph Manipulation

### Direct Graph Manipulation

One approach to building this new Graph is to directly manipulate your old one. To aid in this, we can simply take the Graph we obtain from symbolic tracing and modify it. For example, let’s say we desire to replace torch.add() calls with torch.mul() calls.

```python
import torch
import torch.fx

# Sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)
```

We can also do more involved Graph rewrites, such as deleting or appending nodes. To aid in these transformations, FX has utility functions for transforming the graph that can be found in the Graph documentation. An example of using these APIs to append a torch.relu() call can be found below.

```python
# Specifies the insertion point. Any nodes added to the
# Graph within this scope will be inserted after `node`
with traced.graph.inserting_after(node):
    # Insert a new `call_function` node calling `torch.relu`
    new_node = traced.graph.call_function(
        torch.relu, args=(node,))

    # We want all places that used the value of `node` to
    # now use that value after the `relu` call we've added.
    # We use the `replace_all_uses_with` API to do this.
    node.replace_all_uses_with(new_node)
```