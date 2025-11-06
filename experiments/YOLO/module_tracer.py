"""
Recording Module Hierarchy With a Custom Tracer

In this example, we are going to define a custom `fx.Tracer` instance that--
for each recorded operation--also notes down the qualified name of the module
from which that operation originated. The _qualified name_ is the path to the
Module from the root module. More information about this concept can be
found in the documentation for `Module.get_submodule`:
https://github.com/pytorch/pytorch/blob/9f2aea7b88f69fc74ad90b1418663802f80c1863/torch/nn/modules/module.py#L385
"""
import torch
import torch.fx
from typing import Any, Callable, Dict, Optional, Tuple

class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module : Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).

        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy


# Testing: Ultralytics YOLO model
from ultralytics import YOLO

# Load a pretrained YOLO11n model
yolo = YOLO("yolo11n.pt")

# Instantiate our ModulePathTracer and use that to trace our ResNet18
tracer = ModulePathTracer()
traced_yolo = tracer.trace(yolo.model)

# Print (node, module qualified name) for every node in the Graph
for node in traced_yolo.nodes:
    module_qualname = tracer.node_to_originating_module.get(node)
    print('Node', node, 'is from module', module_qualname)

"""
That TraceError: Proxy object cannot be iterated error when tracing 
a YOLO model with PyTorch FX is very common. It typically happens 
because the model contains unsupported control flow in its forward 
method, such as a loop that iterates over a list of modules or tensors 
(like in the neck or head of a YOLO model) or unpacking (*args, **kwargs) 
inside the graph.

Unfortunately, directly tracing Ultralytics YOLO models (like YOLOv5, 
YOLOv8) using the standard torch.fx.symbolic_trace or a simple 
custom tracer often fails because their forward pass includes logic 
that FX's symbolic tracer can't handle.

How to Trace a YOLO Model with FX:
The most reliable approach is to use an FX-compatible wrapper or a more 
advanced tracing tool like TorchDynamo (part of the torch.compile stack) 
or to modify the model's forward method to remove the unsupported 
control flow.

See dynamo_tracer.py for an example of using TorchDynamo to trace
a YOLO model. It is not working either, but the explanation at the bottom
of that file describes why.
"""