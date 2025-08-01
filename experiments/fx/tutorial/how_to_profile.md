To use the custom ProfilerTracer in your own PyTorch program to measure time spent in different compute nodes (operations), follow these steps:

1. Define your model as a torch.nn.Module
```python
class MyModel(torch.nn.Module):
    def forward(self, x):
        with torch.profiler.record_function('my_block'):
            y = torch.relu(x)
            return y * 2
```
2. Instantiate your model and input
```python
model = MyModel()
input_tensor = torch.randn(10, 10)
```
3. Use the custom ProfilerTracer to trace your model
```python
pt = ProfilerTracer()
graph_with_profiler = pt.trace(model)
traced_with_profiler = torch.fx.GraphModule(pt.root, graph_with_profiler)
```
4. Profile the traced model
```python
with torch.autograd.profiler.profile() as prof:
    traced_with_profiler(input_tensor)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```
## What happens?
The custom tracer injects special profiler ops into the FX graph, so named ranges (like 'my_block') appear in the profiler output.
The profiler output will show time spent in each named range and operation.
Summary:

Wrap code blocks in with torch.profiler.record_function('name').
Trace your model with ProfilerTracer.
Run the traced model inside a torch.autograd.profiler.profile() context.
Print the profiler results to see timing for each compute node and named range.