# How to find the input tensor shape of a model

Crucial problem: **how to determine the correct `example_input` for your PyTorch `.bin` file.** 
This is often the most challenging part of converting pre-trained models to MLIR or other formats.

Here's a breakdown of the problem and how to approach it:

**The Challenge**

* **No Metadata:** PyTorch `.bin` files (TorchScript models) store the model's architecture and weights, but they typically don't explicitly store the exact shape and data type of the expected input.
* **Need for Inference:** The `torch-mlir` compiler needs an example input to trace the model's execution and determine the shapes and data types of intermediate tensors.

**How to Find the Correct `example_input`**

1.  **Original Model Code (Best Approach):**
    * If you have the original Python code that defined and trained the model, this is the ideal scenario.
    * Look for the input layer of the model. The shape and data type of the input tensor in the Python code will be your `example_input`.
    * Example:
        ```python
        # Original model code (example)
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 10)  # Input size is 20

            def forward(self, x):
                return self.linear(x)

        # Then you know the input must be of size 20.
        example_input = torch.rand(1, 20) #batch size of 1, input size of 20.
        ```

2.  **Documentation or Model Description:**
    * If the model came from a library or repository, check the documentation or any accompanying information.
    * Often, the expected input shape and data type will be specified.

3.  **Trial and Error (with Caution):**
    * If you don't have the original code or documentation, you might have to resort to trial and error.
    * Start with reasonable guesses for the input shape and data type.
    * Use `torch.rand()` or `torch.randn()` to create tensors with the guessed shape.
    * Run your conversion script, and look for error messages.
    * Error messages from `torch-mlir` or IREE might give you clues about shape mismatches or data type errors.
    * **Important:** Trial and error can be time-consuming and might not always lead to the correct `example_input`.

4.  **Inspect the Model (Advanced):**
    * You can use `torch.jit.load()` to load the model and then inspect its `graph` attribute.
    * The graph will contain information about the model's operations, and you might be able to infer the input shape from the first layer.
    * This method requires a deeper understanding of TorchScript graphs.
    * Example:
        ```python
        model = torch.jit.load("your_model.bin")
        graph = model.graph
        print(graph) #this will print the graph, which might give you clues.
        ```

**Adapting the `example_input` in Your Script**

* Once you've determined the correct `example_input`, replace the placeholder `torch.rand(10)` in your script with the actual input tensor.
* Example:
    ```python
    if __name__ == "__main__":
        # ... (rest of the code) ...
        # Replace with the correct example_input
        example_input = torch.randn(1, 28, 28)  # Example: For a model expecting 28x28 images
        convert_bin_to_mlir(input_pt_bin_file, example_input, output_mlir_file)
    ```

**Important Considerations**

* **Data Type:** Make sure the data type of your `example_input` matches the model's expected input type (e.g., `torch.float32`, `torch.int64`).
* **Batch Dimension:** In many cases, models are designed to handle batches of inputs. Include a batch dimension in your `example_input` (e.g., `torch.rand(batch_size, input_size)`).
* **Normalization:** If your model expects normalized input data, make sure your `example_input` is also normalized.
* **Input order:** If your model expects multiple inputs, make sure that the example input is a tuple of tensors, and that the order of those tensors is correct.

By carefully considering these points, you'll be able to determine the correct `example_input` and successfully convert your PyTorch `.bin` file to MLIR.

