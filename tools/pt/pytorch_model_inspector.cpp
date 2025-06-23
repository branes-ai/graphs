#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>

class ModelInspector {
private:
    torch::jit::script::Module model;
    bool model_loaded;

public:
    ModelInspector() : model_loaded(false) {}

    bool loadModel(const std::string& model_path) {
        try {
            model = torch::jit::load(model_path);
            model.eval();  // Set to evaluation mode
            model_loaded = true;
            std::cout << "Model loaded successfully from: " << model_path << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            model_loaded = false;
            return false;
        }
    }

    void printModelMethods() {
        if (!model_loaded) {
            std::cerr << "No model loaded!" << std::endl;
            return;
        }

        std::cout << "\n=== Available Methods ===" << std::endl;
        try {
            // Try different ways to get method information
            if (model.find_method("forward")) {
                std::cout << "Method: forward (available)" << std::endl;
            }

            // List other common methods
            std::vector<std::string> common_methods = { "forward", "__call__", "parameters", "named_parameters" };
            for (const auto& method_name : common_methods) {
                if (model.find_method(method_name)) {
                    std::cout << "Method: " << method_name << " (available)" << std::endl;
                }
            }

        }
        catch (const std::exception& e) {
            std::cout << "Could not enumerate methods: " << e.what() << std::endl;
        }
    }

    void inspectModelInterface() {
        if (!model_loaded) {
            std::cerr << "No model loaded!" << std::endl;
            return;
        }

        try {
            std::cout << "\n=== Model Interface Information ===" << std::endl;

            // Try to get the forward method
            if (model.find_method("forward")) {
                std::cout << "Model has 'forward' method" << std::endl;

                // Try to get method schema using the graph
                try {
                    auto graph = model.get_method("forward").graph();
                    std::cout << "Method graph inputs:" << std::endl;

                    auto inputs = graph->inputs();
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        auto input = inputs[i];
                        std::cout << "  [" << i << "] " << input->debugName()
                            << ": " << input->type()->str() << std::endl;
                    }

                    std::cout << "\nMethod graph outputs:" << std::endl;
                    auto outputs = graph->outputs();
                    for (size_t i = 0; i < outputs.size(); ++i) {
                        auto output = outputs[i];
                        std::cout << "  [" << i << "] " << output->debugName()
                            << ": " << output->type()->str() << std::endl;
                    }
                }
                catch (const std::exception& e) {
                    std::cout << "Could not extract detailed schema: " << e.what() << std::endl;
                }
            }
            else {
                std::cout << "No 'forward' method found" << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "Error inspecting model interface: " << e.what() << std::endl;
        }
    }

    void probeWithDummyInput(const std::vector<int64_t>& input_shape) {
        if (!model_loaded) {
            std::cerr << "No model loaded!" << std::endl;
            return;
        }

        try {
            std::cout << "\n=== Probing with dummy input ===" << std::endl;
            std::cout << "Input shape: [";
            for (size_t i = 0; i < input_shape.size(); ++i) {
                std::cout << input_shape[i];
                if (i < input_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;

            // Create dummy input tensor
            auto input_tensor = torch::randn(input_shape);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // Forward pass
            auto output = model.forward(inputs);
            
            // Analyze output
            if (output.isTensor()) {
                auto out_tensor = output.toTensor();
                std::cout << "Output shape: [";
                for (int i = 0; i < out_tensor.dim(); ++i) {
                    std::cout << out_tensor.size(i);
                    if (i < out_tensor.dim() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                std::cout << "Output dtype: " << out_tensor.dtype() << std::endl;
            } else if (output.isTuple()) {
                auto tuple_output = output.toTuple();
                std::cout << "Output is tuple with " << tuple_output->elements().size() << " elements:" << std::endl;
                for (size_t i = 0; i < tuple_output->elements().size(); ++i) {
                    if (tuple_output->elements()[i].isTensor()) {
                        auto tensor = tuple_output->elements()[i].toTensor();
                        std::cout << "  Element " << i << " shape: [";
                        for (int j = 0; j < tensor.dim(); ++j) {
                            std::cout << tensor.size(j);
                            if (j < tensor.dim() - 1) std::cout << ", ";
                        }
                        std::cout << "], dtype: " << tensor.dtype() << std::endl;
                    }
                }
            } else {
                std::cout << "Output type: " << output.type()->str() << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error during dummy inference: " << e.what() << std::endl;
            std::cerr << "Try adjusting the input shape or check model requirements" << std::endl;
        }
    }

    void printModelGraph() {
        if (!model_loaded) {
            std::cerr << "No model loaded!" << std::endl;
            return;
        }

        try {
            std::cout << "\n=== Model Graph Structure ===" << std::endl;

            // Try to get the graph from the forward method
            if (model.find_method("forward")) {
                auto graph = model.get_method("forward").graph();
                std::cout << *graph << std::endl;
            }
            else {
                std::cout << "No forward method found to display graph" << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "Error printing model graph: " << e.what() << std::endl;
        }
    }
};

/*

# Run with just model inspection
./model_inspector model.pt

# Run with dummy input probing (batch_size=1, channels=3, height=224, width=224)
./model_inspector model.pt 1 3 224 224

*/
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.pt> [input_shape...]" << std::endl;
        std::cerr << "Example: " << argv[0] << " model.pt 1 3 224 224" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    ModelInspector inspector;

    // Load the model
    if (!inspector.loadModel(model_path)) {
        return 1;
    }

    // Print available methods
    inspector.printModelMethods();

    // Inspect the model interface
    inspector.inspectModelInterface();

    // If input shape is provided, probe with dummy input
    if (argc > 2) {
        std::vector<int64_t> input_shape;
        for (int i = 2; i < argc; ++i) {
            input_shape.push_back(std::stoll(argv[i]));
        }
        inspector.probeWithDummyInput(input_shape);
    } else {
        std::cout << "\nTo probe with dummy input, provide shape as arguments:" << std::endl;
        std::cout << "Example: " << argv[0] << " " << model_path << " 1 3 224 224" << std::endl;
    }

    // Optionally print model graph (can be verbose)
    // inspector.printModelGraph();

    return 0;
}