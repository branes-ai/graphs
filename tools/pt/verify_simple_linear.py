import torch

# Test loading the file
try:
    model = torch.jit.load('simple_linear.pt')
    print("File loads successfully in Python")
    print(f"Model: {model}")
    
    # Test with some input
    test_input = torch.randn(1, 10)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    
except Exception as e:
    print(f"Error loading in Python: {e}")