import torch

try:
    # Test loading the file we just created
    model = torch.jit.load('simple_linear.pt')
    print("✓ File loads successfully in Python")
    
    # Test inference
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        output = model(test_input)
    print(f"✓ Inference works, output shape: {output.shape}")
    print(f"✓ Output: {output}")
    
except Exception as e:
    print(f"✗ Error loading in Python: {e}")