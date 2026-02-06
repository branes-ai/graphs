# Mixed-precision


## Running in BF16 precision

Question: When we run the bf16 precision, do we actually use bf16 in the arithmetic?

Answer: Yes, when you specify --precision bf16, the code does:

```python
  model = model.to(torch.bfloat16)
  input_tensor = input_tensor.to(torch.bfloat16)
```

This converts all weights and activations to BF16, so the actual arithmetic uses BF16 format. However, whether it's hardware-accelerated depends on the device:

  - Jetson Orin (Ampere GPU): Native BF16 Tensor Core support. You get real BF16 throughput.

  - Intel CPUs: Only Ice Lake server and later have AVX-512_BF16. The i7-12700K (Alder Lake) does NOT have this extension - BF16 will be emulated, making
  it slower than FP32.

  - AMD Ryzen: No native BF16 support in consumer chips. Will be emulated.

To verify if BF16 is actually accelerated on a device, you can check:

```python
  import torch
  # For CUDA
  print(torch.cuda.is_bf16_supported())  # True on Ampere+

  # For CPU - check if it's using native instructions
  # (no direct API, but you'd see similar or better perf than FP32 if native)
```

On the Jetson, BF16 calibration is meaningful. On the i7-12700K, it's measuring emulated BF16 performance, which isn't very useful for prediction.

## Real quantization

Question: int8 does not work that way, is that correct?

Answer: Correct. INT8 is fundamentally different:

  - FP16/BF16: Just change the data type - the model structure stays the same.
  model = model.half()  # Works directly

  - INT8: Requires quantization - you need scale factors and zero points for each tensor, plus quantized operator implementations.
  model = model.to(torch.int8)  # Does NOT work for inference

The current code just prints a warning and runs FP32:
```python
  elif args.precision == 'int8':
      print("  NOTE: INT8 calibration uses FP32 model; actual INT8 requires quantization")
```

For real INT8 calibration, you'd need to:
  1. Use PyTorch's quantization API (torch.quantization.quantize_dynamic or static quantization)
  2. Or use TensorRT/ONNX Runtime which handle quantization
  3. Run the quantized model through the timing interpreter

This is a more complex feature - the calibration would need to quantize each model first, which requires calibration data for static quantization.

What is required to add proper INT8 support?

Depends on which approach:

- Dynamic Quantization (2-3 hours)
  - Quick to add, CPU-only
  - Only quantizes Linear layers (not Conv2d)
  - Limited usefulness for CNNs

- Static Quantization (4-6 hours)
  - Full model quantization (Conv2d + Linear)
  - Requires running calibration data through the model to determine activation ranges
  - CPU-only in PyTorch

- TensorRT INT8 for GPU (1-2 days)
  - Needed for Jetson/CUDA INT8
  - PyTorch's native quantization doesn't support CUDA for most ops
  - Requires building TensorRT engines, different measurement approach
  - Significant complexity

The practical reality: if you want INT8 on Jetson, you need TensorRT, not PyTorch quantization. That's a bigger undertaking.

Given you're focused on BF16 for edge training, I'd suggest leaving INT8 as a known limitation for now and tackling it later if needed. 
The FP32/FP16/BF16 calibration covers the immediate use cases.

