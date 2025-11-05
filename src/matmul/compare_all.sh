#!/bin/bash
# Compare C++ implementations with NumPy/BLAS

echo "================================================================================"
echo "Matrix Multiplication Performance Comparison"
echo "================================================================================"
echo ""

echo "Running C++ V1 (original implementation)..."
echo "--------------------------------------------------------------------------------"
./build/matmul_benchmark
echo ""

echo "Running C++ V2 (optimized implementation)..."
echo "--------------------------------------------------------------------------------"
./build/matmul_benchmark_v2
echo ""

echo "Running NumPy/BLAS (reference)..."
echo "--------------------------------------------------------------------------------"
python3 benchmark_numpy.py
echo ""

echo "================================================================================"
echo "Comparison Summary"
echo "================================================================================"
echo ""
echo "Extract 4096×4096 results and compare:"
echo "  V1 (original):  ~175-192 GFLOPS  (27-30% efficiency)"
echo "  V2 (optimized): ~238 GFLOPS      (37% efficiency)"
echo "  NumPy/BLAS:     ~800+ GFLOPS     (125%+ efficiency)"
echo ""
echo "Key Takeaways:"
echo "  - V2 is 36% faster than V1"
echo "  - NumPy/BLAS is 3.5× faster than our best (using all cores + HT)"
echo "  - To reach 80% efficiency (~512 GFLOPS) requires:"
echo "      * Eliminate packing overhead"
echo "      * Optimize micro-kernel (8×6 blocking)"
echo "      * Non-temporal stores"
echo "      * Better prefetching"
echo "      * Assembly micro-kernel"
echo ""
