#include "tiled_matmul.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace matmul;

struct BenchmarkResult {
    size_t M, N, K;
    double time_ms;
    double gflops;
    float max_error;
};

BenchmarkResult benchmark_matmul(size_t M, size_t N, size_t K, bool validate = false) {
    // Create matrices
    auto A = TiledMatMul::create_random_matrix(M, K);
    auto B = TiledMatMul::create_random_matrix(K, N);
    auto C = TiledMatMul::create_matrix(M, N);

    // Warm-up run
    TiledMatMul::multiply(A, B, C, M, N, K);

    // Benchmark run
    auto start = std::chrono::high_resolution_clock::now();
    TiledMatMul::multiply(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    // Compute metrics
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double flops = 2.0 * M * N * K;  // Each element: K multiplies + K adds
    double gflops = (flops / 1e9) / (time_ms / 1000.0);

    // Validation (optional)
    float max_error = 0.0f;
    if (validate) {
        max_error = TiledMatMul::compute_error(A, B, C, M, N, K);
    }

    return {M, N, K, time_ms, gflops, max_error};
}

void print_header() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Multi-Level Tiled Matrix Multiplication Benchmark\n";
    std::cout << "Target: Intel Core i7-12700K (12 cores @ 5.0 GHz)\n";
    std::cout << "Optimizations: L1/L2/L3 blocking + AVX2 + C++17 parallel algorithms\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void print_result(const BenchmarkResult& result) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Matrix: " << result.M << "x" << result.N << " = "
              << result.M << "x" << result.K << " * " << result.K << "x" << result.N << "\n";
    std::cout << "  Time:       " << std::setw(8) << result.time_ms << " ms\n";
    std::cout << "  Throughput: " << std::setw(8) << result.gflops << " GFLOPS\n";

    if (result.max_error > 0.0f) {
        std::cout << "  Max Error:  " << std::scientific << result.max_error << std::fixed << "\n";
    }

    std::cout << "\n";
}

void print_cache_analysis(const BenchmarkResult& result) {
    size_t total_memory = result.M * result.K + result.K * result.N + result.M * result.N;
    size_t memory_bytes = total_memory * sizeof(float);
    double memory_mb = memory_bytes / (1024.0 * 1024.0);

    std::cout << "Cache Analysis:\n";
    std::cout << "  Total memory: " << memory_mb << " MB\n";

    if (memory_bytes <= 48 * 1024) {
        std::cout << "  ✓ Fits in L1 cache (48 KB)\n";
    } else if (memory_bytes <= 1280 * 1024) {
        std::cout << "  ✓ Fits in L2 cache (1.25 MB)\n";
    } else if (memory_bytes <= 25 * 1024 * 1024) {
        std::cout << "  ✓ Fits in L3 cache (25 MB)\n";
    } else {
        std::cout << "  ⚠ Exceeds L3 cache - using DRAM\n";
    }

    std::cout << "  Tile sizes: L1=" << L1_TILE << "x" << L1_TILE
              << ", L2=" << L2_TILE << "x" << L2_TILE
              << ", L3=" << L3_TILE << "x" << L3_TILE << "\n\n";
}

void print_performance_analysis(double gflops) {
    // i7-12700K theoretical peak (P-cores):
    // 8 P-cores * 2 FMA units * 8 floats (AVX2) * 5.0 GHz = 640 GFLOPS
    // With hyperthreading and E-cores, peak can reach ~800-1000 GFLOPS
    double theoretical_peak = 640.0;  // Conservative estimate for P-cores
    double efficiency = (gflops / theoretical_peak) * 100.0;

    std::cout << "Performance Analysis:\n";
    std::cout << "  Achieved:      " << std::fixed << std::setprecision(1)
              << gflops << " GFLOPS\n";
    std::cout << "  Theoretical:   " << theoretical_peak << " GFLOPS (P-cores only)\n";
    std::cout << "  Efficiency:    " << efficiency << "%\n";

    if (efficiency > 80) {
        std::cout << "  Status: ✓ Excellent - near peak performance\n";
    } else if (efficiency > 60) {
        std::cout << "  Status: ✓ Good - well optimized\n";
    } else if (efficiency > 40) {
        std::cout << "  Status: ⚠ Fair - room for improvement\n";
    } else {
        std::cout << "  Status: ✗ Poor - likely memory-bound\n";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    print_header();

    // Parse command-line arguments
    bool validate = false;
    bool run_all = true;
    size_t custom_size = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--validate" || arg == "-v") {
            validate = true;
        } else if (arg == "--size" || arg == "-s") {
            if (i + 1 < argc) {
                custom_size = std::stoul(argv[++i]);
                run_all = false;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n";
            std::cout << "Options:\n";
            std::cout << "  -v, --validate    Validate against naive implementation\n";
            std::cout << "  -s, --size N      Run single benchmark with NxN matrices\n";
            std::cout << "  -h, --help        Show this help message\n";
            return 0;
        }
    }

    if (custom_size > 0) {
        // Single custom benchmark
        std::cout << "Running benchmark with " << custom_size << "x" << custom_size << " matrices...\n\n";
        auto result = benchmark_matmul(custom_size, custom_size, custom_size, validate);
        print_result(result);
        print_cache_analysis(result);
        print_performance_analysis(result.gflops);
    } else {
        // Standard benchmark suite
        std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};
        std::vector<BenchmarkResult> results;

        std::cout << "Running benchmark suite...\n\n";

        for (size_t size : sizes) {
            std::cout << "Testing " << size << "x" << size << "...\n";
            auto result = benchmark_matmul(size, size, size, validate);
            results.push_back(result);
            print_result(result);
        }

        // Summary
        std::cout << std::string(80, '=') << "\n";
        std::cout << "Summary\n";
        std::cout << std::string(80, '=') << "\n\n";

        std::cout << std::setw(10) << "Size" << " | "
                  << std::setw(12) << "Time (ms)" << " | "
                  << std::setw(12) << "GFLOPS" << " | "
                  << std::setw(10) << "Error" << "\n";
        std::cout << std::string(60, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::setw(10) << r.M << " | "
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.time_ms << " | "
                      << std::setw(12) << r.gflops << " | ";

            if (validate) {
                std::cout << std::setw(10) << std::scientific << std::setprecision(2) << r.max_error;
            } else {
                std::cout << std::setw(10) << "N/A";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // Best performance
        auto best = *std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.gflops < b.gflops; });

        std::cout << "Peak Performance: " << best.gflops << " GFLOPS "
                  << "(" << best.M << "x" << best.M << " matrices)\n\n";

        print_performance_analysis(best.gflops);
    }

    return 0;
}
