#include "tiled_matmul_v2.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace matmul;

struct BenchmarkResult {
    size_t M, N, K;
    double time_ms;
    double gflops;
    float max_error;
};

BenchmarkResult benchmark_matmul(size_t M, size_t N, size_t K, bool validate = false) {
    auto A = TiledMatMul::create_random_matrix(M, K);
    auto B = TiledMatMul::create_random_matrix(K, N);
    auto C = TiledMatMul::create_matrix(M, N);

    // Warm-up
    TiledMatMul::multiply(A, B, C, M, N, K);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    TiledMatMul::multiply(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (time_ms / 1000.0);

    float max_error = 0.0f;
    if (validate) {
        max_error = TiledMatMul::compute_error(A, B, C, M, N, K);
    }

    return {M, N, K, time_ms, gflops, max_error};
}

int main(int argc, char** argv) {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "OPTIMIZED Matrix Multiplication Benchmark (V2)\n";
    std::cout << "Target: Intel Core i7-12700K\n";
    std::cout << "Optimizations:\n";
    std::cout << "  - 6×16 register blocking (12 YMM registers)\n";
    std::cout << "  - B-matrix packing (column-major panels)\n";
    std::cout << "  - A-matrix packing (row-major panels)\n";
    std::cout << "  - Adaptive parallelism (sequential <512, parallel >=512)\n";
    std::cout << "  - Prefetching hints\n";
    std::cout << std::string(80, '=') << "\n\n";

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
            std::cout << "  -v, --validate    Validate correctness\n";
            std::cout << "  -s, --size N      Run single benchmark\n";
            std::cout << "  -h, --help        Show help\n";
            return 0;
        }
    }

    if (custom_size > 0) {
        std::cout << "Running " << custom_size << "×" << custom_size << "...\n\n";
        auto result = benchmark_matmul(custom_size, custom_size, custom_size, validate);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Time:       " << result.time_ms << " ms\n";
        std::cout << "Throughput: " << result.gflops << " GFLOPS\n";

        double theoretical_peak = 640.0;
        double efficiency = (result.gflops / theoretical_peak) * 100.0;
        std::cout << "Efficiency: " << efficiency << "%\n";

        if (validate) {
            std::cout << "Max Error:  " << std::scientific << result.max_error << "\n";
        }
    } else {
        std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};
        std::vector<BenchmarkResult> results;

        std::cout << "Running benchmark suite...\n\n";

        for (size_t size : sizes) {
            std::cout << "Testing " << size << "×" << size << "... " << std::flush;
            auto result = benchmark_matmul(size, size, size, validate);
            results.push_back(result);
            std::cout << result.gflops << " GFLOPS\n";
        }

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "Summary\n";
        std::cout << std::string(80, '=') << "\n\n";

        std::cout << std::setw(10) << "Size" << " | "
                  << std::setw(12) << "Time (ms)" << " | "
                  << std::setw(12) << "GFLOPS" << " | "
                  << std::setw(12) << "Efficiency" << " | "
                  << std::setw(10) << "Error" << "\n";
        std::cout << std::string(75, '-') << "\n";

        double theoretical_peak = 640.0;
        for (const auto& r : results) {
            double efficiency = (r.gflops / theoretical_peak) * 100.0;
            std::cout << std::setw(10) << r.M << " | "
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.time_ms << " | "
                      << std::setw(12) << r.gflops << " | "
                      << std::setw(11) << efficiency << "% | ";

            if (validate) {
                std::cout << std::setw(10) << std::scientific << std::setprecision(2) << r.max_error;
            } else {
                std::cout << std::setw(10) << "N/A";
            }
            std::cout << "\n";
        }

        auto best = *std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.gflops < b.gflops; });

        double best_efficiency = (best.gflops / theoretical_peak) * 100.0;

        std::cout << "\n";
        std::cout << "Peak Performance: " << std::fixed << std::setprecision(1)
                  << best.gflops << " GFLOPS (" << best.M << "×" << best.M << ")\n";
        std::cout << "Peak Efficiency:  " << best_efficiency << "%\n";

        if (best_efficiency > 80) {
            std::cout << "Status: ✓ EXCELLENT - Near peak performance!\n";
        } else if (best_efficiency > 60) {
            std::cout << "Status: ✓ GOOD - Well optimized\n";
        } else if (best_efficiency > 40) {
            std::cout << "Status: ⚠ FAIR - Room for improvement\n";
        } else {
            std::cout << "Status: ✗ POOR - Likely memory-bound\n";
        }
    }

    return 0;
}
