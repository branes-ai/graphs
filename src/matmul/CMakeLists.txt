cmake_minimum_required(VERSION 3.20)
project(TiledMatMul VERSION 1.0 LANGUAGES CXX)

# Require C++17 for std::execution parallel algorithms
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang optimizations
    add_compile_options(
        -O3                      # Maximum optimization
        -march=native            # Target current CPU (enables AVX2/FMA)
        -mtune=native            # Tune for current CPU
        -ffast-math              # Aggressive floating-point optimizations
        -funroll-loops           # Loop unrolling
        -fopenmp-simd            # SIMD optimizations
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # MSVC optimizations
    add_compile_options(
        /O2                      # Maximum optimization
        /arch:AVX2               # Enable AVX2
        /fp:fast                 # Fast floating-point
        /Qpar                    # Auto-parallelization hints
    )
endif()

# Try to find TBB (required for std::execution on GCC/Clang)
find_package(TBB QUIET)

if(TBB_FOUND)
    message(STATUS "Found TBB: ${TBB_VERSION}")
    set(TBB_LIBRARIES TBB::tbb)
else()
    # Try pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TBB QUIET tbb)
    endif()

    if(TBB_FOUND)
        message(STATUS "Found TBB via pkg-config")
        set(TBB_LIBRARIES ${TBB_LINK_LIBRARIES})
    else()
        # Last resort: try direct link
        message(WARNING "TBB not found via CMake or pkg-config. Attempting direct link.")
        message(WARNING "If build fails, install TBB: sudo apt-get install libtbb-dev")
        set(TBB_LIBRARIES tbb)
    endif()
endif()

# Header-only library target
add_library(tiled_matmul INTERFACE)
target_include_directories(tiled_matmul INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(tiled_matmul INTERFACE cxx_std_17)

# Benchmark executables
add_executable(matmul_benchmark benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE tiled_matmul ${TBB_LIBRARIES})

add_executable(matmul_benchmark_v2 benchmark_v2.cpp)
target_link_libraries(matmul_benchmark_v2 PRIVATE tiled_matmul ${TBB_LIBRARIES})

# Print configuration info
message(STATUS "=== TiledMatMul Configuration ===")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "TBB Libraries: ${TBB_LIBRARIES}")
message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "=================================")

# Installation rules
install(TARGETS matmul_benchmark DESTINATION bin)
install(FILES tiled_matmul.hpp DESTINATION include)
