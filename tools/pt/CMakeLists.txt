set(INSPECTOR "inspector")
project(${INSPECTOR})

set(INSPECTOR_SRCS pytorch_model_inspector.cpp)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# basic usage: invoke cmake with the path to LibTorch
#TORCH_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
#cmake .. -DCMAKE_PREFIX_PATH="$TORCH_PATH"

# Auto-detect LibTorch if not specified
if(NOT DEFINED CMAKE_PREFIX_PATH)
    # Try to find LibTorch from Python installation
    execute_process(
        COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_CMAKE_PATH)
        message(STATUS "Found LibTorch via Python: ${TORCH_CMAKE_PATH}")
        set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
    endif()
endif()

# Common LibTorch installation paths to check
set(TORCH_SEARCH_PATHS
    "/usr/local/libtorch"
    "/opt/libtorch"
    "$ENV{HOME}/libtorch"
    "C:/libtorch"
    "C:/Program Files/libtorch"
)

# Try to find LibTorch in common locations
if(NOT Torch_DIR AND NOT CMAKE_PREFIX_PATH)
    foreach(SEARCH_PATH ${TORCH_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
            message(STATUS "Found LibTorch at: ${SEARCH_PATH}")
            set(CMAKE_PREFIX_PATH "${SEARCH_PATH}")
            break()
        endif()
    endforeach()
endif()

# Find required packages
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

# Print LibTorch information
message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "LibTorch CXX flags: ${TORCH_CXX_FLAGS}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "LibTorch include directories: ${TORCH_INCLUDE_DIRS}")

add_executable(${INSPECTOR} ${INSPECTOR_SRCS})

# Set properties for the executable
set_property(TARGET ${INSPECTOR} PROPERTY CXX_STANDARD 17)

# Link libraries
target_link_libraries(${INSPECTOR} 
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# Apply LibTorch compile flags
target_compile_definitions(${INSPECTOR} PRIVATE ${TORCH_CXX_FLAGS})

# Handle different LibTorch configurations
if(MSVC)
    # Suppress specific warnings that come from LibTorch headers
    target_compile_options(${INSPECTOR} PRIVATE
        /wd4267  # conversion from 'size_t' to 'type', possible loss of data
        /wd4244  # conversion from 'type1' to 'type2', possible loss of data
        /wd4996  # deprecated function warnings
        /wd4251  # class needs to have dll-interface
        /wd4275  # non dll-interface class used as base
        /wd4305  # truncation from 'double' to 'float'
        /wd4624  # destructor was implicitly defined as deleted
        /bigobj  # increase number of sections in object files
    )
    
    # Set the correct subsystem for console application
    set_target_properties(${INSPECTOR} PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )

    # Windows-specific settings
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${INSPECTOR}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:${INSPECTOR}>)
    
    # Set runtime library for MSVC
    set_property(TARGET ${INSPECTOR} PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    # Define preprocessor macros to reduce warnings
    target_compile_definitions(${INSPECTOR} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _SCL_SECURE_NO_WARNINGS
        NOMINMAX  # Prevent Windows.h from defining min/max macros
    )
endif()

# Optional: Add CUDA support detection
if(TORCH_CUDA)
    message(STATUS "CUDA support detected in LibTorch")
    enable_language(CUDA)
    set_property(TARGET ${INSPECTOR} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "CPU-only LibTorch detected")
endif()

# Installation
install(TARGETS ${INSPECTOR}
        RUNTIME DESTINATION bin)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Optional: Create a simple test
#enable_testing()
#add_test(NAME model_inspector_help 
#         COMMAND model_inspector)
#set_tests_properties(model_inspector_help PROPERTIES
#                     WILL_FAIL TRUE)  # Expected to fail without arguments

# Package configuration (optional)
#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)

# Print usage instructions
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build Instructions:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch")
message(STATUS "  cmake --build . --config Release")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./model_inspector <model.pt> [input_shape...]")
message(STATUS "========================================")
