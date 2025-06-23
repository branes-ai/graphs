cmake_minimum_required(VERSION 3.31)

####
## Enable project() command to manage VERSION variables
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

####
# Set project variables
if(NOT DEFINED GRAPHS_VERSION_MAJOR)
  set(GRAPHS_VERSION_MAJOR 3)
endif()
if(NOT DEFINED GRAPHS_VERSION_MINOR)
  set(GRAPHS_VERSION_MINOR 81)
endif()
if(NOT DEFINED GRAPHS_VERSION_PATCH)
  set(GRAPHS_VERSION_PATCH 1)
endif()
project(graphs 
    DESCRIPTION "BRANES.AI Tools and Utilities for Deep Learning graph creation and reflection"
	VERSION "${GRAPHS_VERSION_MAJOR}.${GRAPHS_VERSION_MINOR}.${GRAPHS_VERSION_PATCH}" 
	LANGUAGES C CXX ASM
	HOMEPAGE_URL "https://github.com/branes-ai/graphs")

# Set the C++ standard (e.g., C++20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # For strict standard conformance
#message(STATUS "C++11 has been enabled by default")
#message(STATUS "C++14 has been enabled by default")
#message(STATUS "C++17 has been enabled by default")
message(STATUS "C++20 has been enabled by default")
#message(STATUS "C++23 has been enabled by default")

####
# Set environmental options for tracing, testing, and verbosity
option(GRAPHS_CMAKE_TRACE "Tracing CMake results, i.e. printing variable settings." OFF)
option(GRAPHS_ENABLE_TESTS "Enable the build and run of tests." ON)
option(GRAPHS_VERBOSE_TESTS "Always print test output, otherwise only errors. Only relevant when tests enabled." OFF)

# regression test intensity
# sanity is level 1 (=min) regression testing
option(GRAPHS_BUILD_REGRESSION_SANITY           "Set to ON to enable regression level sanity"         ON)
# stress is level 4 (= max) regression testing
option(GRAPHS_BUILD_REGRESSION_STRESS           "Set to ON to enable regression level stress"         OFF)
# four levels of intensity
option(GRAPHS_BUILD_REGRESSION_LEVEL_1          "Set to ON to enable regression level 1"              OFF)
option(GRAPHS_BUILD_REGRESSION_LEVEL_2          "Set to ON to enable regression level 2"              OFF)
option(GRAPHS_BUILD_REGRESSION_LEVEL_3          "Set to ON to enable regression level 3"              OFF)
option(GRAPHS_BUILD_REGRESSION_LEVEL_4          "Set to ON to enable regression level 4"              OFF)

# documentation
option(GRAPHS_BUILD_DOCS                        "Set to ON to build documentation"                    OFF)

option(GRAPHS_USE_FOLDERS                       "Enable solution folders in Visual Studio"            ON)
if (GRAPHS_USE_FOLDERS) 
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

macro(trace_variable variable)
    if (GRAPHS_CMAKE_TRACE)
        message(STATUS "${variable} = ${${variable}}")
    endif()
endmacro()

# Use cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake")
trace_variable(CMAKE_MODULE_PATH)

# Must be located in root dir, doesn't work in tests
if (GRAPHS_ENABLE_TESTS)
    enable_testing()
    # include(Dart)
endif()

# enable the regression testing level of the build
if(GRAPHS_BUILD_REGRESSION_STRESS)
	message(STATUS "GRAPHS_BUILD_REGRESSION_STRESS   ${GRAPHS_BUILD_REGRESSION_STRESS}")
	set(GRAPHS_BUILD_REGRESSION_LEVEL_4 ON)
	set(GRAPHS_BUILD_REGRESSION_SANITY OFF)
elseif(GRAPHS_BUILD_REGRESSION_SANITY)
	message(STATUS "GRAPHS_BUILD_REGRESSION_SANITY   ${GRAPHS_BUILD_REGRESSION_SANITY}")
	set(GRAPHS_BUILD_REGRESSION_LEVEL_1 ON)
endif(GRAPHS_BUILD_REGRESSION_STRESS)

if(GRAPHS_BUILD_REGRESSION_LEVEL_4)
	message(STATUS "GRAPHS_BUILD_REGRESSION_LEVEL_4  ${GRAPHS_BUILD_REGRESSION_LEVEL_4}")
	add_definitions(-D REGRESSION_LEVEL_OVERRIDE)
	add_definitions(-D REGRESSION_LEVEL_1=1)
	add_definitions(-D REGRESSION_LEVEL_2=1)
	add_definitions(-D REGRESSION_LEVEL_3=1)
	add_definitions(-D REGRESSION_LEVEL_4=1)
elseif(GRAPHS_BUILD_REGRESSION_LEVEL_3)
	message(STATUS "GRAPHS_BUILD_REGRESSION_LEVEL_3  ${GRAPHS_BUILD_REGRESSION_LEVEL_3}")
	add_definitions(-D REGRESSION_LEVEL_OVERRIDE)
	add_definitions(-D REGRESSION_LEVEL_1=1)
	add_definitions(-D REGRESSION_LEVEL_2=1)
	add_definitions(-D REGRESSION_LEVEL_3=1)
	add_definitions(-D REGRESSION_LEVEL_4=0)
elseif(GRAPHS_BUILD_REGRESSION_LEVEL_2)
	message(STATUS "GRAPHS_BUILD_REGRESSION_LEVEL_2  ${GRAPHS_BUILD_REGRESSION_LEVEL_2}")
	add_definitions(-D REGRESSION_LEVEL_OVERRIDE)
	add_definitions(-D REGRESSION_LEVEL_1=1)
	add_definitions(-D REGRESSION_LEVEL_2=1)
	add_definitions(-D REGRESSION_LEVEL_3=0)
	add_definitions(-D REGRESSION_LEVEL_4=0)
elseif(GRAPHS_BUILD_REGRESSION_LEVEL_1)
	message(STATUS "GRAPHS_BUILD_REGRESSION_LEVEL_1  ${GRAPHS_BUILD_REGRESSION_LEVEL_1}")
	add_definitions(-D REGRESSION_LEVEL_OVERRIDE)
	add_definitions(-D REGRESSION_LEVEL_1=1)
	add_definitions(-D REGRESSION_LEVEL_2=0)
	add_definitions(-D REGRESSION_LEVEL_3=0)
	add_definitions(-D REGRESSION_LEVEL_4=0)
else(GRAPHS_BUILD_REGRESSION_LEVEL_4)
	message(STATUS "Nothing has been set")
endif(GRAPHS_BUILD_REGRESSION_LEVEL_4)

# Add directories to search for header files
# include/ is for public headers
# src/ is often added for private headers or convenience, though
# good practice pushes for only public headers in include/
include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src" # Often useful for headers that are "internal" to a module but not "public" for the entire project
)

# Find all source files in the src/ directory and its subdirectories
# This is a convenient way for smaller to medium projects.
# For very large projects, you might list source files explicitly or use more granular globbing.
file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SOURCES_CORE "${CMAKE_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE SOURCES_UTILITIES "${CMAKE_SOURCE_DIR}/src/utilities/*.cpp")

# Combine all source files
list(APPEND SOURCES ${SOURCES_CORE} ${SOURCES_UTILITIES})

####
# macro to read all cpp files in a directory
# and create a test target for that cpp file
macro (compile_all testing prefix folder)
    # cycle through the sources
    # For the according directories, we assume that each cpp file is a separate test
    # so, create a executable target and an associated test target
    foreach (source ${ARGN})
        get_filename_component (test ${source} NAME_WE)
        string(REPLACE " " ";" new_source ${source})
        set(test_name ${prefix}_${test})
        #message(STATUS "Add test ${test_name} from source ${new_source}.")
        add_executable (${test_name} ${new_source})

        #add_custom_target(valid SOURCES ${SOURCES})
        set_target_properties(${test_name} PROPERTIES FOLDER ${folder})
        if (${testing} STREQUAL "true")
            if (GRAPHS_CMAKE_TRACE)
                message(STATUS "testing: ${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name}")
            endif()
            add_test(${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name})
        endif()
    endforeach (source)
endmacro (compile_all)

####
# macro to create an executable target consisting of all cpp files in a directory
# and create a test target for that cpp file
macro (compile_multifile_target testing test_name folder)
    message(STATUS "Add test ${test_name} from source folder ${folder}.")
    add_executable (${test_name} ${ARGN})

    #add_custom_target(valid SOURCES ${SOURCES})
    set_target_properties(${test_name} PROPERTIES FOLDER ${folder})
    if (${testing} STREQUAL "true")
        if (GRAPHS_CMAKE_TRACE)
            message(STATUS "testing: ${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name}")
        endif()
        add_test(${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name})
    endif()
endmacro (compile_multifile_target)

# construct the different executables
add_subdirectory("tools/pt")

# Example: If you were linking to an external library (e.g., an imaginary 'foo' library)
# find_package(Foo REQUIRED)
# target_link_libraries(${PROJECT_NAME} Foo::Foo)

# Example: If you have a 'tests' directory with a separate test executable
# add_subdirectory(tests) # This would require a CMakeLists.txt inside the tests/ directory
