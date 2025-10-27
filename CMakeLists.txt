cmake_minimum_required(VERSION 3.20)
project(cppgrad_examples)

# Add executable with all sources
add_executable(examples 
    test/examples.cpp
    src/tensor.cpp
    src/tensor_ops.cpp
    src/blas.cpp
)

# Include directories
target_include_directories(examples PRIVATE include)

# Simple C++26 flag
target_compile_options(examples PRIVATE -std=c++26)