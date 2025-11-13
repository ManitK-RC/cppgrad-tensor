cmake_minimum_required(VERSION 3.20)
project(cppgrad_tensor)

add_library(cppgrad_tensor
    src/tensor.cpp
    src/tensor_ops.cpp  
    src/blas.cpp
)

target_include_directories(cppgrad_tensor PUBLIC include)

target_compile_options(cppgrad_tensor PRIVATE -std=c++26 -O3)

add_executable(examples test/examples.cpp)
target_link_libraries(examples PRIVATE cppgrad_tensor)
target_compile_options(examples PRIVATE -std=c++26)

add_subdirectory(benchmarks)