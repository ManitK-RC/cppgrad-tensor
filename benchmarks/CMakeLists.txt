cmake_minimum_required(VERSION 3.20)
find_package(benchmark REQUIRED)

set(OpenBLAS_DIR "/opt/homebrew/opt/openblas/lib/cmake/openblas")
find_package(OpenBLAS REQUIRED)

add_executable(transpose_benchmarks
    src/transpose_benchmark.cpp
)

add_executable(matmul_benchmarks
    src/matmul_benchmark.cpp
)

add_executable(blas_matmul_benchmarks
    src/blas_matmul_benchmark.cpp
)

add_executable(dot_benchmarks
    src/dot_benchmark.cpp
)

add_executable(blas_dot_benchmarks
    src/blas_dot_benchmark.cpp
)

add_executable(tensordot_benchmarks
    src/tensordot_benchmark.cpp
)

add_executable(blas_tensordot_benchmarks
    src/blas_tensordot_benchmark.cpp
)

target_link_libraries(transpose_benchmarks 
    PRIVATE 
    cppgrad_tensor
    benchmark::benchmark
)

target_link_libraries(matmul_benchmarks 
    PRIVATE 
    cppgrad_tensor
    benchmark::benchmark
)

target_link_libraries(blas_matmul_benchmarks 
    PRIVATE 
    cppgrad_tensor
    benchmark::benchmark
)

target_link_libraries(dot_benchmarks 
    PRIVATE 
    cppgrad_tensor
    benchmark::benchmark
)

target_link_libraries(blas_dot_benchmarks 
    PRIVATE 
    cppgrad_tensor
    benchmark::benchmark
)

target_link_libraries(tensordot_benchmarks 
    PRIVATE 
    cppgrad_tensor
    benchmark::benchmark
)

target_link_libraries(blas_tensordot_benchmarks 
    PRIVATE 
    cppgrad_tensor
    benchmark::benchmark
)

target_include_directories(transpose_benchmarks PRIVATE ../include)
target_include_directories(matmul_benchmarks PRIVATE ../include)
target_include_directories(blas_matmul_benchmarks PRIVATE ../include)
target_include_directories(dot_benchmarks PRIVATE ../include)
target_include_directories(blas_dot_benchmarks PRIVATE ../include)
target_include_directories(tensordot_benchmarks PRIVATE ../include)
target_include_directories(blas_tensordot_benchmarks PRIVATE ../include)
target_include_directories(cppgrad_tensor PRIVATE 
    ${OpenBLAS_INCLUDE_DIRS}
    src/
)

target_link_libraries(cppgrad_tensor PRIVATE 
    ${OpenBLAS_LIBRARIES}
)

target_compile_options(transpose_benchmarks PRIVATE 
    -std=c++26
    -O3
    -DNDEBUG
    -march=native
)

target_compile_options(matmul_benchmarks PRIVATE 
    -std=c++26
    -O3
    -DNDEBUG
    -march=native
)

target_compile_options(blas_matmul_benchmarks PRIVATE 
    -std=c++26
    -O3
    -DNDEBUG
    -march=native
)

target_compile_options(dot_benchmarks PRIVATE 
    -std=c++26
    -O3
    -DNDEBUG
    -march=native
)

target_compile_options(blas_dot_benchmarks PRIVATE 
    -std=c++26
    -O3
    -DNDEBUG
    -march=native
)

target_compile_options(tensordot_benchmarks PRIVATE 
    -std=c++26
    -O3
    -DNDEBUG
    -march=native
)

target_compile_options(blas_tensordot_benchmarks PRIVATE 
    -std=c++26
    -O3
    -DNDEBUG
    -march=native
)