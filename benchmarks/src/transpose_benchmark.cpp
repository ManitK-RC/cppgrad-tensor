#include <benchmark/benchmark.h>
#include "cppgrad_tensor/tensor.hpp"
#include <random>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0f, 1.0f);

// Fill tensor with random data
template<typename T>
void fill_random(cppgrad_tensor::Tensor<T>& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.raw_ptr()[i] = dis(gen);
    }
}

// Benchmark transpose for different sizes
static void BM_Transpose_Small(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> tensor({32, 32});
    fill_random(tensor);
    
    for (auto _ : state) {
        auto result = tensor.transpose();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Transpose_Small);

static void BM_Transpose_Medium(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> tensor({256, 256});
    fill_random(tensor);
    
    for (auto _ : state) {
        auto result = tensor.transpose();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Transpose_Medium);

static void BM_Transpose_Large(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> tensor({1024, 1024});
    fill_random(tensor);
    
    for (auto _ : state) {
        auto result = tensor.transpose();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Transpose_Large);

// Benchmark different tensor shapes
static void BM_Transpose_3D(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> tensor({64, 64, 64});
    fill_random(tensor);
    
    for (auto _ : state) {
        auto result = tensor.transpose();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Transpose_3D);

// Benchmark with different data types
static void BM_Transpose_Double(benchmark::State& state) {
    cppgrad_tensor::Tensor<double> tensor({256, 256});
    std::uniform_real_distribution<double> dis_double(0.0, 1.0);
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.raw_ptr()[i] = dis_double(gen);
    }
    
    for (auto _ : state) {
        auto result = tensor.transpose();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Transpose_Double);

BENCHMARK_MAIN();