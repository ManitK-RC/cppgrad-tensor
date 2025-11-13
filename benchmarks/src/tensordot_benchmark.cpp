#include <benchmark/benchmark.h>
#include "cppgrad_tensor/tensor.hpp"
#include <random>
#include <vector>

std::mt19937 gen(123456);
std::uniform_real_distribution<float> dis(0.0f, 1.0f);

template<typename T>
void fill_random(cppgrad_tensor::Tensor<T>& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.raw_ptr()[i] = dis(gen);
    }
}

static void BM_Tensordot_SingleAxis_Small(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> a({16, 32, 64});
    cppgrad_tensor::Tensor<float> b({32, 64, 128});
    fill_random(a);
    fill_random(b);
    
    size_t axis = 2;
    size_t M = 16;
    size_t K = 32 * 64;
    size_t N = 128;

    for (auto _ : state) {
        auto result = cppgrad_tensor::Tensor<float>::tensordot(a, b, axis);
        benchmark::DoNotOptimize(result);
        
    }
}
BENCHMARK(BM_Tensordot_SingleAxis_Small);

static void BM_Tensordot_SingleAxis_Medium(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> a({32, 64, 128});
    cppgrad_tensor::Tensor<float> b({64, 128, 256});
    fill_random(a);
    fill_random(b);
    
    size_t axis = 2;
    size_t M = 32;
    size_t K = 64 * 128;
    size_t N = 256;
    
    for (auto _ : state) {
        auto result = cppgrad_tensor::Tensor<float>::tensordot(a, b, axis);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Tensordot_SingleAxis_Medium);

static void BM_Tensordot_SingleAxis_Large(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> a({64, 128, 256});
    cppgrad_tensor::Tensor<float> b({128, 256, 512});
    fill_random(a);
    fill_random(b);
    
    size_t axis = 2;
    size_t M = 64;
    size_t K = 128 * 256;
    size_t N = 512;

    for (auto _ : state) {
        auto result = cppgrad_tensor::Tensor<float>::tensordot(a, b, axis);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Tensordot_SingleAxis_Large);

static void BM_Tensordot_MultiAxis_Small(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> a({2, 4, 8, 16});
    cppgrad_tensor::Tensor<float> b({4, 8, 16, 32});
    fill_random(a);
    fill_random(b);
    std::vector<int> a_axes = {1,2};
    std::vector<int> b_axes = {0,1};

    size_t contracted_size = 4 * 8;
    size_t M = 2 * 16;
    size_t N = 16 * 32;
    for (auto _ : state) {
        auto result = cppgrad_tensor::Tensor<float>::tensordot(a, b, a_axes, b_axes);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Tensordot_MultiAxis_Small);

static void BM_Tensordot_MultiAxis_Medium(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> a({4, 8, 16, 32});
    cppgrad_tensor::Tensor<float> b({8, 16, 32, 64});
    fill_random(a);
    fill_random(b);
    
    std::vector<int> a_axes = {1, 2};
    std::vector<int> b_axes = {0, 1};
    
    size_t contracted_size = 8 * 16;
    size_t M = 4 * 32;
    size_t N = 32 * 64;
    for (auto _ : state) {
        auto result = cppgrad_tensor::Tensor<float>::tensordot(a, b, a_axes, b_axes);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Tensordot_MultiAxis_Medium);

static void BM_Tensordot_MultiAxis_Large(benchmark::State& state) {
    cppgrad_tensor::Tensor<float> a({8, 16, 32, 64});
    cppgrad_tensor::Tensor<float> b({16, 32, 64, 128});
    fill_random(a);
    fill_random(b);
    
    std::vector<int> a_axes = {1, 2};
    std::vector<int> b_axes = {0, 1};
    
    size_t contracted_size = 16 * 32;
    size_t M = 8 * 64;
    size_t N = 64 * 128;
    for (auto _ : state) {
        auto result = cppgrad_tensor::Tensor<float>::tensordot(a, b, a_axes, b_axes);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Tensordot_MultiAxis_Large);

static void BM_Tensordot_HighDim(benchmark::State& state) {
    size_t batch = state.range(0);
    cppgrad_tensor::Tensor<float> a({batch, 16, 32, 64});
    cppgrad_tensor::Tensor<float> b({32, 64, 128});
    fill_random(a);
    fill_random(b);
    
    size_t axis = 2;
    size_t M = batch * 16;
    size_t K = 32 * 64;
    size_t N = 128;
    for (auto _ : state) {
        auto result = cppgrad_tensor::Tensor<float>::tensordot(a, b, axis);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Tensordot_HighDim)->RangeMultiplier(2)->Range(4, 32);

BENCHMARK_MAIN();