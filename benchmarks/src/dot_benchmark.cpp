#include <benchmark/benchmark.h>
#include "cppgrad/tensor.hpp"
#include <random>

std::mt19937 gen(123456);
std::uniform_real_distribution<float> dis(0.0f, 1.0f);

// tensor filling
template<typename T>
void fill_random(cppgrad::Tensor<T>& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.raw_ptr()[i] = dis(gen);
    }
}

static void BM_Dot_Vector_Small(benchmark::State& state) {
    cppgrad::Tensor<float> a({64});
    cppgrad::Tensor<float> b({64});
    fill_random(a);
    fill_random(b);

    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::dot(a, b);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Dot_Vector_Small);

static void BM_Dot_Vector_Medium(benchmark::State& state) {
    cppgrad::Tensor<float> a({1024});
    cppgrad::Tensor<float> b({1024});
    fill_random(a);
    fill_random(b);

    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::dot(a, b);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Dot_Vector_Medium);

static void BM_Dot_Vector_Large(benchmark::State& state) {
    cppgrad::Tensor<float> a({65536}); // 64K elements
    cppgrad::Tensor<float> b({65536});
    fill_random(a);
    fill_random(b);
    
    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::dot(a, b);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Dot_Vector_Large);

static void BM_Dot_Vector_Args(benchmark::State& state) {
    size_t vector_size = state.range(0);
    cppgrad::Tensor<float> a({vector_size});
    cppgrad::Tensor<float> b({vector_size});
    fill_random(a);
    fill_random(b);

    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::dot(a, b);
        benchmark::DoNotOptimize(result);
    }
    
}
BENCHMARK(BM_Dot_Vector_Args)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 20);  // 64 to 1M elements

BENCHMARK_MAIN();