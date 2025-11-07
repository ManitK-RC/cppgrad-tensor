#include <benchmark/benchmark.h>
#include "cppgrad/tensor.hpp"
#include <random>

std::mt19937 gen(123456);
std::uniform_real_distribution<float> dis(0.0f, 1.0f);

template<typename T>
void fill_random(cppgrad::Tensor<T>& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.raw_ptr()[i] = dis(gen);
    }
}

static void BM_Matmul_2D_Small(benchmark::State& state) {
    cppgrad::Tensor<float> a({64, 128});
    cppgrad::Tensor<float> b({128, 32});
    fill_random(a);
    fill_random(b);

    size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}
BENCHMARK(BM_Matmul_2D_Small);

static void BM_Matmul_2D_Medium(benchmark::State& state) {
    cppgrad::Tensor<float> a({256, 512});
    cppgrad::Tensor<float> b({512, 256});
    fill_random(a);
    fill_random(b);
    
    size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}
BENCHMARK(BM_Matmul_2D_Medium);

static void BM_Matmul_2D_Large(benchmark::State& state) {
    cppgrad::Tensor<float> a({1024, 2048});
    cppgrad::Tensor<float> b({2048, 1024});
    fill_random(a);
    fill_random(b);
    
    size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}
BENCHMARK(BM_Matmul_2D_Large);

static void BM_Matmul_Batched_Small(benchmark::State& state) {
    cppgrad::Tensor<float> a({16, 32, 64});
    cppgrad::Tensor<float> b({16, 64, 32});
    fill_random(a);
    fill_random(b);
    
    size_t batch = a.shape()[0];
    size_t M = a.shape()[1], K = a.shape()[2], N = b.shape()[2];

    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["BatchSize"] = batch;
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}
BENCHMARK(BM_Matmul_Batched_Small);

static void BM_Matmul_Batched_Medium(benchmark::State& state) {
    cppgrad::Tensor<float> a({32, 64, 128});
    cppgrad::Tensor<float> b({32, 128, 64});
    fill_random(a);
    fill_random(b);
    size_t batch = a.shape()[0];
    size_t M = a.shape()[1], K = a.shape()[2], N = b.shape()[2];

    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["BatchSize"] = batch;
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}
BENCHMARK(BM_Matmul_Batched_Medium);

static void BM_Matmul_Batched_Large(benchmark::State& state) {
    cppgrad::Tensor<float> a({64, 128, 256});
    cppgrad::Tensor<float> b({64, 256, 128});
    fill_random(a);
    fill_random(b);
    size_t batch = a.shape()[0];
    size_t M = a.shape()[1], K = a.shape()[2], N = b.shape()[2];

    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["BatchSize"] = batch;
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}
BENCHMARK(BM_Matmul_Batched_Large);

static void BM_Matmul_VectorMatrix(benchmark::State& state) {
    size_t size = state.range(0);
    cppgrad::Tensor<float> a({size});
    cppgrad::Tensor<float> b({size, size});
    fill_random(a);
    fill_random(b);

    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["VectorSize"] = size;
    }
}
BENCHMARK(BM_Matmul_VectorMatrix)->RangeMultiplier(2)->Range(64, 4096);

static void BM_Matmul_MatrixVector(benchmark::State& state) {
    size_t size = state.range(0);
    cppgrad::Tensor<float> a({size, size});
    cppgrad::Tensor<float> b({size});
    fill_random(a);
    fill_random(b);
    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["VectorSize"] = size;
    }
}
BENCHMARK(BM_Matmul_MatrixVector)->RangeMultiplier(2)->Range(64, 4096);

static void BM_Matmul_2D_Args(benchmark::State& state) {
    size_t M = state.range(0);
    size_t K = state.range(1);
    size_t N = state.range(2);
    
    cppgrad::Tensor<float> a({M, K});
    cppgrad::Tensor<float> b({K, N});
    fill_random(a);
    fill_random(b);
    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}

BENCHMARK(BM_Matmul_2D_Args)->Args({128, 256, 128});    // Small
BENCHMARK(BM_Matmul_2D_Args)->Args({512, 1024, 512});   // Medium  
BENCHMARK(BM_Matmul_2D_Args)->Args({1024, 2048, 1024}); // Large
BENCHMARK(BM_Matmul_2D_Args)->Args({2048, 4096, 2048}); // Very Large


static void BM_Matmul_Broadcast(benchmark::State& state) {
    size_t batch1 = state.range(0);
    size_t batch2 = state.range(1);
    
    // Test broadcasting: (batch1, M, K) @ (batch2, K, N)
    cppgrad::Tensor<float> a({batch1, 64, 128});
    cppgrad::Tensor<float> b({batch2, 128, 32});
    fill_random(a);
    fill_random(b);
    
    size_t M = 64, K = 128, N = 32;
    for (auto _ : state) {
        auto result = cppgrad::Tensor<float>::matmul(a, b);
        benchmark::DoNotOptimize(result);
        state.counters["Batch1"] = batch1;
        state.counters["Batch2"] = batch2;
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}

BENCHMARK(BM_Matmul_Broadcast)->Args({1, 32});    // Broadcast A
BENCHMARK(BM_Matmul_Broadcast)->Args({32, 1});    // Broadcast B
BENCHMARK(BM_Matmul_Broadcast)->Args({16, 16});   // Same batch

BENCHMARK_MAIN();