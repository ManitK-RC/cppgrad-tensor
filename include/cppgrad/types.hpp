#pragma once
#include <cstddef>
#include <type_traits>
#include <cstdint>

namespace cppgrad {

enum class DType : uint8_t {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    BOOL
};

enum class Device : uint8_t {
    CPU,
    CUDA
};

template<typename T>
constexpr DType get_dtype() {
    if constexpr (std::is_same_v<T, float>) return DType::FLOAT32;
    if constexpr (std::is_same_v<T, double>) return DType::FLOAT64;
    if constexpr (std::is_same_v<T, int32_t>) return DType::INT32;
    if constexpr (std::is_same_v<T, int64_t>) return DType::INT64;
    if constexpr (std::is_same_v<T, bool>) return DType::BOOL;
}
}