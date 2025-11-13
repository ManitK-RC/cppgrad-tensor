#pragma once
#include <functional>
#include <ostream>
#include "tensor.hpp"

namespace cppgrad_tensor {

// Template function declarations
template<typename T, typename Op>
Tensor<T> elementwise_op(const Tensor<T>& a, const Tensor<T>& b, Op op) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Shape mismatch in elementwise operation");
    }
    Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
    for (size_t i = 0; i < a.size(); i++) {
        result.raw_ptr()[i] = op(a.raw_ptr()[i], b.raw_ptr()[i]);
    }
    return result;
}

template<typename T, typename Op>
Tensor<T> elementwise_op(const Tensor<T>& a, T scalar, Op op) {
    Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
    for (size_t i = 0; i < a.size(); i++) {
        result.raw_ptr()[i] = op(a.raw_ptr()[i], scalar);
    }
    return result;
}

template<typename T, typename Op>
Tensor<T> elementwise_op(T scalar, const Tensor<T>& a, Op op) {
    Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
    for (size_t i = 0; i < a.size(); i++) {
        result.raw_ptr()[i] = op(scalar, a.raw_ptr()[i]);
    }
    return result;
}

template<typename T, typename Op>
Tensor<T> elementwise_op(const Tensor<T>& a, Op op) {
    Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
    for (size_t i = 0; i < a.size(); i++) {
        result.raw_ptr()[i] = op(a.raw_ptr()[i]);
    }
    return result;
}

// Arithmetic operators
template<typename T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
    return elementwise_op(a, b, [](T x, T y) { return x + y; });
}

template<typename T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
    return elementwise_op(a, b, [](T x, T y) { return x - y; });
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
    return elementwise_op(a, b, [](T x, T y) { return x * y; });
}

template<typename T>
Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
    return elementwise_op(a, b, [](T x, T y) { return x / y; });
}

template<typename T>
Tensor<T> operator+(const Tensor<T>& a, T scalar) {
    return elementwise_op(a, scalar, [](T x, T y) { return x + y; });
}

template<typename T>
Tensor<T> operator-(const Tensor<T>& a, T scalar) {
    return elementwise_op(a, scalar, [](T x, T y) { return x - y; });
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& a, T scalar) {
    return elementwise_op(a, scalar, [](T x, T y) { return x * y; });
}

template<typename T>
Tensor<T> operator/(const Tensor<T>& a, T scalar) {
    return elementwise_op(a, scalar, [](T x, T y) { return x / y; });
}

template<typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& a) {
    return a + scalar;
}

template<typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& a) {
    return a * scalar;
}

template<typename T>
Tensor<T> operator-(T scalar, const Tensor<T>& a) {
    return elementwise_op(scalar, a, [](T x, T y) { return x - y; });
}

template<typename T>
Tensor<T> operator/(T scalar, const Tensor<T>& a) {
    return elementwise_op(scalar, a, [](T x, T y) { return x / y; });
}

// Mathematical functions implementation
template<typename T>
Tensor<T> abs(const Tensor<T>& a) {
    return elementwise_op(a, [](T x) { return std::abs(x); });
}

template<typename T>
Tensor<T> exp(const Tensor<T>& a) {
    return elementwise_op(a, [](T x) { return std::exp(x); });
}

template<typename T>
Tensor<T> log(const Tensor<T>& a) {
    return elementwise_op(a, [](T x) { return std::log(x); });
}

template<typename T>
Tensor<T> sqrt(const Tensor<T>& a) {
    return elementwise_op(a, [](T x) { return std::sqrt(x); });
}

// Stream operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
    for (size_t i = 0; i < tensor.size(); i++) {
        os << tensor.raw_ptr()[i] << " ";
    }
    return os;
}
}