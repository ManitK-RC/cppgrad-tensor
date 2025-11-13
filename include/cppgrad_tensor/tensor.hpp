#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <span>
#include "types.hpp"

namespace cppgrad_tensor {

template<typename T>
class Tensor {
public:
    // CONSTRUCTORS
    Tensor(const std::vector<size_t>& shape, Device device = Device::CPU, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, std::nullptr_t, Device device = Device::CPU, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, T num, Device device = Device::CPU, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, std::vector<T>& data, Device device = Device::CPU, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, std::initializer_list<T> data, Device device = Device::CPU, bool requires_grad = false);

    // ATTRIBUTES
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    std::size_t size() const { return size_; }
    std::string device() const {
            switch(device_) {
                case Device::CPU: return "CPU";
                case Device::CUDA: return "CUDA";
            }
        }
    Device device_enum() const { return device_; }
    std::string dtype() const {
            switch(dtype_) {
                case DType::FLOAT32: return "FLOAT32";
                case DType::FLOAT64: return "FLOAT64";
                case DType::INT32: return "INT32";
                case DType::INT64: return "INT64";
                case DType::BOOL: return "BOOL";
            }
        }
    bool requires_grad() const { return requires_grad_; }
    T* data_ptr() const { return data_.get(); }
    T* raw_ptr() const { return raw_ptr_; }

    // DATA MANIPULATION
    T get(const std::vector<size_t>& indices) const;
    Tensor<T> transpose() const;
    Tensor<T> reshape(const std::vector<size_t>& new_shape) const;
    Tensor<T> permute(const std::span<const size_t>& axes) const;
    Tensor<T> flatten() const;
    Tensor<T> copy() const;
    Tensor<T> squeeze(int dim = -1) const;

    // BLAS OPERATIONS
    static Tensor<T> dot(const Tensor<T>& a, const Tensor<T>& b);
    static Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);
    static Tensor<T> tensordot(const Tensor<T>& a, const Tensor<T>& b, std::span<const int> a_axes, std::span<const int> b_axes);
    static Tensor<T> tensordot(const Tensor<T>& a, const Tensor<T>& b, size_t axis);

    static Tensor<float> blas_dot(const Tensor<float>& a, const Tensor<float>& b);
    static Tensor<float> blas_matmul(const Tensor<float>& a, const Tensor<float>& b);
    static Tensor<float> blas_tensordot(const Tensor<float>& a, const Tensor<float>& b, std::span<const int> a_axes, std::span<const int> b_axes);
    static Tensor<float> blas_tensordot(const Tensor<float>& a, const Tensor<float>& b, size_t axis);

private:
    std::unique_ptr<T[]> data_;
    T* raw_ptr_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::size_t size_;
    DType dtype_;
    Device device_;
    bool requires_grad_;

    // View constructor
    Tensor(T* raw_ptr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, 
           Device device, bool requires_grad, DType dtype, size_t size);
};

}