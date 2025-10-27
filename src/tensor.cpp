#include "cppgrad/tensor.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace cppgrad {

    inline std::vector<size_t> get_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides[0] = 1;
        for (size_t i = 1; i < shape.size(); i++) strides[i] = strides[i - 1] * shape[i - 1];
        return strides;
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape, Device device, bool requires_grad)
        : shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) 
    {
        size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
        strides_ = get_strides(shape);
        data_ = std::make_unique<T[]>(size_); // zero initialized
        raw_ptr_ = data_.get();
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape, std::nullptr_t, Device device, bool requires_grad)
        : shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>())
    {
        size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
        strides_ = get_strides(shape);
        data_ = nullptr;
        raw_ptr_ = nullptr;
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape, T fill_value, Device device, bool requires_grad)
        : shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>())
    {
        size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
        strides_ = get_strides(shape);
        data_ = std::make_unique<T[]>(size_);
        raw_ptr_ = data_.get();
        std::fill(raw_ptr_, raw_ptr_ + size_, fill_value);
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape, std::vector<T>& data, Device device, bool requires_grad)
        : shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>())
    {
        size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
        strides_ = get_strides(shape);
        if (size_ != data.size())
            throw std::invalid_argument("Data size does not match shape");
        data_ = std::make_unique<T[]>(size_);
        raw_ptr_ = data_.get();
        std::move(data.begin(), data.end(), raw_ptr_);
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape, std::initializer_list<T> data, Device device, bool requires_grad)
        : shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>())
    {
        size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
        strides_ = get_strides(shape);
        if (size_ != data.size())
            throw std::invalid_argument("Data size does not match shape");
        data_ = std::make_unique<T[]>(size_);
        raw_ptr_ = data_.get();
        std::copy(data.begin(), data.end(), raw_ptr_);
    }

    // View Constructor
    template <typename T>
    Tensor<T>::Tensor(T* raw_ptr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, Device device, bool requires_grad, DType dtype, size_t size)
        : raw_ptr_(raw_ptr), shape_(shape), strides_(strides), device_(device), requires_grad_(requires_grad), dtype_(dtype), size_(size)
    {
        data_ = nullptr; // tensor does not own data
    }
}

template class cppgrad::Tensor<float>;
template class cppgrad::Tensor<double>;
template class cppgrad::Tensor<int32_t>;
template class cppgrad::Tensor<int64_t>;
template class cppgrad::Tensor<bool>;