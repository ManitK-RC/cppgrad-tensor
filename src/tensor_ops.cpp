#include "cppgrad/tensor.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <limits>

namespace cppgrad {

// Implement member functions
template <typename T>
T Tensor<T>::get(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) throw std::out_of_range("Invalid Tensor Indexing");
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i]) throw std::out_of_range("Index Out of Bounds");
        offset += indices[i] * strides_[i];
    }
    return *(data_.get() + offset);  // Pointer arithmetic
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    Tensor<T> transposed(raw_ptr_, shape_, strides_, device_, requires_grad_, dtype_, size_);
    std::reverse(transposed.shape_.begin(), transposed.shape_.end());
    std::reverse(transposed.strides_.begin(), transposed.strides_.end());
    return transposed;
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<size_t>& new_shape) const{
    size_t new_size = 1;
    for(auto dim : new_shape) new_size *= dim;
    if(new_size != size_) throw std::invalid_argument("Invalid Reshape Size");
    std::vector<size_t> new_strides;
    new_strides.push_back(1);
    for(size_t i = 0; i < new_shape.size() - 1; i++) {
        new_strides.push_back(new_strides.back() * new_shape[i]);
    }
    return Tensor<T>(raw_ptr_, new_shape, new_strides, device_, requires_grad_, dtype_, size_);
}

template <typename T>
Tensor<T> Tensor<T>::flatten() const {
    return Tensor<T>(raw_ptr_, {size_}, {1}, device_, requires_grad_, dtype_, size_);
}

template <typename T>
Tensor<T> Tensor<T>::copy() const {
    Tensor<T> copy(shape_, device_, requires_grad_);
    std::copy_n(raw_ptr_, size_, copy.raw_ptr_);
    return copy;
}

template <typename T>
Tensor<T> Tensor<T>::squeeze(int dim) const {
    const int REMOVE_ALL = std::numeric_limits<int>::min();

    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;

    if (dim == REMOVE_ALL) {
        // removing all axes of length 1
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] != 1) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
    } else {
        if (shape_.size() == 0) throw std::invalid_argument("Invalid Squeeze Dimensions");
        if (dim < 0) dim += shape_.size();
        if (dim < 0 || dim >= shape_.size() || shape_[dim] != 1) throw std::invalid_argument("Invalid Squeeze Dimensions");

        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i == dim) continue;
            new_shape.push_back(shape_[i]);
            new_strides.push_back(strides_[i]);
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
        new_strides.push_back(1);
    }

    return Tensor<T>(raw_ptr_, new_shape, new_strides, device_, requires_grad_, dtype_, size_);
}

}

template class cppgrad::Tensor<float>;
template class cppgrad::Tensor<double>;
template class cppgrad::Tensor<int32_t>;
template class cppgrad::Tensor<int64_t>;
template class cppgrad::Tensor<bool>;