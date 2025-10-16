#include <vector>
#include <iostream>
#include <memory>
#include <cstddef>
#include <type_traits> 
#include <new>
#include <utility>

// constructor - random

// reason for enum class is that its just 1 byte of memory compared to a string which is 24 bytes
enum class DType: uint8_t {
    FLOAT32,
    FLOAT64, 
    INT32,
    INT64,
    BOOL
};

enum class Device: uint8_t {
    CPU,
    CUDA
};

namespace cppgrad{

template<typename T>
constexpr DType get_dtype() {
    if constexpr (std::is_same_v<T, float>) return DType::FLOAT32;
    if constexpr (std::is_same_v<T, double>) return DType::FLOAT64;
    if constexpr (std::is_same_v<T, int32_t>) return DType::INT32;
    if constexpr (std::is_same_v<T, int64_t>) return DType::INT64;
    if constexpr (std::is_same_v<T, bool>) return DType::BOOL;
}

template <typename T>
class Tensor{
    public:
        // CONSTRUCTORS
        // constructor fills zeroes
        Tensor(const std::vector<size_t>& shape, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
            size_ = 1;
            for (auto dim : shape_) size_ *= dim;
            data_ = std::make_unique<T[]>(size_); // by default initializes each element to zero
        }

        // empty constructor & no memory allocation
        Tensor(const std::vector<size_t>& shape, std::nullptr_t, Device device = Device::CPU, bool requires_grad = false)
            : shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) 
        {
            size_ = 1;
            for (auto dim : shape_) size_ *= dim;
            data_ = nullptr;
        }

        // constructor fills 'num' value
        Tensor(const std::vector<size_t>& shape, T num, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
            size_ = 1;
            for (auto dim : shape_) size_ *= dim;
            data_ = std::make_unique<T[]>(size_);
            std::fill(data_.get(), data_.get() + size_, num);
        }

        // constructor fills by vector
        Tensor(const std::vector<size_t>& shape, std::vector<T>& data, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
            size_ = 1;
            for (auto dim : shape_) size_ *= dim;
            if (size_ != data.size()) {
                throw std::invalid_argument("Invalid Shape");
            }
            data_ = std::make_unique<T[]>(size_);
            std::move(data.begin(), data.end(), data_.get()); // using move instead of copy
        }

        // constructor fills by initializer_list
        Tensor(const std::vector<size_t>& shape, std::initializer_list<T> data, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
            size_ = 1;
            for (auto dim : shape_) size_ *= dim;
            if (size_ != data.size()) {
                throw std::invalid_argument("Invalid Shape");
            }
            data_ = std::make_unique<T[]>(size_);
            std::copy(data.begin(), data.end(), data_.get()); // have to use copy here because initializer_list does not support move 
        }

        // ATTRIBUTES
        // get shape of the tensor
        const std::vector<size_t>& shape() const { return shape_; }

        // get size of the tensor
        std::size_t size() const { return size_; }

        // get the device of the tensor
        std::string device() const {
            switch(device_) {
                case Device::CPU: return "CPU";
                case Device::CUDA: return "CUDA";
            }
        }

        // get the dtype of the tensor
        std::string dtype() const {
            switch(dtype_) {
                case DType::FLOAT32: return "FLOAT32";
                case DType::FLOAT64: return "FLOAT64";
                case DType::INT32: return "INT32";
                case DType::INT64: return "INT64";
                case DType::BOOL: return "BOOL";
            }
        }

        // check if tensor requires grad or not
        bool requires_grad() const { return requires_grad_; }

        // get the data pointer of the tensor
        T* data_ptr() const { return data_.get(); }

        // DATA MANIPULATION - TO DO


    private:
        std::unique_ptr<T[]> data_;
        std::vector<size_t> shape_;
        std::size_t size_;
        DType dtype_;
        Device device_;
        bool requires_grad_;
    };

// get the data of the tensor as a stream
template<typename T>
std::ostream& operator<<(std::ostream &os, const Tensor<T> &tensor){
    for(size_t i=0; i<tensor.size(); i++){
        os << tensor.data_ptr()[i] << " ";
    }
    return os;
}

};