#include <vector>
#include <iostream>
#include <memory>
#include <cstddef>
#include <type_traits> 

// attributes - shape, dtype, device, requires_grad
// storing the tensor in a 1d format
// there is a pointer to data stored in the tensor class

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
        // constructor that takes shape, device and requires_grad and allocates memory for data
        Tensor(const std::vector<size_t>& shape, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
            size_t n = 1;
            for (auto dim : shape_) n *= dim;
            data_ = std::make_unique<T[]>(n);
        }

        // constructor that takes shape, data, device and requires_grad and allocates memory for data and moves the data
        Tensor(const std::vector<size_t>& shape, std::vector<T>& data, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
            size_t n = 1;
            for (auto dim : shape_) n *= dim;
            if (n != data.size()) {
                throw std::invalid_argument("Invalid Shape");
            }
            data_ = std::make_unique<T[]>(n);
            std::move(data.begin(), data.end(), data_.get()); // using move instead of copy because 
        }

        Tensor(const std::vector<size_t>& shape, std::initializer_list<T> data, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
            size_t n = 1;
            for (auto dim : shape_) n *= dim;
            if (n != data.size()) {
                throw std::invalid_argument("Invalid Shape");
            }
            data_ = std::make_unique<T[]>(n);
            std::copy(data.begin(), data.end(), data_.get()); // have to use copy here because initializer_list does not support move 
        }

        // accessors
        const std::vector<size_t>& shape() const { return shape_; }
        std::string device() const {
            switch(device_) {
                case Device::CPU: return "CPU";
                case Device::CUDA: return "CUDA";
            }
        }
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
        const T* data() const { return data_.get(); } // this returns a pointer to the data


    private:
        std::unique_ptr<T[]> data_;
        std::vector<size_t> shape_;
        DType dtype_;
        Device device_;
        bool requires_grad_;
    };
};