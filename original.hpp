// #include <vector>
// #include <iostream>
// #include <memory>
// #include <cstddef>
// #include <type_traits> 
// #include <new>
// #include <utility>
// #include <functional> 

// enum class DType: uint8_t {
//     FLOAT32,
//     FLOAT64, 
//     INT32,
//     INT64,
//     BOOL
// };

// enum class Device: uint8_t {
//     CPU,
//     CUDA
// };

// namespace cppgrad{

// template<typename T>
// constexpr DType get_dtype() {
//     if constexpr (std::is_same_v<T, float>) return DType::FLOAT32;
//     if constexpr (std::is_same_v<T, double>) return DType::FLOAT64;
//     if constexpr (std::is_same_v<T, int32_t>) return DType::INT32;
//     if constexpr (std::is_same_v<T, int64_t>) return DType::INT64;
//     if constexpr (std::is_same_v<T, bool>) return DType::BOOL;
// }

// template <typename T>
// class Tensor{
//     public:
//         // CONSTRUCTORS

//         // constructor fills zeroes
//         Tensor(const std::vector<size_t>& shape, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
//             size_ = 1;
//             strides_.push_back(1); // initial stride is 1
//             for (auto dim : shape_) {
//                 size_ *= dim;
//                 if(strides_.size() < shape_.size()) strides_.push_back(strides_.back() * dim); // calculate stride for each dimension
//             }
//             data_ = std::make_unique<T[]>(size_); // by default initializes each element to zero
//             raw_ptr_ = data_.get();
//         }

//         // empty constructor & no memory allocation
//         Tensor(const std::vector<size_t>& shape, std::nullptr_t, Device device = Device::CPU, bool requires_grad = false)
//             : shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) 
//         {
//             size_ = 1;
//             strides_.push_back(1);
//             for (auto dim : shape_) {
//                 size_ *= dim;
//                 if(strides_.size() < shape_.size()) strides_.push_back(strides_.back() * dim);
//             }
//             data_ = nullptr;
//             raw_ptr_ = nullptr;
//         }

//         // constructor fills 'num' value
//         Tensor(const std::vector<size_t>& shape, T num, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
//             size_ = 1;
//             strides_.push_back(1);
//             for (auto dim : shape_) {
//                 size_ *= dim;
//                 if(strides_.size() < shape_.size()) strides_.push_back(strides_.back() * dim);
//             }
//             data_ = std::make_unique<T[]>(size_);
//             raw_ptr_ = data_.get();
//             std::fill(data_.get(), data_.get() + size_, num);
//         }

//         // constructor fills by vector
//         Tensor(const std::vector<size_t>& shape, std::vector<T>& data, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
//             size_ = 1;
//             strides_.push_back(1);
//             for (auto dim : shape_) {
//                 size_ *= dim;
//                 if(strides_.size() < shape_.size()) strides_.push_back(strides_.back() * dim);
//             }
//             if (size_ != data.size()) throw std::invalid_argument("Invalid Shape");
//             data_ = std::make_unique<T[]>(size_);
//             raw_ptr_ = data_.get();
//             std::move(data.begin(), data.end(), data_.get()); // using move instead of copy
//         }

//         // constructor fills by initializer_list
//         Tensor(const std::vector<size_t>& shape, std::initializer_list<T> data, Device device = Device::CPU, bool requires_grad = false): shape_(shape), device_(device), requires_grad_(requires_grad), dtype_(get_dtype<T>()) {
//             size_ = 1;
//             strides_.push_back(1);
//             for (auto dim : shape_) {
//                 size_ *= dim;
//                 if(strides_.size() < shape_.size()) strides_.push_back(strides_.back() * dim);
//             }
//             if (size_ != data.size()) throw std::invalid_argument("Invalid Shape");
//             data_ = std::make_unique<T[]>(size_);
//             raw_ptr_ = data_.get();
//             std::copy(data.begin(), data.end(), data_.get()); // have to use copy here because initializer_list does not support move 
//         }

//         // ATTRIBUTES
//         // get shape of the tensor
//         const std::vector<size_t>& shape() const { return shape_; }

//         // get size of the tensor
//         std::size_t size() const { return size_; }

//         // get the device of the tensor
//         std::string device() const {
//             switch(device_) {
//                 case Device::CPU: return "CPU";
//                 case Device::CUDA: return "CUDA";
//             }
//         }

//         // returns device enum
//         Device device_enum() const { return device_; }

//         // get the dtype of the tensor
//         std::string dtype() const {
//             switch(dtype_) {
//                 case DType::FLOAT32: return "FLOAT32";
//                 case DType::FLOAT64: return "FLOAT64";
//                 case DType::INT32: return "INT32";
//                 case DType::INT64: return "INT64";
//                 case DType::BOOL: return "BOOL";
//             }
//         }

//         // check if tensor requires grad or not
//         bool requires_grad() const { return requires_grad_; }

//         // get the unique data pointer of the tensor
//         T* data_ptr() const { return data_.get(); }

//         // get the raw pointer of the tensor
//         T* raw_ptr() const { return raw_ptr_; }

//         // DATA MANIPULATION

//         // accessing single element
//         T get(const std::vector<size_t>& indices) const {
//             if (indices.size() != shape_.size()) throw std::out_of_range("Invalid Tensor Indexing");
//             size_t offset = 0;
//             for (size_t i = 0; i < indices.size(); ++i) {
//                 if (indices[i] >= shape_[i]) throw std::out_of_range("Index Out of Bounds");
//                 offset += indices[i] * strides_[i];
//             }
//             return *(data_.get() + offset);  // Pointer arithmetic
//         }

//         // transponse
//         Tensor<T> transpose() const{
//             Tensor<T> transposed(raw_ptr_, shape_, strides_, device_, requires_grad_, dtype_, size_);
//             std::reverse(transposed.shape_.begin(), transposed.shape_.end());
//             std::reverse(transposed.strides_.begin(), transposed.strides_.end());
//             return transposed;
//         }

//         // reshape
//         Tensor<T> reshape(const std::vector<size_t>& new_shape) const{
//             size_t new_size = 1;
//             for(auto dim : new_shape) new_size *= dim;
//             if(new_size != size_) throw std::invalid_argument("Invalid Reshape Size");
//             std::vector<size_t> new_strides;
//             new_strides.push_back(1);
//             for(size_t i = 0; i < new_shape.size() - 1; i++) {
//                 new_strides.push_back(new_strides.back() * new_shape[i]);
//             }
//             return Tensor<T>(raw_ptr_, new_shape, new_strides, device_, requires_grad_, dtype_, size_);
//         }

//         // flatten
//         Tensor<T> flatten() const{
//             std::vector<size_t> new_shape = {size_};
//             std::vector<size_t> new_strides = {1};
//             return Tensor<T>(raw_ptr_, new_shape, new_strides, device_, requires_grad_, dtype_, size_);
//         }

//         // copy - creates a deep copy of the tensor
//         Tensor<T> copy() const{
//             Tensor<T> copied_tensor(shape_, device_, requires_grad_);
//             std::copy_n(raw_ptr_, size_, copied_tensor.raw_ptr_);
//             return copied_tensor;
//         }

//         // BLAS 

//         // dot product
//         Tensor<T> dot(const Tensor<T>& a, const Tensor<T>& b) const{
//             // vector-vector
//             if(a.shape().size() == 1 && b.shape().size() == 1){
//                 if (shape_a[0] != shape_b[0]) {
//                     throw std::invalid_argument("Incorrect Vector Dimensions");
//                 }
//                 T result_val = 0;
//                 for (size_t i = 0; i < shape_a[0]; i++) {
//                     result_val += a.raw_ptr()[i] * b.raw_ptr()[i];
//                 }
                
//             }

//             // matrix-vector

//             // matrix-matrix
//         }

//     private:
//         std::unique_ptr<T[]> data_;
//         T* raw_ptr_;
//         std::vector<size_t> shape_;
//         std::vector<size_t> strides_;
//         std::size_t size_;
//         DType dtype_;
//         Device device_;
//         bool requires_grad_;

//         // View constructor
//         Tensor(T* raw_ptr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, Device device, bool requires_grad, DType dtype, size_t size) : raw_ptr_(raw_ptr), shape_(shape), strides_(strides), size_(size), device_(device), requires_grad_(requires_grad), dtype_(dtype){
//             data_ = nullptr;
//         }
//     };

//     // IMPLEMENTATIONS OF FREE FUNCTIONS

//     // template for tensor tensor operations
//     template<typename T, typename Op>
//     Tensor<T> elementwise_op(const Tensor<T>& a, const Tensor<T>& b, Op op) {
//     if (a.shape() != b.shape()) throw std::invalid_argument("Shape mismatch in elementwise operation");
//     Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
//     for (size_t i = 0; i < a.size(); i++) result.raw_ptr()[i] = op(a.raw_ptr()[i], b.raw_ptr()[i]);
//     return result;
//     }

//     // template for tensor scalar operations
//     template<typename T, typename Op>
//     Tensor<T> elementwise_op(const Tensor<T>& a, T scalar, Op op) {
//         Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
//         for (size_t i = 0; i < a.size(); i++) result.raw_ptr()[i] = op(a.raw_ptr()[i], scalar);
//         return result;
//     }

//     // template for scalar tensor operations
//     template<typename T, typename Op>
//     Tensor<T> elementwise_op(T scalar, const Tensor<T>& a, Op op) {
//         Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
//         for (size_t i = 0; i < a.size(); i++) result.raw_ptr()[i] = op(scalar, a.raw_ptr()[i]);
//         return result;
//     }

//     // template for unary operations
//     template<typename T, typename Op>
//     Tensor<T> elementwise_op(const Tensor<T>& a, Op op) {
//         Tensor<T> result(a.shape(), a.device_enum(), a.requires_grad());
//         for (size_t i = 0; i < a.size(); i++) result.raw_ptr()[i] = op(a.raw_ptr()[i]);
//         return result;
//     }

//     // Arithmetic Operations

//     // tensor-tensor
//     template<typename T>
//     Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
//         return elementwise_op(a, b, [](T x, T y) { return x + y; });
//     }

//     template<typename T>
//     Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
//         return elementwise_op(a, b, [](T x, T y) { return x - y; });
//     }

//     template<typename T>
//     Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
//         return elementwise_op(a, b, [](T x, T y) { return x * y; });
//     }

//     template<typename T>
//     Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
//         return elementwise_op(a, b, [](T x, T y) { return x / y; });
//     }

//     // tensor-scalar
//     template<typename T>
//     Tensor<T> operator+(const Tensor<T>& a, T scalar) {
//         return elementwise_op(a, scalar, [](T x, T y) { return x + y; });
//     }

//     template<typename T>
//     Tensor<T> operator-(const Tensor<T>& a, T scalar) {
//         return elementwise_op(a, scalar, [](T x, T y) { return x - y; });
//     }

//     template<typename T>
//     Tensor<T> operator*(const Tensor<T>& a, T scalar) {
//         return elementwise_op(a, scalar, [](T x, T y) { return x * y; });
//     }

//     template<typename T>
//     Tensor<T> operator/(const Tensor<T>& a, T scalar) {
//         return elementwise_op(a, scalar, [](T x, T y) { return x / y; });
//     }

//     // scalar - tensor
//     template<typename T>
//     Tensor<T> operator+(T scalar, const Tensor<T>& a) {
//         return a + scalar;
//     }

//     template<typename T>
//     Tensor<T> operator*(T scalar, const Tensor<T>& a) {
//         return a * scalar;
//     }

//     template<typename T>
//     Tensor<T> operator-(T scalar, const Tensor<T>& a) {
//         return elementwise_op(scalar, a, [](T x, T y) { return x - y; });
//     }

//     template<typename T>
//     Tensor<T> operator/(T scalar, const Tensor<T>& a) {
//         return elementwise_op(scalar, a, [](T x, T y) { return x / y; });
//     }

//     // Mathematical Functions
//     template<typename T>
//     Tensor<T> abs(const Tensor<T>& a) {
//         return elementwise_op(a, [](T x) { return std::abs(x); });
//     }

//     template<typename T>
//     Tensor<T> exp(const Tensor<T>& a) {
//         return elementwise_op(a, [](T x) { return std::exp(x); });
//     }

//     template<typename T>
//     Tensor<T> log(const Tensor<T>& a) {
//         return elementwise_op(a, [](T x) { return std::log(x); });
//     }

//     template<typename T>
//     Tensor<T> sqrt(const Tensor<T>& a) {
//         return elementwise_op(a, [](T x) { return std::sqrt(x); });
//     }

//     // get the data of the tensor as a stream
//     template<typename T>
//     std::ostream& operator<<(std::ostream &os, const Tensor<T> &tensor){
//         for(size_t i=0; i<tensor.size(); i++){
//             os << tensor.raw_ptr()[i] << " ";
//         }
//         return os;
//     }

// };