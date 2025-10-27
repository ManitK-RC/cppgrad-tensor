#include "cppgrad/tensor.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace cppgrad {

    // Scalar dot product between two vectors
    template <typename T>
    Tensor<T> Tensor<T>::dot(const Tensor<T>& a, const Tensor<T>& b){
        if(a.shape().size() == 1 && b.shape().size() == 1){
            if(a.shape()[0] != b.shape()[0]) throw std::invalid_argument("Invalid Tensor Shapes for Dot Product");
            T dot_prod = 0;
            for (size_t i = 0; i < a.shape()[0]; ++i) dot_prod += a.raw_ptr()[i * a.strides()[0]] * b.raw_ptr()[i * b.strides()[0]];
            return Tensor<T>({1}, dot_prod, a.device_enum(), a.requires_grad() || b.requires_grad()); // returning scalar tensor
        }
        else throw std::invalid_argument("Invalid Tensor Shapes for Dot Product");
    }

    template <typename T>
    inline Tensor<T> matrix_matrix_matmul(const Tensor<T>& a, const Tensor<T>& b){
        // (M, K) @ (K, N) -> (M, N)
        if(a.shape().back() != b.shape().front()) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");
        
        // m = a[0], k = a[1], n = b[1]
        Tensor<T> result({a.shape()[0], b.shape()[1]}, a.device_enum(), a.requires_grad() || b.requires_grad());
        
        // i,j is at index i + j * num_rows
        for (size_t j = 0; j < b.shape()[1]; j++) {        // columns of result
            for (size_t i = 0; i < a.shape()[0]; i++) {    // rows of result
                T sum = 0;
                for (size_t k = 0; k < a.shape()[1]; k++) { // inner dimension
                    // a[i,k] = a_ptr[i + k * m]
                    // b[k,j] = b_ptr[k + j * n]
                    sum += a.raw_ptr()[i + k * a.shape()[0]] * b.raw_ptr()[k + j * b.shape()[0]];
                }
                result.raw_ptr()[i + j * a.shape()[0]] = sum;
            }
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> vector_matrix_matmul(const Tensor<T>& a, const Tensor<T>& b){
        // (n,) @ (n, p) -> (1, p)
        if(a.shape()[0] != b.shape()[0]) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");

        // creating result (p,)
        Tensor<T> result({b.shape()[1]}, a.device_enum(), a.requires_grad() || b.requires_grad());
        
        for (size_t j = 0; j < b.shape()[1]; j++) {
            T sum = 0;
            for (size_t k = 0; k < a.shape()[0]; k++) {
                sum += a.raw_ptr()[k] * b.raw_ptr()[k + j*a.shape()[0]];
            }
            result.raw_ptr()[j] = sum;
        }
    return result;
    }

    template <typename T>
    inline Tensor<T> matrix_vector_matmul(const Tensor<T>& a, const Tensor<T>& b){
        // (m,n) @ (n,) -> (m,1) -> (m,)
        if (a.shape()[1] != b.shape()[0]) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");

        // creating result (m,)
        Tensor<T> result({a.shape()[0]}, a.device_enum(), a.requires_grad() || b.requires_grad());
        for (size_t i = 0; i < a.shape()[0]; i++) {
            T sum = 0;
            for (size_t j = 0; j < a.shape()[1]; j++) {
                sum += a.raw_ptr()[i + j*a.shape()[0]]*b.raw_ptr()[j];
            }
            result.raw_ptr()[i] = sum;
        }
        
        return result;
    }

    template <typename T>
    inline Tensor<T> batched_matmul(const Tensor<T>& a, const Tensor<T>& b){
        // (..., m, n) @ (..., n, p) -> (..., m, p)
        if (a.shape()[a.shape().size() - 1] != b.shape()[b.shape().size() - 2]) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication"); 
        const size_t m = a.shape()[a.shape().size() - 2];
        const size_t n = a.shape()[a.shape().size() - 1];
        const size_t p = b.shape()[b.shape().size() - 1];
        
        const size_t batch_rank = std::max(a.shape().size() - 2, b.shape().size() - 2); 
        // number of batch axes in a and b are all except the last 2 & batch rank is the number of axes after broadcasting

        std::vector<size_t> batch_shape(batch_rank); // this stores the dims of axes after broadcasting

        std::vector<size_t> a_dims(batch_rank, 1); // dims of a 
        std::vector<size_t> b_dims(batch_rank, 1); // dims of b
        // init all to 1 because of broadcasting

        std::vector<size_t> a_batch_strides(batch_rank, 0);
        std::vector<size_t> b_batch_strides(batch_rank, 0);

        // offset for a/b is batch_rank - a/b.shape().size() - 2 - this gives number of dims which are to be broadcasted for a/b
        // for one of a or b the offset will be 0

        for (size_t i = 0; i < batch_rank; i++) {
            if (i >= (batch_rank - (a.shape().size() - 2))) { // if tensor a has an offset due to broadcasting, if we are at indices before the offset they would just be 1
                a_dims[i] = a.shape()[i - (batch_rank - (a.shape().size() - 2))];
                a_batch_strides[i] = a.strides()[i - (batch_rank - (a.shape().size() - 2))];
            }

            if (i >= (batch_rank - (b.shape().size() - 2))) { // // if tensor b has an offset due to broadcasting
                b_dims[i] = b.shape()[i - (batch_rank - (b.shape().size() - 2))];
                b_batch_strides[i] = b.strides()[i - (batch_rank - (b.shape().size() - 2))];
            }

            if (a_dims[i] != b_dims[i] && a_dims[i] != 1 && b_dims[i] != 1) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication"); // raise an error because these tensor shapes are not supported
            batch_shape[i] = std::max(a_dims[i], b_dims[i]); // take the one which is not 1 because 1 will have to be dropped later on
        }

        // result_shape = batch_shape + [m, p]
        std::vector<size_t> result_shape = batch_shape;
        result_shape.push_back(m);
        result_shape.push_back(p);
        Tensor<T> result(result_shape, a.device_enum(), a.requires_grad() || b.requires_grad()); // creating the result tensor of the desired shape

        // number of batches is size of batch_shape i.e. product of all elements in it
        size_t batch_count = 1;
        for (auto d : batch_shape) batch_count *= d;

        std::vector<size_t> coords(batch_rank, 0); // to store current batches coordinates 
        size_t base_a = 0;
        size_t base_b = 0;
        size_t base_r = 0;
        T sum;
        for (size_t flat = 0; flat < batch_count; ++flat) { // looping over each batch

            // doing matmul on current batch
            for (size_t j = 0; j < p; j++) { // the col
                for (size_t i = 0; i < m; i++) { // the row
                    sum = 0;
                    for (size_t k = 0; k < n; k++) { // the shared axis

                        // for a = base of a + i * stride of m + k * stride of n
                        // for b = base of b + k * stride of n + j * stride of p
                        sum += a.raw_ptr()[(base_a + (i) * a.strides()[a.shape().size() - 2] + (k) * a.strides()[a.shape().size() - 1])] * b.raw_ptr()[(base_b + (k) * b.strides()[b.shape().size() - 2] + (j) * b.strides()[b.shape().size() - 1])];
                    }
                    result.raw_ptr()[(base_r + j*result.strides()[batch_rank + 1] + i*result.strides()[batch_rank])] = sum;
                }
            }

            // changing coordinates & offsets for the next batch 
            for (int axis = batch_rank-1; axis >= 0; axis--) {

                // go from current batch to next batch 0,0,0 to 0,0,1 to 0,0,2 ...
                coords[axis] += 1;
                if (coords[axis] < batch_shape[axis]) {
                    if (a_dims[axis] > 1) base_a += (a_batch_strides[axis]);
                    if (b_dims[axis] > 1) base_b += (b_batch_strides[axis]);
                    base_r += (result.strides()[axis]);
                    break;
                }
                else {
                    coords[axis] = 0;
                    if (a_dims[axis] > 1) base_a -= (a_batch_strides[axis])*(batch_shape[axis]-1);
                    if (b_dims[axis] > 1) base_b -= (b_batch_strides[axis])*(batch_shape[axis]-1);
                    base_r -= (result.strides()[axis])*(batch_shape[axis] - 1);
                }
            }
        }

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T>& a, const Tensor<T>& b){
        if(a.shape().size() == 1 && b.shape().size() == 1){
            return dot(a,b);
        }
        else if (a.shape().size() == 2 && b.shape().size() == 2) {
            return matrix_matrix_matmul(a, b);
        }
        else if (a.shape().size() == 1) {
            if (b.shape().size() == 2) return vector_matrix_matmul(a, b);
            else {
                // Reshape (,n) -> (1, n)
                std::vector<size_t> new_a_shape = {1, a.shape()[0]};
                auto a_reshaped = a.reshape(new_a_shape);
                auto result = matmul(a_reshaped, b);
                std::vector<size_t> final_shape;
                for (size_t i = 0; i < result.shape().size(); i++) {
                    if (i != result.shape().size() - 2) final_shape.push_back(result.shape()[i]);
                }
                Tensor<T> final_result(final_shape, result.device_enum(), result.requires_grad());
                std::copy_n(result.raw_ptr(), result.size(), final_result.raw_ptr());
                return final_result;
            }
        }
        else if (b.shape().size() == 1) {
            if (a.shape().size() == 2) return matrix_vector_matmul(a, b);
            else{
                // Reshape (n,) -> (n, 1)
                std::vector<size_t> new_b_shape = {b.shape()[0], 1};
                auto b_reshaped = b.reshape(new_b_shape);
                auto result = matmul(a, b_reshaped);
                std::vector<size_t> final_shape;
                for (size_t i = 0; i < result.shape().size(); i++) {
                    if (i != result.shape().size() - 1) final_shape.push_back(result.shape()[i]);
                }
                Tensor<T> final_result(final_shape, result.device_enum(), result.requires_grad());
                std::copy_n(result.raw_ptr(), result.size(), final_result.raw_ptr());
                return final_result;
            }
        }
        else if (a.shape().size() >= 1 && b.shape().size() >= 1 && (a.shape().size() > 2 || b.shape().size() > 2)) {
            return batched_matmul(a, b);
        }
        else {
            throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");
        }
    }
}

template class cppgrad::Tensor<float>;
template class cppgrad::Tensor<double>;
template class cppgrad::Tensor<int32_t>;
template class cppgrad::Tensor<int64_t>;
template class cppgrad::Tensor<bool>;