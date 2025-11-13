#include "cppgrad_tensor/tensor.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cblas.h>

namespace cppgrad_tensor {

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

    template<typename T>
    Tensor<T> Tensor<T>::tensordot(const Tensor<T>& a, const Tensor<T>& b, std::span<const int> a_axes, std::span<const int> b_axes){
        // validation 
        if (a_axes.size() != b_axes.size()) throw std::invalid_argument("Invalid Axes for Tensordot");
        size_t contracted_size = 1;
        std::vector<bool> a_contracted(a.shape().size(), false);
        std::vector<bool> b_contracted(b.shape().size(), false);
        for (size_t i = 0; i < a_axes.size(); i++) {
            if (a_axes[i] < 0 || a_axes[i] >= a.shape().size()) throw std::invalid_argument("Invalid Axes for Tensordot");
            if (b_axes[i] < 0 || b_axes[i] >= b.shape().size()) throw std::invalid_argument("Invalid Axes for Tensordot");
            if (a.shape()[a_axes[i]] != b.shape()[b_axes[i]]) throw std::invalid_argument("Invalid Axes for Tensordot");

            contracted_size *= a.shape()[a_axes[i]];
            a_contracted[a_axes[i]] = true;
            b_contracted[b_axes[i]] = true;
        }

        std::vector<size_t> a_free_axes, b_free_axes;
        std::vector<size_t> a_free_dims, b_free_dims;
        a_free_axes.reserve(a.shape().size());
        a_free_dims.reserve(a.shape().size());
        for (size_t i = 0; i < a.shape().size(); i++) {
            if (!a_contracted[i]) {
                a_free_axes.push_back(i);
                a_free_dims.push_back(a.shape()[i]);
            }
        }
        
        b_free_axes.reserve(b.shape().size());
        b_free_dims.reserve(b.shape().size());
        for (size_t i = 0; i < b.shape().size(); i++) {
            if (!b_contracted[i]) {
                b_free_axes.push_back(i);
                b_free_dims.push_back(b.shape()[i]);
            }
        }

        std::vector<size_t> a_perm;
        a_perm.reserve(a.shape().size());
        a_perm.insert(a_perm.end(), a_free_axes.begin(), a_free_axes.end());
        a_perm.insert(a_perm.end(), a_axes.begin(), a_axes.end());
        
        std::vector<size_t> b_perm;
        b_perm.reserve(b.shape().size());
        b_perm.insert(b_perm.end(), b_axes.begin(), b_axes.end());
        b_perm.insert(b_perm.end(), b_free_axes.begin(), b_free_axes.end());

        size_t a_free_size = a.size()/contracted_size;
        size_t b_free_size = b.size()/contracted_size;

        Tensor<T> a_permuted = a.permute(a_perm);
        Tensor<T> b_permuted = b.permute(b_perm);
        Tensor<T> a_reshaped = a_permuted.reshape({a_free_size, contracted_size});
        Tensor<T> b_reshaped = b_permuted.reshape({contracted_size, b_free_size});
        Tensor<T> result_mat = matmul(a_reshaped, b_reshaped);
        
        std::vector<size_t> result_shape;
        result_shape.reserve(a_free_dims.size() + b_free_dims.size());
        result_shape.insert(result_shape.end(), a_free_dims.begin(), a_free_dims.end());
        result_shape.insert(result_shape.end(), b_free_dims.begin(), b_free_dims.end());
        
        // new tensor with correct shape and data of result_mat due to dangling pointer issue
        Tensor<T> result(result_shape, result_mat.device_enum(), result_mat.requires_grad());
        std::copy_n(result_mat.raw_ptr(), result_mat.size(), result.raw_ptr());
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::tensordot(const Tensor<T>& a, const Tensor<T>& b, size_t axis){
        if(axis < 0 || axis > std::min(a.shape().size(), b.shape().size())) throw std::invalid_argument("Invalid Axis for Tensordot");
        size_t contracted_size = 1;
        for (size_t i = 0; i < axis; i++) {
            if (a.shape()[a.shape().size() - axis + i] != b.shape()[i]) throw std::invalid_argument("Invalid Axis for Tensordot");
            contracted_size *= a.shape()[a.shape().size() - axis + i];
        }
        size_t a_free_size = a.size()/contracted_size;
        size_t b_free_size = b.size()/contracted_size;

        Tensor<T> a_reshaped = a.reshape({a_free_size, contracted_size});
        Tensor<T> b_reshaped = b.reshape({contracted_size, b_free_size});
        Tensor<T> result_mat = matmul(a_reshaped, b_reshaped);
        
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < a.shape().size() - axis; i++) {
            result_shape.push_back(a.shape()[i]);
        }
        for (size_t i = axis; i < b.shape().size(); i++) {
            result_shape.push_back(b.shape()[i]);
        }
        // new tensor with correct shape and data of result_mat due to dangling pointer issue
        Tensor<T> result(result_shape, result_mat.device_enum(), result_mat.requires_grad());
        std::copy_n(result_mat.raw_ptr(), result_mat.size(), result.raw_ptr());
        return result;
    }

    template<>
    Tensor<float> Tensor<float>::blas_dot(const Tensor<float>& a, const Tensor<float>& b) {
        if(a.shape().size() == 1 && b.shape().size() == 1){
            if(a.shape()[0] != b.shape()[0]) throw std::invalid_argument("Invalid Tensor Shapes for Dot Product");
            float result = cblas_sdot(static_cast<int>(a.size()), a.raw_ptr_, 1, b.raw_ptr_, 1);
            return Tensor<float>({1}, {result});
        }
        else throw std::invalid_argument("Invalid Tensor Shapes for Dot Product");
    }

    template<>
    Tensor<float> Tensor<float>::blas_matmul(const Tensor<float>& a, const Tensor<float>& b){ // general matmul for any 2 tensors
        if(a.shape().size() == 1 && b.shape().size() == 1){
            return blas_dot(a,b);
        }
        else if (a.shape().size() == 2 && b.shape().size() == 2) {
            if (a.shape()[1] != b.shape()[0]) throw std::invalid_argument("blas_matmul: dimension mismatch");
            int M = a.shape()[0];
            int N = b.shape()[1];
            int K = a.shape()[1];
            Tensor<float> result({a.shape()[0], b.shape()[1]});
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, // do in col major order and dont transpose A or B
                        M, N, K, 
                        1.0f, // alpha
                        a.raw_ptr_, M,
                        b.raw_ptr_, K,
                        0.0f, // beta
                        result.raw_ptr_, M); // result ptr and leading dim of output
            return result;
        }
        else if (a.shape().size() == 1) {
            if (b.shape().size() == 2){
                if (a.shape()[0] != b.shape()[0]) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");
                int K = a.shape()[0];
                int N = b.shape()[1];
                Tensor<float> result({b.shape()[1]});
                int M = 1;
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 
                    1.0f,
                    a.raw_ptr_, M,
                    b.raw_ptr_, K,
                    0.0f, 
                    result.raw_ptr_, M);
                return result;
            }
            else {
                // Reshape (,n) -> (1, n)
                std::vector<size_t> new_a_shape = {1, a.shape()[0]};
                auto a_reshaped = a.reshape(new_a_shape);
                auto result = blas_matmul(a_reshaped, b);
                std::vector<size_t> final_shape;
                for (size_t i = 0; i < result.shape().size(); i++) {
                    if (i != result.shape().size() - 2) final_shape.push_back(result.shape()[i]);
                }
                Tensor<float> final_result(final_shape, result.device_enum(), result.requires_grad());
                std::copy_n(result.raw_ptr(), result.size(), final_result.raw_ptr());
                return final_result;
            }
        }
        else if (b.shape().size() == 1) {
            if (a.shape().size() == 2){
                if (a.shape()[1] != b.shape()[0]) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");
                int M = a.shape()[0];
                int K = a.shape()[1];
                Tensor<float> result({a.shape()[0]});
                cblas_sgemv(CblasColMajor, CblasNoTrans,
                            M, K, 
                            1.0f, 
                            a.raw_ptr_, M,
                            b.raw_ptr_, 1,
                            0.0f, 
                            result.raw_ptr_, 1);
                return result;
            }
            else{
                // Reshape (n,) -> (n, 1)
                std::vector<size_t> new_b_shape = {b.shape()[0], 1};
                auto b_reshaped = b.reshape(new_b_shape);
                auto result = blas_matmul(a, b_reshaped);
                std::vector<size_t> final_shape;
                for (size_t i = 0; i < result.shape().size(); i++) {
                    if (i != result.shape().size() - 1) final_shape.push_back(result.shape()[i]);
                }
                Tensor<float> final_result(final_shape, result.device_enum(), result.requires_grad());
                std::copy_n(result.raw_ptr(), result.size(), final_result.raw_ptr());
                return final_result;
            }
        }
        else if (a.shape().size() >= 1 && b.shape().size() >= 1 && (a.shape().size() > 2 || b.shape().size() > 2)) {
            if (a.shape()[a.shape().size() - 1] != b.shape()[b.shape().size() - 2]) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication"); 

            const size_t m = a.shape()[a.shape().size() - 2];
            const size_t n = a.shape()[a.shape().size() - 1];
            const size_t p = b.shape()[b.shape().size() - 1];
            
            const size_t batch_rank = std::max(a.shape().size() - 2, b.shape().size() - 2); 

            std::vector<size_t> batch_shape(batch_rank);
            std::vector<size_t> a_dims(batch_rank, 1);
            std::vector<size_t> b_dims(batch_rank, 1);
            std::vector<size_t> a_batch_strides(batch_rank, 0);
            std::vector<size_t> b_batch_strides(batch_rank, 0);

            for (size_t i = 0; i < batch_rank; i++) {
                if (i >= (batch_rank - (a.shape().size() - 2))) {
                    a_dims[i] = a.shape()[i - (batch_rank - (a.shape().size() - 2))];
                    a_batch_strides[i] = a.strides()[i - (batch_rank - (a.shape().size() - 2))];
                }
                if (i >= (batch_rank - (b.shape().size() - 2))) {
                    b_dims[i] = b.shape()[i - (batch_rank - (b.shape().size() - 2))];
                    b_batch_strides[i] = b.strides()[i - (batch_rank - (b.shape().size() - 2))];
                }
                if (a_dims[i] != b_dims[i] && a_dims[i] != 1 && b_dims[i] != 1) throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");
                batch_shape[i] = std::max(a_dims[i], b_dims[i]);
            }

            // result_shape = batch_shape + [m, p]
            std::vector<size_t> result_shape = batch_shape;
            result_shape.push_back(m);
            result_shape.push_back(p);
            Tensor<float> result(result_shape, a.device_enum(), a.requires_grad() || b.requires_grad());

            // number of batches
            size_t batch_count = 1;
            for (auto d : batch_shape) batch_count *= d;

            // element strides for matrix dims (assumes your strides are element counts)
            const size_t a_row_stride = a.strides()[a.shape().size() - 2];
            const size_t a_col_stride = a.strides()[a.shape().size() - 1];
            const size_t b_row_stride = b.strides()[b.shape().size() - 2];
            const size_t b_col_stride = b.strides()[b.shape().size() - 1];
            const size_t r_row_stride = result.strides()[batch_rank];
            const size_t r_col_stride = result.strides()[batch_rank + 1];

            std::vector<size_t> coords(batch_rank, 0);
            size_t base_a = 0;
            size_t base_b = 0;
            size_t base_r = 0;

            // temporaries (reused) for non-contiguous blocks; column-major packing (since we'll call cblas_sgemm with CblasColMajor)
            std::vector<float> A_tmp, B_tmp, C_tmp;
            A_tmp.reserve(m * n);
            B_tmp.reserve(n * p);
            C_tmp.reserve(m * p);

            for (size_t flat = 0; flat < batch_count; ++flat) {
                const float* A_block = a.raw_ptr() + base_a;
                const float* B_block = b.raw_ptr() + base_b;
                float* R_block = result.raw_ptr() + base_r;

                // check column-major contiguous: index (i,j) stored at j*m + i -> row_stride == 1 and col_stride == m
                const bool A_contig = (a_row_stride == 1 && a_col_stride == m);
                const bool B_contig = (b_row_stride == 1 && b_col_stride == n);
                const bool R_contig = (r_row_stride == 1 && r_col_stride == m);

                const float *A_for_blas = nullptr, *B_for_blas = nullptr;
                float *C_for_blas = nullptr;
                int lda = 0, ldb = 0, ldc = 0; // leading dims (#rows for column-major)

                if (A_contig && B_contig && R_contig) {
                    // direct pointers 
                    A_for_blas = A_block;
                    B_for_blas = B_block;
                    C_for_blas = R_block;
                    lda = static_cast<int>(m);
                    ldb = static_cast<int>(n);
                    ldc = static_cast<int>(m);
                } else {
                    // pack A (m * n) into column-major A_tmp
                    A_tmp.assign(m * n, 0.0f);
                    for (size_t j = 0; j < n; ++j)
                        for (size_t i = 0; i < m; ++i)
                            A_tmp[j * m + i] = *(A_block + i * a_row_stride + j * a_col_stride);

                    // pack B (n * p) into column-major B_tmp
                    B_tmp.assign(n * p, 0.0f);
                    for (size_t j = 0; j < p; ++j)
                        for (size_t i = 0; i < n; ++i)
                            B_tmp[j * n + i] = *(B_block + i * b_row_stride + j * b_col_stride);

                    C_tmp.assign(m * p, 0.0f);

                    A_for_blas = A_tmp.data();
                    B_for_blas = B_tmp.data();
                    C_for_blas = C_tmp.data();
                    lda = static_cast<int>(m);
                    ldb = static_cast<int>(n);
                    ldc = static_cast<int>(m);
                }

                // GEMM: C = A * B  (column-major)
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(m), static_cast<int>(p), static_cast<int>(n),
                            1.0f,
                            A_for_blas, lda,
                            B_for_blas, ldb,
                            0.0f,
                            C_for_blas ? C_for_blas : R_block, ldc);

                if (!(A_contig && B_contig && R_contig)) {
                    for (size_t j = 0; j < p; ++j)
                        for (size_t i = 0; i < m; ++i)
                            *(R_block + i * r_row_stride + j * r_col_stride) = C_tmp[j * m + i];
                }

                for (int axis = int(batch_rank) - 1; axis >= 0; --axis) {
                    coords[axis] += 1;
                    if (coords[axis] < batch_shape[axis]) {
                        if (a_dims[axis] > 1) base_a += (a_batch_strides[axis]);
                        if (b_dims[axis] > 1) base_b += (b_batch_strides[axis]);
                        base_r += (result.strides()[axis]);
                        break;
                    } else {
                        coords[axis] = 0;
                        if (a_dims[axis] > 1) base_a -= (a_batch_strides[axis]) * (batch_shape[axis]-1);
                        if (b_dims[axis] > 1) base_b -= (b_batch_strides[axis]) * (batch_shape[axis]-1);
                        base_r -= (result.strides()[axis]) * (batch_shape[axis] - 1);
                    }
                }
            }

            return result;
        }
        else {
            throw std::invalid_argument("Invalid Tensor Shapes for Matrix Multiplication");
        }
    }

    template<>
    Tensor<float> Tensor<float>::blas_tensordot(const Tensor<float>& a, const Tensor<float>& b, size_t axis){
        if(axis < 0 || axis > std::min(a.shape().size(), b.shape().size())) throw std::invalid_argument("Invalid Axis for Tensordot");
        size_t contracted_size = 1;
        for (size_t i = 0; i < axis; i++) {
            if (a.shape()[a.shape().size() - axis + i] != b.shape()[i]) throw std::invalid_argument("Invalid Axis for Tensordot");
            contracted_size *= a.shape()[a.shape().size() - axis + i];
        }
        size_t a_free_size = a.size()/contracted_size;
        size_t b_free_size = b.size()/contracted_size;

        Tensor<float> a_reshaped = a.reshape({a_free_size, contracted_size});
        Tensor<float> b_reshaped = b.reshape({contracted_size, b_free_size});
        Tensor<float> result_mat = blas_matmul(a_reshaped, b_reshaped);
        
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < a.shape().size() - axis; i++) {
            result_shape.push_back(a.shape()[i]);
        }
        for (size_t i = axis; i < b.shape().size(); i++) {
            result_shape.push_back(b.shape()[i]);
        }
        // new tensor with correct shape and data of result_mat due to dangling pointer issue
        Tensor<float> result(result_shape, result_mat.device_enum(), result_mat.requires_grad());
        std::copy_n(result_mat.raw_ptr(), result_mat.size(), result.raw_ptr());
        return result;
    }

    template<>
    Tensor<float> Tensor<float>::blas_tensordot(const Tensor<float>& a, const Tensor<float>& b, std::span<const int> a_axes, std::span<const int> b_axes){
        // validation 
        if (a_axes.size() != b_axes.size()) throw std::invalid_argument("Invalid Axes for Tensordot");
        size_t contracted_size = 1;
        std::vector<bool> a_contracted(a.shape().size(), false);
        std::vector<bool> b_contracted(b.shape().size(), false);
        for (size_t i = 0; i < a_axes.size(); i++) {
            if (a_axes[i] < 0 || a_axes[i] >= a.shape().size()) throw std::invalid_argument("Invalid Axes for Tensordot");
            if (b_axes[i] < 0 || b_axes[i] >= b.shape().size()) throw std::invalid_argument("Invalid Axes for Tensordot");
            if (a.shape()[a_axes[i]] != b.shape()[b_axes[i]]) throw std::invalid_argument("Invalid Axes for Tensordot");

            contracted_size *= a.shape()[a_axes[i]];
            a_contracted[a_axes[i]] = true;
            b_contracted[b_axes[i]] = true;
        }

        std::vector<size_t> a_free_axes, b_free_axes;
        std::vector<size_t> a_free_dims, b_free_dims;
        a_free_axes.reserve(a.shape().size());
        a_free_dims.reserve(a.shape().size());
        for (size_t i = 0; i < a.shape().size(); i++) {
            if (!a_contracted[i]) {
                a_free_axes.push_back(i);
                a_free_dims.push_back(a.shape()[i]);
            }
        }
        
        b_free_axes.reserve(b.shape().size());
        b_free_dims.reserve(b.shape().size());
        for (size_t i = 0; i < b.shape().size(); i++) {
            if (!b_contracted[i]) {
                b_free_axes.push_back(i);
                b_free_dims.push_back(b.shape()[i]);
            }
        }

        std::vector<size_t> a_perm;
        a_perm.reserve(a.shape().size());
        a_perm.insert(a_perm.end(), a_free_axes.begin(), a_free_axes.end());
        a_perm.insert(a_perm.end(), a_axes.begin(), a_axes.end());
        
        std::vector<size_t> b_perm;
        b_perm.reserve(b.shape().size());
        b_perm.insert(b_perm.end(), b_axes.begin(), b_axes.end());
        b_perm.insert(b_perm.end(), b_free_axes.begin(), b_free_axes.end());

        size_t a_free_size = a.size()/contracted_size;
        size_t b_free_size = b.size()/contracted_size;

        Tensor<float> a_permuted = a.permute(a_perm);
        Tensor<float> b_permuted = b.permute(b_perm);
        Tensor<float> a_reshaped = a_permuted.reshape({a_free_size, contracted_size});
        Tensor<float> b_reshaped = b_permuted.reshape({contracted_size, b_free_size});
        Tensor<float> result_mat = blas_matmul(a_reshaped, b_reshaped);
        
        std::vector<size_t> result_shape;
        result_shape.reserve(a_free_dims.size() + b_free_dims.size());
        result_shape.insert(result_shape.end(), a_free_dims.begin(), a_free_dims.end());
        result_shape.insert(result_shape.end(), b_free_dims.begin(), b_free_dims.end());
        
        // new tensor with correct shape and data of result_mat due to dangling pointer issue
        Tensor<float> result(result_shape, result_mat.device_enum(), result_mat.requires_grad());
        std::copy_n(result_mat.raw_ptr(), result_mat.size(), result.raw_ptr());
        return result;
    }

}

template class cppgrad_tensor::Tensor<float>;
template class cppgrad_tensor::Tensor<double>;
template class cppgrad_tensor::Tensor<int32_t>;
template class cppgrad_tensor::Tensor<int64_t>;
template class cppgrad_tensor::Tensor<bool>;