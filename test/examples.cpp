#include <iostream>
#include <print>
#include <cppgrad_tensor/cppgrad_tensor.hpp>
using namespace std;

int main() {
    using namespace cppgrad_tensor;
    // CONSTRUCTORS

    // 1. filled with 0s
    Tensor<float> zero_tensor({2, 3});
    cout<<zero_tensor<<endl;

    // 2. empty tensor of int32
    Tensor<int32_t> empty_tensor({3, 2, 2}, nullptr);
    cout<<empty_tensor.data_ptr()<<endl;
    
    // 3. filled with a specific number
    Tensor<float> filled_tensor({2, 2}, 3.14);
    cout<<filled_tensor<<endl;

    // 4. filled with data from vector
    std::vector<float> data = {1, 2, 3, 4};
    Tensor<float> tensor1({2,2}, data);
    cout<<tensor1<<endl;

    // 5. filled with data from initializer list
    Tensor<float> tensor2({2, 2}, {5, 6, 7, 8});
    cout<<tensor2<<endl;

    // ATTRIBUTES
    cout<<tensor1.data_ptr()<<endl;
    cout<<tensor1.size()<<endl;
    vector<size_t> shape = tensor1.shape();
    for(auto dim : shape) cout<<dim<<" ";
    cout<<tensor1.requires_grad()<<endl;
    cout<<tensor1.device()<<endl;
    cout<<tensor1.dtype()<<endl;

    //DATA MANIPULATION
    
    // 1. accessing single element 
    Tensor<float> tensor3({1,2,3}, {1,2,3,4,5,6});
    float l = tensor3.get({0,1,2});
    cout<<l<<endl;

    // 2. transpose
    for(auto dim : tensor3.shape()) cout<<dim<<" ";
    cout<<endl;
    Tensor<float> tensor4 = tensor3.transpose();
    for(auto dim : tensor4.shape()) cout<<dim<<" ";
    cout<<endl;

    // 3. reshape
    Tensor<float> tensor5 = tensor3.reshape({2,3,1});
    for(auto dim : tensor5.shape()) cout<<dim<<" ";
    cout<<endl;

    // 4. flatten
    Tensor<float> tensor6 = tensor3.flatten();
    for(auto dim : tensor6.shape()) cout<<dim<<" ";
    cout<<endl;

    // 5. copy
    Tensor<float> tensor7 = tensor3.copy();
    cout<<tensor3.data_ptr()<<endl;
    cout<<tensor7.data_ptr()<<endl;

    // 6. squeeze
    Tensor<float> tensor8({1, 3, 1, 4}, {1,2,3,4,5,6,7,8,9,10,11,12});
    Tensor<float> squeezed_all = tensor8.squeeze(std::numeric_limits<int>::min());
    for (auto dim : squeezed_all.shape()) cout << dim << " ";
    cout<<endl;
    cout<<squeezed_all<<endl;

    // Using your squeeze function with dim = 0 (remove only first dim if size 1)
    Tensor<float> squeeze_dim0 = tensor8.squeeze(0);
    for (auto dim : squeeze_dim0.shape()) cout << dim << " ";
    cout<<endl;
    cout<<squeeze_dim0<<endl;

    // ARITHMETIC OPERATIONS
    Tensor<float> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> B({2, 3}, {2, 2, 2, 2, 2, 2});

    Tensor<float> C = A + B;
    cout<<C<<endl;
    Tensor<float> D = A - B;
    cout<<D<<endl;
    Tensor<float> E = A * B;
    cout<<E<<endl;
    Tensor<float> F = A / B;
    cout<<F<<endl;
    
    Tensor<float> G = A + 5.0f;
    cout<<G<<endl;
    Tensor<float> H = A - 2.0f;
    cout<<H<<endl;
    Tensor<float> I = A * 3.0f;
    cout<<I<<endl;
    Tensor<float> J = A / 2.0f;
    cout<<J<<endl;

    Tensor<float> K = 10.0f + A;
    cout<<K<<endl;
    Tensor<float> L = 10.0f * A;
    cout<<L<<endl;
    Tensor<float> M = 10.0f - A;
    cout<<M<<endl;
    Tensor<float> N = 10.0f / A;
    cout<<N<<endl;

    // MATHEMATICAL FUNCTIONS
    Tensor<float> X({2, 2}, {1, 4, 9, 16});
    Tensor<float> neg({2, 2}, {-1, -4, -9, -16});
    Tensor<float> abs_result = abs(neg); 
    cout<<abs_result<<endl;
    Tensor<float> exp_result = exp(X);
    cout<<exp_result<<endl;
    Tensor<float> log_result = log(X);
    cout<<log_result<<endl;
    Tensor<float> sqrt_result = sqrt(X);
    cout<<sqrt_result<<endl;

    // BLAS

    // dot product

    // (3,) @ (3,) -> scalar
    Tensor<float> v1({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> v2({3}, {4.0f, 5.0f, 6.0f});
    Tensor<float> vec_dot_result = Tensor<float>::dot(v1, v2);
    for(auto dim : vec_dot_result.shape()) cout<<dim<<" ";
    cout<<endl;
    cout<<vec_dot_result<<endl;

    Tensor<float> vec_dot_result_blas = Tensor<float>::blas_dot(v1, v2);
    for(auto dim : vec_dot_result_blas.shape()) cout<<dim<<" ";
    cout<<endl;
    cout<<vec_dot_result_blas<<endl;

    // matmul

    // 2,3 @ 3,2 -> 2,2
    Tensor<float> AA({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> BB({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor<float> mat_mul_result = Tensor<float>::matmul(AA, BB);
    for(auto dim : mat_mul_result.shape()) cout<<dim<<" ";
    cout<<endl;
    cout<<mat_mul_result<<endl;
    // correct result is 2,2 and values are 76 100 103 136
    Tensor<float> mat_mul_result_blas = Tensor<float>::blas_matmul(AA, BB);
    for(auto dim : mat_mul_result_blas.shape()) cout<<dim<<" ";
    cout<<endl;
    cout<<mat_mul_result_blas<<endl;

    // (3,) @ (3,2) -> (2,)
    Tensor<float> CC({3}, {4, 5, 6});
    Tensor<float> DD({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor<float> mat_mul_result2 = Tensor<float>::matmul(CC, DD);
    for(auto dim : mat_mul_result2.shape()) cout<<dim<<" ";
    cout<<endl;
    cout<<mat_mul_result2<<endl;
    // correct result is 2 and values are 122 167
    Tensor<float> mat_mul_result2_blas = Tensor<float>::blas_matmul(CC, DD);
    for(auto dim : mat_mul_result2_blas.shape()) cout<<dim<<" ";
    cout<<endl;
    cout<<mat_mul_result2_blas<<endl;

    // (4,5) @ (5,) -> (4,)
    Tensor<float> EE({4, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    Tensor<float> FF({5}, {2, 3, 4, 5, 6});
    Tensor<float> mat_mul_result3 = Tensor<float>::matmul(EE, FF);
    for(auto dim : mat_mul_result3.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result3 << endl;
    // correct result is 4 and values are 220 240 260 280 
    Tensor<float> mat_mul_result3_blas = Tensor<float>::blas_matmul(EE, FF);
    for(auto dim : mat_mul_result3_blas.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result3_blas << endl;
    
    Tensor<float> a({2, 2, 3}, {1,2,3,4,5,6,7,8,9,10,11,12});
    Tensor<float> b({2, 3, 2}, {1,2,3,4,5,6,7,8,9,10,11,12});
    auto mat_mul_result4 = Tensor<float>::matmul(a, b);
    for(auto dim : mat_mul_result4.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result4 << endl;

    auto mat_mul_result4_blas = Tensor<float>::blas_matmul(a, b);
    for(auto dim : mat_mul_result4_blas.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result4_blas << endl;

    // Broadcasting: (3, 1, 2, 3) @ (1, 4, 3, 2) → (3, 4, 2, 2)
    Tensor<float> c({3, 1, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    Tensor<float> d({4, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto mat_mul_result5 = Tensor<float>::matmul(c, d);
    for(auto dim : mat_mul_result5.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result5 << endl;

    auto mat_mul_result5_blas = Tensor<float>::blas_matmul(c, d);
    for(auto dim : mat_mul_result5_blas.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result5_blas << endl;

    // 1D @ ND
    Tensor<float> vec({3}, {1, 2, 3});
    Tensor<float> mat({2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto mat_mul_result6 = Tensor<float>::matmul(vec, mat); 
    for(auto dim : mat_mul_result6.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result6 << endl;

    auto mat_mul_result6_blas = Tensor<float>::blas_matmul(vec, mat);
    for(auto dim : mat_mul_result6_blas.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result6_blas << endl;

    // ND @ 1D
    Tensor<float> vec2({4}, {1, 2, 3, 4});
    Tensor<float> mat2({2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto mat_mul_result7 = Tensor<float>::matmul(mat2, vec2); 
    for(auto dim : mat_mul_result7.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result7 << endl;

    auto mat_mul_result7_blas = Tensor<float>::blas_matmul(mat2, vec2);
    for(auto dim : mat_mul_result7_blas.shape()) cout << dim << " ";
    cout << endl;
    cout << mat_mul_result7_blas << endl;

    // tensordot
    Tensor<float> O({2, 3, 4}, {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    Tensor<float> P({3, 4, 5}, {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60});
    Tensor<float> tensordot_result = Tensor<float>::tensordot(O,P,{1, 2}, {0, 1});
    for (auto dim : tensordot_result.shape()) cout << dim << " ";
    cout << endl;
    cout << tensordot_result << endl;

    Tensor<float> tensordot_result_blas = Tensor<float>::blas_tensordot(O,P,{1, 2}, {0, 1});
    for (auto dim : tensordot_result_blas.shape()) cout << dim << " ";
    cout << endl;
    cout << tensordot_result_blas << endl;
    
    Tensor<float> T({2, 3, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    Tensor<float> U({4, 2}, {1,2,3,4,5,6,7,8});
    Tensor<float> tensordot_result2 = Tensor<float>::tensordot(T,U,1);
    for (auto dim : tensordot_result2.shape()) cout << dim << " ";
    cout << endl;
    cout << tensordot_result2 << endl;
    
    Tensor<float> tensordot_result2_blas = Tensor<float>::blas_tensordot(T,U,1);
    for (auto dim : tensordot_result2_blas.shape()) cout << dim << " ";
    cout << endl;
    cout << tensordot_result2_blas << endl;

    return 0;
}
