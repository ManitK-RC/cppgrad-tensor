#include <iostream>
#include <print>
#include "include/tensor.hpp"
using namespace std;
using namespace cppgrad;

int main() {
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

    // DATA MANIPULATION - TO DO

    
    return 0;
}
