#include <iostream>
#include <print>
#include "include/tensor.hpp"
using namespace std;
using namespace cppgrad;

int main() {
    std::vector<float> data = {1, 2, 3, 4};
    Tensor<float> tensor1({2,2}, data);

    cout<<tensor1.data()<<endl;
    vector<size_t> shape = tensor1.shape();
    for(auto dim : shape) cout<<dim<<" ";
    cout<<endl;
    cout<<tensor1.requires_grad()<<endl;
    cout<<tensor1.device()<<endl;
    cout<<tensor1.dtype()<<endl;
    Tensor<float> tensor2({2, 2}, {5, 6, 7, 8});
    cout<<tensor2.data()<<endl;
    shape = tensor2.shape();
    for(auto dim : shape) cout<<dim<<" ";
    cout<<endl;
    return 0;
}
