#include"helper_cuda.h"
#include"fun.cuh"
#include"relu.cuh"

int main(){
    dim3 grid(1, 1, 2), block(1, 2, 1);
    fun<<<grid, block>>>();
    relu_forward<float><<<grid, block>>>(nullptr, nullptr, 2, 2, 2);
    checkCudaErrors(cudaDeviceSynchronize());
}