#include<cuda_runtime.h>
#include"helper_cuda.h"
#include"fun.cuh"

int main(){
    dim3 grid(1, 1, 2), block(1, 2, 1);
    fun<<<grid, block>>>();
    checkCudaErrors(cudaDeviceSynchronize());
}