#include<stdio.h>
#include<cuda_runtime.h>
#include"helper_cuda.h"

__global__ void fun () {
    // Block index
    int bx = threadIdx.x;
    int by = threadIdx.y;

    printf("bx:%d + by:%d = %d\n", bx, by, bx + by);
    printf("block(%d, %d, %d), thread(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
    dim3 grid(1, 1, 2), block(1, 2, 1);
    fun<<<grid, block>>>();
    checkCudaErrors(cudaDeviceSynchronize());
}