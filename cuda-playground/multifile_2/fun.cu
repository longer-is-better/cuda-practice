#include<stdio.h>
#include"fun.cuh"

__global__ void fun () {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    printf("bx:%d by:%d\n", bx, by);
    printf("block(%d, %d, %d), thread(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}