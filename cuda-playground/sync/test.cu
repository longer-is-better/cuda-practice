#include<stdio.h>
__global__ void f() {
    if(threadIdx.x % 2 == 0){
        printf("%d \n", threadIdx.x);
        // __syncthreads();
    }else{
        printf("%d \n", threadIdx.x);
        // __syncthreads();
    }
}

int main() {
    f<<<1, 20>>>();
    cudaDeviceSynchronize();
}