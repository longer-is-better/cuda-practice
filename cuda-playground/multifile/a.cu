#include"stdio.h"
#include"helper_cuda.h"
#include "ut.h"

__global__ void mk(){
    // int a = 10;
    // printf("%s%d", "aaaaaaaa", a);
    printf("%d", f(5));
}


int main(){
    mk<<<1,1>>>();
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
}