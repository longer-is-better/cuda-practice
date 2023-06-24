#include"helper_cuda.h"
#include "ut.h"

int main(){
    mk<<<1,1>>>();
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
}