#include"stdio.h"
#include "ut.h"

__device__ int f(int a){ 
    return a*a;
}



__global__ void mk(){
    // int a = 10;
    // printf("%s%d", "aaaaaaaa", a);
    printf("%d", f(5));
}
