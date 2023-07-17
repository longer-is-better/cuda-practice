#include "pool2d.cuh"
#include "descriptor.h"

template<>
__global__ void avg_pool2d (
    float *input,
    TensorDesc *input_desc,
    Pool2dDesc *pool_desc,
    float *output,
    TensorDesc *output_desc
);