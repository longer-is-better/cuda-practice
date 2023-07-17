#include "pool2d.cuh"
#include "descriptor.h"


TensorDesc* pool2d_forward_shape_infer(
    const TensorDesc* in,
    const Pool2dDesc* pool
){
    if (!in || !pool) {
        return nullptr;
    } else {
        return new TensorDesc(
            "nchw",
            {
                in->shape[0],
                in->shape[1],
                (in->shape[2] + 2 * pool->padding[0] - pool->window_shape[0]) / pool->stride[0] + 1,
                (in->shape[3] + 2 * pool->padding[1] - pool->window_shape[1]) / pool->stride[1] + 1
            }
        );
    }
}

template<>
__global__ void max_pool2d (
    float *input,
    TensorDesc *input_desc,
    Pool2dDesc *pool_desc,
    float *output,
    TensorDesc *output_desc
);