#include "log.h"
#include"conv2d.cuh"

TensorDesc* conv2d_forward_shape_infer(
    const TensorDesc* in,
    const TensorDesc* kernel,
    const Conv2dDesc* conv
){
    if (!in || !kernel || !conv) {
        LOGERR("conv2d_forward_shape_infer fail, got nullptr");
        return nullptr;
    } else if (in->shape[1] != kernel->shape[1]) {
        LOGERR("in->shape[1](" + std::to_string(in->shape[1]) + ") != kernel->shape[1](" + std::to_string(kernel->shape[1]) + ")");
        return nullptr;
    } else {
        return new TensorDesc(
            "nchw",
            {
                in->shape[0],
                kernel->shape[0],
                (in->shape[2] + 2 * conv->padding - kernel->shape[2]) / conv->stride + 1,
                (in->shape[3] + 2 * conv->padding - kernel->shape[3]) / conv->stride + 1
            }
        );
    }
}

template<>
__global__ void conv2d_forward_naive(
    const float* in, const TensorDesc* desc_in,
    const float* kernel, const TensorDesc* desc_kernel, const Conv2dDesc* desc_conv,
    float* out, const TensorDesc* desc_out,
    bool invers_kernel
);


template <>
__global__ void conv2d_bias_active_forward_naive(
    const float* in, const TensorDesc* desc_in,
    const float* kernel, const TensorDesc* desc_kernel, const Conv2dDesc* desc_conv,
    const float* bias, const TensorDesc* desc_bias,
    ActivationMode active_mode,
    float* out, const TensorDesc* desc_out,
    bool invers_kernel
);

