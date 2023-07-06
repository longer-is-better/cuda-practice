#pragma once
#include"print_tensor.h"
#include"descriptor.h"


TensorDesc* conv2d_forward_shape_infer(
    const TensorDesc* in,
    const TensorDesc* kernel,
    const Conv2dDesc* conv
){
    if (!in || !kernel || !conv) {
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

// __constant__ T kernel_const[64*1024/sizeof(T)];
// extern __shared__ float block_in[];
template <typename T>
__global__ void conv2d_forward_naive(
    const T* in, const TensorDesc* desc_in,
    const T* kernel, const TensorDesc* desc_kernel, const Conv2dDesc* desc_conv,
    T* out, const TensorDesc* desc_out
){
    for (int n = 0; n < desc_out->shape[0]; n++) {
        for (int c = blockDim.x * blockIdx.x + threadIdx.x; c < desc_out->shape[1]; c += gridDim.x * blockDim.x) {
            for (int h = blockDim.y * blockIdx.y + threadIdx.y; h < desc_out->shape[2]; h += gridDim.y * blockDim.y) {
                for (int w = blockDim.z * blockIdx.z + threadIdx.z; w < desc_out->shape[3]; w += gridDim.z * blockDim.z) {
                    float ans = 0;
                    int top_left_h = h * desc_conv->stride - desc_conv->padding, top_left_w = w * desc_conv->stride - desc_conv->padding;
                    for (int in_c = 0; in_c < desc_in->shape[1]; in_c++) {
                        for(int kernel_h = 0; kernel_h < desc_kernel->shape[2]; kernel_h++) {
                            for(int kernel_w = 0; kernel_w < desc_kernel->shape[3]; kernel_w++) {
                                int in_indx = n * desc_in->stride[0] + in_c * desc_in->stride[1] + (top_left_h + kernel_h) * desc_in->stride[2] + (top_left_w + kernel_w) * desc_in->stride[3];
                                // int kernel_idx = c * desc_kernel->stride[0] + in_c * desc_kernel->stride[1] + kernel_h * desc_kernel->stride[2] + kernel_w * desc_kernel->stride[3];
                                int kernel_idx = c * desc_kernel->stride[0] + in_c * desc_kernel->stride[1] + (desc_kernel->shape[2] - kernel_h - 1) * desc_kernel->stride[2] + (desc_kernel->shape[3] - kernel_w - 1) * desc_kernel->stride[3];
                                float in_val = 0;
                                if ((top_left_h + kernel_h >= 0) && (top_left_h + kernel_h < desc_in->shape[2]) && (top_left_w + kernel_w >= 0) && (top_left_w + kernel_w < desc_in->shape[3])) in_val = in[in_indx];
                                ans += in_val * kernel[kernel_idx];
                            }
                        }
                    }
                    int out_idx = n * desc_out->stride[0] + c * desc_out->stride[1] + h * desc_out->stride[2] + w * desc_out->stride[3];
                    out[out_idx] = ans;
                }
            }
        }
    }
}


template <typename T>
__global__ void conv2d_bias_active_forward_naive(
    const T* in, const TensorDesc* desc_in,
    const T* kernel, const TensorDesc* desc_kernel, const Conv2dDesc* desc_conv,
    const T* bias, const TensorDesc* desc_bias,
    ActivationMode active_mode,
    T* out, const TensorDesc* desc_out
){
    for (int n = 0; n < desc_out->shape[0]; n++) {
        for (int c = blockDim.x * blockIdx.x + threadIdx.x; c < desc_out->shape[1]; c += gridDim.x * blockDim.x) {
            for (int h = blockDim.y * blockIdx.y + threadIdx.y; h < desc_out->shape[2]; h += gridDim.y * blockDim.y) {
                for (int w = blockDim.z * blockIdx.z + threadIdx.z; w < desc_out->shape[3]; w += gridDim.z * blockDim.z) {
                    float ans = 0;
                    int top_left_h = h * desc_conv->stride - desc_conv->padding, top_left_w = w * desc_conv->stride - desc_conv->padding;
                    for (int in_c = 0; in_c < desc_in->shape[1]; in_c++) {
                        for(int kernel_h = 0; kernel_h < desc_kernel->shape[2]; kernel_h++) {
                            for(int kernel_w = 0; kernel_w < desc_kernel->shape[3]; kernel_w++) {
                                int in_indx = n * desc_in->stride[0] + in_c * desc_in->stride[1] + (top_left_h + kernel_h) * desc_in->stride[2] + (top_left_w + kernel_w) * desc_in->stride[3];
                                // int kernel_idx = c * desc_kernel->stride[0] + in_c * desc_kernel->stride[1] + kernel_h * desc_kernel->stride[2] + kernel_w * desc_kernel->stride[3];
                                int kernel_idx = c * desc_kernel->stride[0] + in_c * desc_kernel->stride[1] + (desc_kernel->shape[2] - kernel_h - 1) * desc_kernel->stride[2] + (desc_kernel->shape[3] - kernel_w - 1) * desc_kernel->stride[3];
                                float in_val = 0;
                                if ((top_left_h + kernel_h >= 0) && (top_left_h + kernel_h < desc_in->shape[2]) && (top_left_w + kernel_w >= 0) && (top_left_w + kernel_w < desc_in->shape[3])) in_val = in[in_indx];
                                ans += in_val * kernel[kernel_idx];
                            }
                        }
                    }
                    int out_idx = n * desc_out->stride[0] + c * desc_out->stride[1] + h * desc_out->stride[2] + w * desc_out->stride[3];
                    if (active_mode == RELU) out[out_idx] = (ans + bias[c]) > 0 ? (ans + bias[c]) > 0 : 0;
                    else assert(1);
                }
            }
        }
    }
}