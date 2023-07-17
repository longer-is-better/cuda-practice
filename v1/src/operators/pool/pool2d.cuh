#pragma once
#include <cfloat>
// #include <stdio.h>
#include"descriptor.h"


TensorDesc* pool2d_forward_shape_infer(
    const TensorDesc* in,
    const Pool2dDesc* pool
);

template<typename T>
__global__ void max_pool2d (T *input, TensorDesc *input_desc, Pool2dDesc *pool_desc, T *output, TensorDesc *output_desc) {
    for (int n = 0; n < output_desc->shape[0]; n++) {
        for (int c = blockDim.x * blockIdx.x + threadIdx.x; c < output_desc->shape[1]; c += gridDim.x * blockDim.x) {
            for (int h = blockDim.y * blockIdx.y + threadIdx.y; h < output_desc->shape[2]; h += gridDim.y * blockDim.y) {
                for (int w = blockDim.z * blockIdx.z + threadIdx.z; w < output_desc->shape[3]; w += gridDim.z * blockDim.z) {
                    float ans = -FLT_MAX;
                    int top_left_h = h * pool_desc->stride[0] - pool_desc->padding[0], top_left_w = w * pool_desc->stride[1] - pool_desc->padding[1];
                    for(int window_h = 0; window_h < pool_desc->window_shape[0]; window_h++) {
                        for(int window_w = 0; window_w < pool_desc->window_shape[1]; window_w++) {
                            int in_indx = n * input_desc->stride[0] + c * input_desc->stride[1] + (top_left_h + window_h) * input_desc->stride[2] + (top_left_w + window_w) * input_desc->stride[3];
                            float in_val = -FLT_MAX;
                            if ((top_left_h + window_h >= 0) && (top_left_h + window_h < input_desc->shape[2]) && (top_left_w + window_w >= 0) && (top_left_w + window_w < input_desc->shape[3])) in_val = input[in_indx];
                            ans = fmaxf(in_val, ans);
                        }
                    }
                    int out_idx = n * output_desc->stride[0] + c * output_desc->stride[1] + h * output_desc->stride[2] + w * output_desc->stride[3];
                    output[out_idx] = ans;
                }
            }
        }
    }
}

template<typename T>
__global__ void avg_pool2d (T *input, TensorDesc *input_desc, Pool2dDesc *pool_desc, T *output, TensorDesc *output_desc) {
    for (int n = 0; n < output_desc->shape[0]; n++) {
        for (int c = blockDim.x * blockIdx.x + threadIdx.x; c < output_desc->shape[1]; c += gridDim.x * blockDim.x) {
            for (int h = blockDim.y * blockIdx.y + threadIdx.y; h < output_desc->shape[2]; h += gridDim.y * blockDim.y) {
                for (int w = blockDim.z * blockIdx.z + threadIdx.z; w < output_desc->shape[3]; w += gridDim.z * blockDim.z) {
                    float ans = 0;
                    int top_left_h = h * pool_desc->stride[0] - pool_desc->padding[0], top_left_w = w * pool_desc->stride[1] - pool_desc->padding[1];
                    for(int window_h = 0; window_h < pool_desc->window_shape[0]; window_h++) {
                        for(int window_w = 0; window_w < pool_desc->window_shape[1]; window_w++) {
                            int in_indx = n * input_desc->stride[0] + c * input_desc->stride[1] + (top_left_h + window_h) * input_desc->stride[2] + (top_left_w + window_w) * input_desc->stride[3];
                            float in_val = 0;
                            if ((top_left_h + window_h >= 0) && (top_left_h + window_h < input_desc->shape[2]) && (top_left_w + window_w >= 0) && (top_left_w + window_w < input_desc->shape[3])) in_val = input[in_indx];
                            ans += in_val;
                        }
                    }
                    int out_idx = n * output_desc->stride[0] + c * output_desc->stride[1] + h * output_desc->stride[2] + w * output_desc->stride[3];
                    output[out_idx] = ans / (pool_desc->window_shape[0] * pool_desc->window_shape[1]);
                }
            }
        }
    }
}