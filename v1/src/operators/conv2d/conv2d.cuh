#pragma once

#include"descriptor.h"


template <typename T>
__global__ void conv2d_forward_naive(
    T* in, TensorDesc desc_in,
    T* kernel, TensorDesc desc_kernel,
    T* out, TensorDesc desc_out,
    const Conv2dDesc& desc_conv
){
    // for (int n = 0; n < desc_out.shape[0]; n++) {
    //     for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < desc_out.shape[1]; x += gridDim.x * blockDim.x) {
    //         for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < desc_out.shape[2]; y += gridDim.y * blockDim.y) {
    //             for (int z = blockDim.z * blockIdx.z + threadIdx.z; z < desc_out.shape[3]; z += gridDim.z * blockDim.z) {
    //                 int ans_c = blockDim.x * blockIdx.x + threadIdx.x;
    //                 int ans_h = blockDim.y * blockIdx.y + threadIdx.y;
    //                 int ans_w = blockDim.z * blockIdx.z + threadIdx.z;
    //                 out[idx] = (in[idx] > static_cast<T>(0)) ? in[idx] : static_cast<T>(0);
    //             }
    //         }
    //     }
    // }
}

template <typename T>
__global__ void conv2d_forward_v1(
    T* in, TensorDesc desc_in,
    T* kernel, TensorDesc desc_kernel,
    T* out, TensorDesc desc_out,
    const Conv2dDesc& desc_conv
){
    // __shared__ T block_in[desc_in.shape[0]][desc_in.shape[1]][desc_conv.stride * blockDim.y + desc_kernel.shape[2] - 1][desc_conv.stride * blockDim.z + desc_kernel.shape[3] - 1];
    // for (int n = 0; n < desc_out.shape[0]; n++) {
    //     // grid stride loop
    //     for (int block_in_c = blockDim.x * blockIdx.x + threadIdx.x; block_in_c < desc_in.shape[1]; block_in_c += gridDim.x * blockDim.x) {
    //         for (int ans_h = blockDim.y * blockIdx.y + threadIdx.y; ans_h < desc_out.shape[2]; ans_h += gridDim.y * blockDim.y) {
    //             for (int ans_w = blockDim.z * blockIdx.z + threadIdx.z; ans_w < desc_out.shape[3]; ans_w += gridDim.z * blockDim.z) {
    //                 // copy ans element's in tile
    //                 for (int in_h = desc_conv.stride * ans_h - desc_conv.padding; in_h < desc_conv.stride * (ans_h + 1) - desc_conv.padding; in_h++) {
    //                     for (int in_w = desc_conv.stride * ans_w - desc_conv.padding; in_w < desc_conv.stride * (ans_w + 1) - desc_conv.padding; in_w++) {
    //                         block_in[n][block_in_c][in_h - blockDim.y * blockIdx.y][in_w - blockDim.z * blockIdx.z] = in[n][block_in_c][in_h][in_w];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    // __syncthreads();
    // __shared__ T block_ans[desc_out.shape[0]][desc_out.shape[1]][desc_out.shape[2]][desc_out.shape[3]];
    // for (int n = 0; n < desc_out.shape[0]; n++) {
    //     for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < desc_out.shape[1]; x += gridDim.x * blockDim.x) {
    //         for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < desc_out.shape[2]; y += gridDim.y * blockDim.y) {
    //             for (int z = blockDim.z * blockIdx.z + threadIdx.z; z < desc_out.shape[3]; z += gridDim.z * blockDim.z) {
    //                 int ans_c = blockDim.x * blockIdx.x + threadIdx.x;
    //                 int ans_h = blockDim.y * blockIdx.y + threadIdx.y;
    //                 int ans_w = blockDim.z * blockIdx.z + threadIdx.z;
    //                 out[idx] = (in[idx] > static_cast<T>(0)) ? in[idx] : static_cast<T>(0);
    //             }
    //         }
    //     }
    // }
}



void conv2d_forward_shape_infer(
    const TensorDesc& in,
    const TensorDesc& kernel,
    const Conv2dDesc& desc_conv,
    TensorDesc& out
){

}