#pragma once

#include"print_tensor.h"
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


// __constant__ T kernel_const[64*1024/sizeof(T)];
extern __shared__ float block_in[];
template <typename T>
__global__ void conv2d_forward_memopt(
    const T* in, const TensorDesc* desc_in,
    const T* kernel, const TensorDesc* desc_kernel, const Conv2dDesc* desc_conv,
    T* out, const TensorDesc* desc_out
){
    for (int n = 0; n < desc_out->shape[0]; n++) {
        for (int block_in_c = blockDim.x * blockIdx.x + threadIdx.x; block_in_c < desc_in->shape[1]; block_in_c += gridDim.x * blockDim.x) {
            for (int ans_h = blockDim.y * blockIdx.y + threadIdx.y; ans_h < desc_out->shape[2]; ans_h += gridDim.y * blockDim.y) {
                for (int ans_w = blockDim.z * blockIdx.z + threadIdx.z; ans_w < desc_out->shape[3]; ans_w += gridDim.z * blockDim.z) {
                    // copy ans element's in tile
                    for (int in_h = desc_conv->stride * ans_h - desc_conv->padding; in_h < desc_conv->stride * (ans_h + 1) - desc_conv->padding; in_h++) {
                        for (int in_w = desc_conv->stride * ans_w - desc_conv->padding; in_w < desc_conv->stride * (ans_w + 1) - desc_conv->padding; in_w++) {
                            int block_idx = n * desc_in->shape[1] * blockDim.y * blockDim.z +\
                                            block_in_c * blockDim.y * blockDim.z +\
                                            (in_h - blockDim.y * blockIdx.y) * blockDim.z +\
                                            (in_w - blockDim.z * blockIdx.z);
                            int in_idx = n * desc_in->shape[1] * desc_in->shape[2] * desc_in->shape[3] +\
                                         block_in_c * desc_in->shape[2] * desc_in->shape[3] +\
                                         in_h * desc_in->shape[3] +\
                                         in_w;
                            block_in[block_idx] = in[in_idx];
                            printf("(%d %d %d %d), %f\n", n, block_in_c, in_h, in_w, block_in[block_idx]);
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

// __global__ void conv2d_forward_memopt(
//     T* in, const TensorDesc& desc_in,
//     T* kernel, const TensorDesc& desc_kernel,
//     const Conv2dDesc& desc_conv,
//     T* out, const TensorDesc& desc_out
// ){
//     for (int n = 0; n < desc_out.shape[0]; n++) {
//         for (int block_in_c = blockDim.x * blockIdx.x + threadIdx.x; block_in_c < desc_in->shape[1]; block_in_c += gridDim.x * blockDim.x) {
//             for (int ans_h = blockDim.y * blockIdx.y + threadIdx.y; ans_h < desc_out.shape[2]; ans_h += gridDim.y * blockDim.y) {
//                 for (int ans_w = blockDim.z * blockIdx.z + threadIdx.z; ans_w < desc_out.shape[3]; ans_w += gridDim.z * blockDim.z) {
//                     // copy ans element's in tile
//                     for (int in_h = desc_conv.stride * ans_h - desc_conv.padding; in_h < desc_conv.stride * (ans_h + 1) - desc_conv.padding; in_h++) {
//                         for (int in_w = desc_conv.stride * ans_w - desc_conv.padding; in_w < desc_conv.stride * (ans_w + 1) - desc_conv.padding; in_w++) {
//                             int block_idx = n * desc_in->shape[1] * blockDim.y * blockDim.z +\
//                                             block_in_c * blockDim.y * blockDim.z +\
//                                             (in_h - blockDim.y * blockIdx.y) * blockDim.z +\
//                                             (in_w - blockDim.z * blockIdx.z);
//                             int in_idx = n * desc_in->shape[1] * desc_in->shape[2] * desc_in->shape[3] +\
//                                          block_in_c * desc_in->shape[2] * desc_in->shape[3] +\
//                                          in_h * desc_in->shape[3] +\
//                                          in_w;
//                             block_in[block_idx] = in[in_idx];
//                             __syncthreads();
//                             // ...
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// template <typename T>
__global__ void step (
    float* in, const TensorDesc& desc_in
) {
    printf("block(%d, %d, %d), thread(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}


bool conv2d_forward_shape_infer(
    const TensorDesc* in,
    const TensorDesc* kernel,
    const Conv2dDesc* conv,
    TensorDesc* out
){
    // TensorDesc* ans = new TensorDesc(
    //     "nchw",
    //     {
    //         in.shape[0],
    //         kernel.shape[0],
    //         (in.shape[2] + 2 * conv.padding - kernel.shape[2]) / conv.stride,
    //         (in.shape[3] + 2 * conv.padding - kernel.shape[3]) / conv.stride
    //     }
    // );
    // return *ans;
    if (!in || !kernel || !conv || !out) {
        return false;
    } else {
        out->init(
            "nchw",
            {
                in->shape[0],
                kernel->shape[0],
                (in->shape[2] + 2 * conv->padding - kernel->shape[2]) / conv->stride,
                (in->shape[3] + 2 * conv->padding - kernel->shape[3]) / conv->stride
            }
        );
        return true;
    }
}