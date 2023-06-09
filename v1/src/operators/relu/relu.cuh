#pragma once

template <typename T, int insize_x, int insize_y, int insize_z>
__global__ void relu_forward(const T* in, T* out) {
    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < insize_x; x += gridDim.x * blockDim.x) {
        for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < insize_y; y += gridDim.y * blockDim.y) {
            for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < insize_y; y += gridDim.y * blockDim.y) {
                int idx = insize_y * insize_z * x + insize_z * y + insize_z;
                out[idx] = (in[idx] > T(0)) ? in[idx] : T(0);
            }
        }
    }
}