#include"relu.cuh"

// __global__ void relu_forward(float* in, float* out, int insize_x, int insize_y, int insize_z) {
//     for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < insize_x; x += gridDim.x * blockDim.x) {
//         for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < insize_y; y += gridDim.y * blockDim.y) {
//             for (int z = blockDim.z * blockIdx.z + threadIdx.z; z < insize_z; z += gridDim.z * blockDim.z) {
//                 int idx = insize_y * insize_z * x + insize_z * y + z;
//                 out[idx] = (in[idx] > static_cast<float>(0)) ? in[idx] : static_cast<float>(0);
//             }
//         }
//     }
// }
