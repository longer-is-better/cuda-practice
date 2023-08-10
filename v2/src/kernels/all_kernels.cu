#include "kelementwise.cuh"
// #include "kpool2d.cuh"
#include "kothers.cuh"
#include "krelu.cuh"
#include "kreduce.cuh"
#include "kmatmul.cuh"
#include "kmap.cuh"




template <>
__global__ void kelementwise(size_t N, float *I1, float alpha, float *I2, float *O, ELE_OP op);

template <>
__global__ void kelementwise_inplace(size_t N, float *IO, float alpha, float *operand, ELE_OP op);

template <>
__global__ void kinitializeRandom(float* data, int size, float lower_bound, float upper_bound);

template <>
__global__ void krelu_forward(float* in, float* out, int insize_x, int insize_y, int insize_z);

template <>
__global__ void kreduce(size_t n, float *I, float *O, REDUCE_OP op);

template <>
__global__ void kmatmul(bool trans_W, bool trans_X, size_t m, size_t k, size_t n, float *W, float *X, float *Y);

template <>
__global__ void kmap(size_t N, float *I, float *O, MAP_OP op, float operand);

template <>
__global__ void kmap_inplace(size_t N, float *I, MAP_OP op, float operand);

template <>
__global__ void kmemset(size_t N, float *I, float val);

template <>
__global__ void kmemset_d(size_t N, float *I, float alpha, float *val);