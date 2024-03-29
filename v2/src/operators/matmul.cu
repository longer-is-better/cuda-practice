#include "tensor.h"
#include "operator.cuh"
#include "matmul.cuh"
#include "kmatmul.cuh"


MatMul::MatMul(
    Tensor* A,
    Tensor* B
):
    Operator({A, B}, {new Tensor()})
{
    ;
}

Operator *MatMul::copy()
{
    return new MatMul();
}

void MatMul::set_cudastream(cudaStream_t cudastream) {
    _cudastream = cudastream;
}

void MatMul::infer_shape() {
    CHECK_EQ(_input_tensors.size(), 2);
    CHECK_EQ(_input_tensors[0]->_dim_n, 2);
    CHECK_EQ(_input_tensors[1]->_dim_n, 2);
    CHECK_EQ(_input_tensors[0]->_shape[1], _input_tensors[1]->_shape[0]);
    _output_tensors[0]->set_shape({_input_tensors[0]->_shape[0], _input_tensors[1]->_shape[1]});
}


void MatMul::forward() {
    dim3 BLOCK(16, 16);
    dim3 GRID(
        (_output_tensors[0]->_shape[1] + BLOCK.x - 1) / BLOCK.x,
        (_output_tensors[0]->_shape[0] + BLOCK.y - 1) / BLOCK.y
    );
    size_t shared_mem = BLOCK.x * BLOCK.y * BLOCK.z * sizeof(float) * 2;
    kmatmul<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        false,
        false,
        _input_tensors[0]->_shape[0],
        _input_tensors[0]->_shape[1],
        _input_tensors[1]->_shape[1],
        _input_tensors[0]->_p_data,
        _input_tensors[1]->_p_data,
        _output_tensors[0]->_p_data
    );
    checkCudaErrors(cudaDeviceSynchronize());
}


void MatMul::backward() {
    dim3 BLOCK;
    dim3 GRID;
    size_t shared_mem;

    BLOCK = dim3(16, 16);
    GRID = dim3(
        (_input_tensors[0]->_shape[1] + BLOCK.x - 1) / BLOCK.x,
        (_input_tensors[0]->_shape[0] + BLOCK.y - 1) / BLOCK.y
    );
    shared_mem = BLOCK.x * BLOCK.y * BLOCK.z * sizeof(float) * 2;
    kmatmul<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        false,
        true,
        _input_tensors[0]->_shape[0],
        _input_tensors[1]->_shape[1],
        _input_tensors[1]->_shape[0],
        _output_tensors[0]->_p_gradient,
        _input_tensors[1]->_p_data,
        _input_tensors[0]->_p_gradient
    );

    BLOCK = dim3(16, 16);
    GRID = dim3(
        (_input_tensors[1]->_shape[1] + BLOCK.x - 1) / BLOCK.x,
        (_input_tensors[1]->_shape[0] + BLOCK.y - 1) / BLOCK.y
    );
    shared_mem = BLOCK.x * BLOCK.y * BLOCK.z * sizeof(float) * 2;
    kmatmul<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        true,
        false,
        _input_tensors[1]->_shape[0],
        _input_tensors[0]->_shape[0],
        _input_tensors[1]->_shape[1],
        _input_tensors[0]->_p_data,
        _output_tensors[0]->_p_gradient,
        _input_tensors[1]->_p_gradient
    );
}