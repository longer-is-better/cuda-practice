#include<cudnn.h>

#include"gtest/gtest.h"

#include"print_tensor.h"
#include"max_pool2d.cuh"
#include"helper_cuda.h"
#include"cudnn_error.cuh"
#include"log.h"
#include"descriptor.h"

static inline int
getFwdConvPaddedImageDim(int tensorDim, int pad) {
    return tensorDim + (2 * pad);
}

static inline int
getFwdConvDilatedFilterDim(int filterDim, int dilation) {
    return ((filterDim - 1) * dilation) + 1;
}

static inline int
getFwdConvOutputDim(int tensorDim, int pad, int filterDim, int stride, int dilation) {
    int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return (p);
}

static void
generateStrides(const int* dimA, int* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW || filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
        strideA[nbDims - 1] = 1;
        for (int d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

class test_pool2d_float:
    public testing::TestWithParam<
        std::tuple<
            std::vector<int>,  // input shape
            std::function<float(const std::vector<int>&)>,  // input generator
            std::vector<int>,  // window shape
            std::vector<int>,  // padding
            std::vector<int>,  // stride
            dim3,  // grid
            dim3  // block
        >
    >
{
  public:
    std::vector<int> in_shape;
    std::function<float(const std::vector<int>&)> in_gen;
    std::vector<int> window_shape;
    std::vector<int> padding;
    std::vector<int> stride;
    dim3 grid, block;


    R(cudnnHandle_t handle_;)

    TensorDesc *input_desc = nullptr;
    R(cudnnTensorDescriptor_t cudnnIdesc;)
    float* input = nullptr;


    Pool2dDesc *pool_desc = nullptr;
    R(cudnnPoolingDescriptor_t cudnnPdesc;)

    TensorDesc *output_desc = nullptr;
    int outputDimA[4];
    int outinputStrideA[4];
    R(cudnnTensorDescriptor_t cudnnOdesc;
    float* cudnn_output = nullptr;)
    float* output = nullptr;


    test_pool2d_float();
    ~test_pool2d_float();
};

test_pool2d_float::test_pool2d_float(){
    std::tie(
        in_shape,
        in_gen,
        window_shape,
        padding,
        stride,
        grid,
        block
    ) = GetParam();
    
    R(checkCudnnErr(cudnnCreate(&handle_));)

    input_desc = new TensorDesc("nchw", in_shape);

    R(checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnIdesc));)
    R(int inputStrideA[4]; generateStrides(in_shape.data(), inputStrideA, 4, CUDNN_TENSOR_NCHW);)
    R(checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnIdesc, CUDNN_DATA_FLOAT, 4, in_shape.data(), inputStrideA));)

    checkCudaErrors(cudaMallocManaged((void**)&input, in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] * sizeof(float)));
    for (int n = 0; n < in_shape[0]; n++) {
        for (int c = 0; c < in_shape[1]; c++) {
            for (int h = 0; h < in_shape[2]; h++) {
                for (int w = 0; w < in_shape[3]; w++) {
                    input[n * in_shape[1] * in_shape[2] * in_shape[3] + c * in_shape[2] * in_shape[3] + h  * in_shape[3] + w] = in_gen({n, c, h, w});
                }
            }
        }
    }
    D(PrintTensor(input, input_desc->shape, *input_desc->dim_n, "input");)

    pool_desc = new Pool2dDesc(window_shape, padding, stride);

    R(cudnnCreatePoolingDescriptor(&cudnnPdesc);)
    R(
        checkCudnnErr(
            cudnnSetPooling2dDescriptor(
                cudnnPdesc,
                CUDNN_POOLING_MAX_DETERMINISTIC,
                // CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                CUDNN_PROPAGATE_NAN,
                window_shape[0],
                window_shape[1],
                padding[0],
                padding[1],
                stride[0],
                stride[1]
            )
        );
    )

    output_desc = pool2d_forward_shape_infer(input_desc, pool_desc);

    R(checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnOdesc));)
    R(checkCudnnErr(cudnnGetPooling2dForwardOutputDim(cudnnPdesc, cudnnIdesc, &outputDimA[0], &outputDimA[1], &outputDimA[2], &outputDimA[3]));)
    R(generateStrides(outputDimA, outinputStrideA, 4, CUDNN_TENSOR_NCHW);)
    R(checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnOdesc, CUDNN_DATA_FLOAT, 4, outputDimA, outinputStrideA));)
    
    checkCudaErrors(cudaMallocManaged((void**)&output, output_desc->shape[0] * output_desc->shape[1] * output_desc->shape[2] * output_desc->shape[3] * sizeof(float)));
    R(checkCudaErrors(cudaMallocManaged((void**)&cudnn_output, outputDimA[0] * outputDimA[1] * outputDimA[2] * outputDimA[3] * sizeof(float)));)
};

test_pool2d_float::~test_pool2d_float(){
    delete input_desc;
    R(checkCudnnErr(cudnnDestroyTensorDescriptor(cudnnIdesc));)
    checkCudaErrors(cudaFree(input));

    delete pool_desc;
    R(checkCudnnErr(cudnnDestroyPoolingDescriptor(cudnnPdesc));)

    delete output_desc;
    R(checkCudnnErr(cudnnDestroyTensorDescriptor(cudnnOdesc));)
    checkCudaErrors(cudaFree(output));
    R(checkCudaErrors(cudaFree(cudnn_output));)

    R(cudnnDestroy(handle_);)
}


INSTANTIATE_TEST_SUITE_P(
    general,
    test_pool2d_float,
    testing::Values(
        std::make_tuple(
            std::vector<int>{1, 3, 4, 4},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 2},
            std::vector<int>{0, 0},
            std::vector<int>{1, 1},
            dim3(1, 2, 2),
            dim3(1, 2, 2)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 255, 255},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{1, 1},
            std::vector<int>{10, 10},
            std::vector<int>{3, 2},
            dim3(4, 8, 8),
            dim3(3, 4, 4)
        )
    )
);


TEST_P(test_pool2d_float, check_output_vs_cudnn){
    float alpha = 1.f, beta = 0.f;
    R(
        checkCudnnErr(
            cudnnPoolingForward(
                handle_,
                cudnnPdesc,
                &alpha,
                cudnnIdesc,
                input,
                &beta,
                cudnnOdesc,
                cudnn_output
            )
        );
    )
    R(checkCudaErrors(cudaDeviceSynchronize());)
    R(PrintTensor(cudnn_output, outputDimA, 4, "cudnn_output");)

    max_pool2d<<<grid, block/*, sharedmem_size * sizeof(float)*/>>>(
        input,
        input_desc,
        pool_desc,
        output,
        output_desc
    );
    checkCudaErrors(cudaDeviceSynchronize());
    PrintTensor(output, output_desc->shape, *output_desc->dim_n, "output");

    size_t len = 1;
    for (int i = 0; i < 4; i++) {
        len *= outputDimA[i];
        ASSERT_EQ(outputDimA[i], output_desc->shape[i]) << "i: " << i << std::endl;
    }
    R(for (int n = 0; n < output_desc->shape[0]; n++) {
        for (int c = 0; c < output_desc->shape[1]; c++) {
            for (int h = 0; h < output_desc->shape[2]; h++) {
                for (int w = 0; w < output_desc->shape[3]; w++) {
                    ASSERT_FLOAT_EQ(
                        output[n * output_desc->stride[0] + c * output_desc->stride[1] + h * output_desc->stride[2] + w * output_desc->stride[3]],
                        cudnn_output[n * outinputStrideA[0] + c * outinputStrideA[1] + h * outinputStrideA[2] + w * outinputStrideA[3]]
                    ) << "n" << n << ", c" << c << ", h" << h << ", w" << w;
                }
            }
        }
    })
}