#include"unistd.h"
#include<tuple>
#include<limits>
#include<functional>
#include<vector>
#include<gtest/gtest.h>
#include"conv2d.cuh"
#include"helper_cuda.h"

#include <cudnn.h>

#include"print_tensor.h"
#include"log.h"
#include"cudnn_error.cuh"


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


class test_conv2d_bias_active_float:
    public testing::TestWithParam<
        std::tuple<
            std::vector<int>,  // input shape
            std::function<float(const std::vector<int>&)>,  // input generator
            std::vector<int>,  // kernel shape
            std::function<float(const std::vector<int>&)>,  // kernel generator
            int, int,  // padding, stride
            std::function<float(const int&)>,  // bias generator
            ActivationMode,
            dim3,  // grid
            dim3  //block
        >
    >
{
  public:
    std::vector<int> in_shape;
    std::function<float(const std::vector<int>&)> in_gen;
    std::vector<int> kernel_shape;
    std::function<float(const std::vector<int>&)> kernel_gen;
    int padding, stride;
    std::function<float(const int&)> bias_gen;
    ActivationMode act_mode;
    dim3 grid, block;


    R(cudnnHandle_t handle_;)
    R(cudnnConvolutionFwdAlgo_t algo;)
    R(cudnnTensorFormat_t filterFormat;)

    TensorDesc *input_desc = nullptr;
    R(cudnnTensorDescriptor_t cudnnIdesc;)
    float* input = nullptr;

    TensorDesc *kernel_desc = nullptr;
    R(cudnnFilterDescriptor_t cudnnFdesc;)
    float* kernel = nullptr;

    Conv2dDesc *conv_desc = nullptr;
    R(cudnnConvolutionDescriptor_t cudnnConvDesc;)

    TensorDesc *bias_desc = nullptr;
    R(cudnnTensorDescriptor_t cudnnBiasDesc;)
    float* bias = nullptr;


    R(cudnnActivationDescriptor_t cudnnActDesc;)
    

    TensorDesc *output_desc = nullptr;
    int outputDimA[4];
    int outinputStrideA[4];
    R(cudnnTensorDescriptor_t cudnnOdesc;
    float* cudnn_output = nullptr;)
    float* output = nullptr;


    test_conv2d_bias_active_float();
    ~test_conv2d_bias_active_float();
};

test_conv2d_bias_active_float::test_conv2d_bias_active_float(){
    std::tie(in_shape, in_gen, kernel_shape, kernel_gen, padding, stride, bias_gen, act_mode, grid, block) = GetParam();
    
    R(checkCudnnErr(cudnnCreate(&handle_));)
    R(algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;)
    R(filterFormat = CUDNN_TENSOR_NCHW;)


    checkCudaErrors(cudaMallocManaged((void**)&input_desc, sizeof(TensorDesc)));
    input_desc->init("nchw", in_shape);
    input_desc = new TensorDesc("nchw", in_shape);

    R(checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnIdesc));)
    R(int inputStrideA[4]; generateStrides(in_shape.data(), inputStrideA, 4, filterFormat);)
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

    kernel_desc = new TensorDesc("oihw", kernel_shape);

    R(checkCudnnErr(cudnnCreateFilterDescriptor(&cudnnFdesc));)
    R(checkCudnnErr(cudnnSetFilterNdDescriptor(cudnnFdesc, CUDNN_DATA_FLOAT, filterFormat, 4, kernel_shape.data()));)

    cudaMallocManaged((void**)&kernel, kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] * sizeof(float));
    for (int n = 0; n < kernel_shape[0]; n++) {
        for (int c = 0; c < kernel_shape[1]; c++) {
            for (int h = 0; h < kernel_shape[2]; h++) {
                for (int w = 0; w < kernel_shape[3]; w++) {
                    kernel[n * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] + c * kernel_shape[2] * kernel_shape[3] + h  * kernel_shape[3] + w] = kernel_gen({n, c, h, w});
                }
            }
        }
    }
    D(PrintTensor(kernel, kernel_desc->shape, *kernel_desc->dim_n, "kernel");)


    checkCudaErrors(cudaMallocManaged((void**)&conv_desc, sizeof(Conv2dDesc)));
    conv_desc->padding = padding;
    conv_desc->stride = stride;

    R(checkCudnnErr(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));)
    int convPadA[2] = {padding, padding};
    int convStrideA[2] = {stride, stride};
    int convDilationA[2] = {1, 1};
    R(checkCudnnErr(cudnnSetConvolutionNdDescriptor(cudnnConvDesc, 2, convPadA, convStrideA, convDilationA, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));)


    
    bias_desc = new TensorDesc("111o", {1, 1, 1, kernel_shape[0]});
    R(checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnBiasDesc));)
    R(checkCudnnErr(cudnnSetTensor4dDescriptor(cudnnBiasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, kernel_shape[0], 1, 1));)


    cudaMallocManaged((void**)&bias, kernel_shape[0] * sizeof(float));
    for (int i = 0; i < kernel_shape[0]; i++) {
        bias[i] = bias_gen(i);
    }
    D(PrintTensor(bias, bias_desc->shape, *bias_desc->dim_n, "bias");)

    
    R(checkCudnnErr(cudnnCreateActivationDescriptor(&cudnnActDesc));)
    R(checkCudnnErr(cudnnSetActivationDescriptor(cudnnActDesc, static_cast<cudnnActivationMode_t>(act_mode), CUDNN_NOT_PROPAGATE_NAN, 100.f));)


    output_desc = conv2d_forward_shape_infer(input_desc, kernel_desc, conv_desc);

    R(checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnOdesc));)
    outputDimA[0] = in_shape[0];
    outputDimA[1] = kernel_shape[0];
    outputDimA[2] = getFwdConvOutputDim(in_shape[2], convPadA[0], kernel_shape[2], convStrideA[0], convDilationA[0]);
    outputDimA[3] = getFwdConvOutputDim(in_shape[3], convPadA[1], kernel_shape[3], convStrideA[1], convDilationA[1]);
    R(generateStrides(outputDimA, outinputStrideA, 4, CUDNN_TENSOR_NCHW);)
    R(checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnOdesc, CUDNN_DATA_FLOAT, 4, outputDimA, outinputStrideA));)
    
    checkCudaErrors(cudaMallocManaged((void**)&output, output_desc->shape[0] * output_desc->shape[1] * output_desc->shape[2] * output_desc->shape[3] * sizeof(float)));
    R(checkCudaErrors(cudaMallocManaged((void**)&cudnn_output, outputDimA[0] * outputDimA[1] * outputDimA[2] * outputDimA[3] * sizeof(float)));)
};

test_conv2d_bias_active_float::~test_conv2d_bias_active_float(){
    delete input_desc;
    R(checkCudnnErr(cudnnDestroyTensorDescriptor(cudnnIdesc));)
    checkCudaErrors(cudaFree(input));

    delete kernel_desc;
    R(checkCudnnErr(cudnnDestroyFilterDescriptor(cudnnFdesc));)
    checkCudaErrors(cudaFree(kernel));

    checkCudaErrors(cudaFree(conv_desc));
    R(checkCudnnErr(cudnnDestroyConvolutionDescriptor(cudnnConvDesc));)

    delete bias_desc;
    R(checkCudnnErr(cudnnDestroyTensorDescriptor(cudnnBiasDesc));)
    checkCudaErrors(cudaFree(bias));

    R(checkCudnnErr(cudnnDestroyActivationDescriptor(cudnnActDesc));)

    delete output_desc;
    R(checkCudnnErr(cudnnDestroyTensorDescriptor(cudnnOdesc));)
    checkCudaErrors(cudaFree(output));
    R(checkCudaErrors(cudaFree(cudnn_output));)

    R(cudnnDestroy(handle_);)
}


INSTANTIATE_TEST_SUITE_P(
    general,
    test_conv2d_bias_active_float,
    testing::Values(
        // ----------------------------------- padding ------------------------------------------------
        std::make_tuple(
            std::vector<int>{1, 9216, 1, 1},
            [](const std::vector<int>& i){return sqrt(i[0] + i[1] + i[2] + i[3]);},
            std::vector<int>{4096, 9216, 1, 1},
            [](const std::vector<int>& i){return sqrt((i[0] + i[1] + i[2] + i[3]) % 5);},
            0, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(4, 1, 1),
            dim3(1024, 1, 1)
        ),
        std::make_tuple(
            std::vector<int>{1, 256, 6, 6},
            [](const std::vector<int>& i){return sqrt(i[0] + i[1] + i[2] + i[3]);},
            std::vector<int>{4096, 256, 6, 6},
            [](const std::vector<int>& i){return sqrt((i[0] + i[1] + i[2] + i[3]) % 5);},
            0, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(4, 1, 1),
            dim3(1024, 1, 1)
        ),
        std::make_tuple(
            std::vector<int>{1, 1, 3, 3},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{1, 1, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]) % 5;},
            0, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]) % 5;},
            0, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            1, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        // -------------------------------- stride --------------------------------------------------
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 2,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 3,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 4,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1),
            dim3(1)
        ),
        // ----------------------------------- grid block -------------------------------------------------------
        std::make_tuple(
            std::vector<int>{1, 3, 8, 8},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{1, 3, 2, 2},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 2,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(2, 2, 2),
            dim3(2, 2, 2)
        ),
        std::make_tuple(
            std::vector<int>{3, 1, 244, 244},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{1, 1, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 2,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1, 4, 4),
            dim3(1, 4, 4)
        ),
        std::make_tuple(  // cudnn wrong?
            std::vector<int>{1, 4, 2, 2},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{4, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1, 1, 1),
            dim3(1, 1, 1)
        ),
        std::make_tuple(
            std::vector<int>{1, 4, 2, 2},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{4, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 1,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(1, 1, 1),
            dim3(1, 1, 1)
        ),
        std::make_tuple(
            std::vector<int>{3, 4, 7, 7},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{4, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 2,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(2, 4, 4),
            dim3(2, 4, 4)
        ),
        std::make_tuple(
            std::vector<int>{3, 4, 32, 32},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{6, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 2,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(2, 4, 4),
            dim3(2, 4, 4)
        ),
        std::make_tuple(
            std::vector<int>{16, 3, 244, 244},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]) % 7;},
            std::vector<int>{64, 3, 11, 11},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]) % 5;},
            2, 4,
            [](const int& i){return 10 + i * 1.1f;},
            RELU,
            dim3(8, 8, 8),
            dim3(8, 8, 8)
        )
    )
);

TEST_P(test_conv2d_bias_active_float, check_output_vs_cudnn){
    float alpha = 1.f, beta = 0.f;
    size_t workSpaceSize = 0;
    R(checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc, algo, &workSpaceSize));)
    void* workSpace = 0;
    if (workSpaceSize > 0) checkCudaErrors(cudaMalloc(&workSpace, workSpaceSize));

    R(checkCudnnErr(
        cudnnConvolutionBiasActivationForward(
            handle_,
            (void*)(&alpha),
            cudnnIdesc,
            input,
            cudnnFdesc,
            kernel,
            cudnnConvDesc,
            algo,
            workSpace,
            workSpaceSize,
            (void*)(&beta),
            cudnnOdesc,
            cudnn_output,
            cudnnBiasDesc,
            bias,
            cudnnActDesc,
            cudnnOdesc,
            cudnn_output
        )
    );)
    R(checkCudaErrors(cudaDeviceSynchronize());)
    R(PrintTensor(cudnn_output, outputDimA, 4, "cudnn_output");)

    // size_t sharedmem_size =\
    //     input_desc->shape[0] *\
    //     input_desc->shape[1] *\
    //     (conv_desc->stride * block.y + kernel_desc->shape[2] - 2 * conv_desc->padding - 1) *\
    //     (conv_desc->stride * block.z + kernel_desc->shape[3] - 2 * conv_desc->padding - 1);

    conv2d_bias_active_forward_naive<<<grid, block/*, sharedmem_size * sizeof(float)*/>>>(
        input,
        input_desc,
        kernel,
        kernel_desc,
        conv_desc,
        bias,
        bias_desc,
        RELU,
        output,
        output_desc,
        true
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