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

// Generate uniform numbers [0,1)
static void
initImage(float* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10;  // 2^-32
    }
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

TEST(cudnn, smoke) {
    cudnnHandle_t handle_;
    checkCudnnErr(cudnnCreate(&handle_));
}

TEST(cudnnConvolutionForward, smoke) {
    cudnnHandle_t handle_;
    checkCudnnErr(cudnnCreate(&handle_));


    float alpha = 1.f, beta = 0.f;


    cudnnTensorDescriptor_t cudnnIdesc;
    checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnIdesc));
    cudnnTensorFormat_t filterFormat = CUDNN_TENSOR_NCHW;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    int inputDimA[4] = {1, 3, 5, 5};
    int inputStrideA[4]; generateStrides(inputDimA, inputStrideA, 4, filterFormat);
    checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, 4, inputDimA, inputStrideA));

    size_t inputNum = inputDimA[0] * inputDimA[1] * inputDimA[2] * inputDimA[3];
    std::vector<float> hostInput(inputNum);
    for (int n = 0; n < inputDimA[0]; n++) {
        for (int c = 0; c < inputDimA[1]; c++) {
            for (int h = 0; h < inputDimA[2]; h++) {
                for (int w = 0; w < inputDimA[3]; w++) {
                    hostInput[n * inputDimA[1] * inputDimA[2] * inputDimA[3] + c * inputDimA[2] * inputDimA[3] + h  * inputDimA[3] + w] = n + c + h + w;
                }
            }
        }
    }
    PrintTensor(hostInput.data(), inputDimA, 4, "hostInput");
    float* deviceInput;
    checkCudaErrors(cudaMalloc((void**)&deviceInput, inputNum * sizeof(float)));
    checkCudaErrors(cudaMemcpy(deviceInput, hostInput.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice));



    cudnnFilterDescriptor_t cudnnFdesc;
    checkCudnnErr(cudnnCreateFilterDescriptor(&cudnnFdesc));
    int filterDimA[4] = {1, 3, 2, 2};
    checkCudnnErr(cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, CUDNN_TENSOR_NCHW, 4, filterDimA));


    std::vector<float> hostFilter{1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0};
    // std::vector<float> hostFilter(filterDimA[0] * filterDimA[1] * filterDimA[2] * filterDimA[3]);
    // for (int n = 0; n < filterDimA[0]; n++) {
    //     for (int c = 0; c < filterDimA[1]; c++) {
    //         for (int h = 0; h < filterDimA[2]; h++) {
    //             for (int w = 0; w < filterDimA[3]; w++) {
    //                 hostFilter[n * filterDimA[1] * filterDimA[2] * filterDimA[3] + c * filterDimA[2] * filterDimA[3] + h  * filterDimA[3] + w] = n + c + h + w;
    //             }
    //         }
    //     }
    // }
    PrintTensor(hostFilter.data(), filterDimA, 4, "hostFilter");
    float* deviceFilter;
    checkCudaErrors(cudaMalloc((void**)&deviceFilter, hostFilter.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpy(deviceFilter, hostFilter.data(), filterDimA[0] * filterDimA[1] * filterDimA[2] * filterDimA[3] * sizeof(float), cudaMemcpyHostToDevice));


    cudnnConvolutionDescriptor_t cudnnConvDesc;
    checkCudnnErr(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
    int convPadA[2] = {0, 0};
    int convStrideA[2] = {1, 1};
    int convDilationA[2] = {1, 1};
    checkCudnnErr(cudnnSetConvolutionNdDescriptor(cudnnConvDesc, 2, convPadA, convStrideA, convDilationA, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));


    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;


    cudnnTensorDescriptor_t cudnnOdesc;
    checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnOdesc));
    int outputDimA[4] = {
        inputDimA[0],
        filterDimA[0],
        getFwdConvOutputDim(inputDimA[2], convPadA[0], filterDimA[2], convStrideA[0], convDilationA[0]),
        getFwdConvOutputDim(inputDimA[3], convPadA[1], filterDimA[3], convStrideA[1], convDilationA[1])
    };
    int outinputStrideA[4];
    generateStrides(outputDimA, outinputStrideA, 4, CUDNN_TENSOR_NCHW);
    checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, 4, outputDimA, outinputStrideA));

    float* deviceOutput;
    size_t outputNum = outputDimA[0] * outputDimA[1] * outputDimA[2] * outputDimA[3];
    checkCudaErrors(cudaMalloc((void**)&deviceOutput, outputNum * sizeof(float)));


    size_t workSpaceSize = 0;
    checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc, algo, &workSpaceSize));


    void* workSpace = 0;
    if (workSpaceSize > 0) checkCudaErrors(cudaMalloc(&workSpace, workSpaceSize));


    checkCudnnErr(cudnnConvolutionForward(handle_,
                                          (void*)(&alpha),
                                          cudnnIdesc,
                                          deviceInput,
                                          cudnnFdesc,
                                          deviceFilter,
                                          cudnnConvDesc,
                                          algo,
                                          workSpace,
                                          workSpaceSize,
                                          (void*)(&beta),
                                          cudnnOdesc,
                                          deviceOutput));
    checkCudaErrors(cudaDeviceSynchronize());

    float* fetchOutput = static_cast<float*>(malloc(outputNum * sizeof(float)));
    ASSERT_TRUE(fetchOutput) << "fetchOutput malloc failed";
    checkCudaErrors(cudaMemcpy(fetchOutput, deviceOutput, outputNum * sizeof(float), cudaMemcpyDeviceToHost));

    PrintTensor(fetchOutput, outputDimA, 4, "fetchOutput");


    cudaFree(deviceInput);
    cudaFree(deviceFilter);
    cudaFree(deviceOutput);
    cudaFree(workSpace);
    free(fetchOutput);
    
}


// __device__ __constant__ float kernel_const[64*1024/sizeof(float)];
TEST(conv2d_forward_naive, smoke){
    TensorDesc *input_desc = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&input_desc, sizeof(TensorDesc)));
    input_desc->init("nchw", {1, 2, 7, 7});

    float *input_data = nullptr;
    int input_len = 1; for (int n = 0; n < *input_desc->dim_n; n++) input_len *= input_desc->shape[n];
    checkCudaErrors(cudaMallocManaged((void**)&input_data, input_len * sizeof(float)));
    for (int n = 0; n < input_desc->shape[0]; n++) {
        for (int c = 0; c < input_desc->shape[1]; c++) {
            for (int h = 0; h < input_desc->shape[2]; h++) {
                for (int w = 0; w < input_desc->shape[3]; w++) {
                    input_data[n * input_desc->shape[1] * input_desc->shape[2] * input_desc->shape[3] + c * input_desc->shape[2] * input_desc->shape[3] + h  * input_desc->shape[3] + w] = n + c + h + w;
                }
            }
        }
    }
    PrintTensor(input_data, input_desc->shape, 4, "input_data");


    TensorDesc *kernel_desc = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&kernel_desc, sizeof(TensorDesc)));
    kernel_desc->init("nchw", {1, 2, 2, 2});

    std::vector<float> kernel_vec{1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0};
    float* kernel_data = nullptr;
    int kernel_len = 1; for (int n = 0; n < *kernel_desc->dim_n; n++) kernel_len *= kernel_desc->shape[n];
    checkCudaErrors(cudaMallocManaged((void**)&kernel_data, kernel_len * sizeof(float)));
    memcpy(kernel_data, kernel_vec.data(), kernel_len * sizeof(float));
    PrintTensor(kernel_data, kernel_desc->shape, *kernel_desc->dim_n, "kernel_data");
    // checkCudaErrors(cudaMemcpyToSymbol(kernel_const, kernel_vec.data(), kernel_len * sizeof(float)));
    // PrintTensor(kernel_const, kernel_desc->shape, *kernel_desc->dim_n, "kernel_const");


    Conv2dDesc *conv_desc = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&conv_desc, sizeof(Conv2dDesc)));
    conv_desc->padding = 0;
    conv_desc->stride = 1;

    TensorDesc *output_desc = conv2d_forward_shape_infer(input_desc, kernel_desc, conv_desc);
    // checkCudaErrors(cudaMallocManaged((void**)&output_desc, sizeof(TensorDesc)));
    // ASSERT_TRUE(conv2d_forward_shape_infer(input_desc, kernel_desc, conv_desc, output_desc));
    float* output_data = nullptr;
    int output_len = 1; for (int n = 0; n < *output_desc->dim_n; n++) input_len *= output_desc->shape[n];
    checkCudaErrors(cudaMallocManaged((void**)&output_data, output_len * sizeof(float)));


    dim3 grid(1, 1, 2), block(1, 1, 3);
    size_t sharedmem_size =\
        input_desc->shape[0] *\
        input_desc->shape[1] *\
        (conv_desc->stride * block.y + kernel_desc->shape[2] - 2 * conv_desc->padding - 1) *\
        (conv_desc->stride * block.z + kernel_desc->shape[3] - 2 * conv_desc->padding - 1);

    // conv2d_forward_naive<<<grid, block, sharedmem_size * sizeof(float)>>>(input_data, input_desc, kernel_const, kernel_desc, conv_desc, output_data, output_desc);
    conv2d_forward_naive<<<grid, block, sharedmem_size * sizeof(float)>>>(input_data, input_desc, kernel_data, kernel_desc, conv_desc, output_data, output_desc);
    // step<<<grid, block, sharedmem_size * sizeof(float)>>>(input_data, input_desc);
    checkCudaErrors(cudaDeviceSynchronize());
    PrintTensor(output_data, output_desc->shape, *output_desc->dim_n, "output");
}

class test_conv2d_float:
    public testing::TestWithParam<
        std::tuple<
            std::vector<int>,  // input shape
            std::function<float(const std::vector<int>&)>,  // input generator
            std::vector<int>,  // kernel shape
            std::function<float(const std::vector<int>&)>,  // kernel generator
            int, int,  // padding, stride
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

    TensorDesc *output_desc = nullptr;
    int outputDimA[4];
    int outinputStrideA[4];
    R(cudnnTensorDescriptor_t cudnnOdesc;
    float* cudnn_output = nullptr;)
    float* output = nullptr;


    test_conv2d_float();
    ~test_conv2d_float();
};

test_conv2d_float::test_conv2d_float(){
    std::tie(in_shape, in_gen, kernel_shape, kernel_gen, padding, stride, grid, block) = GetParam();
    
    R(checkCudnnErr(cudnnCreate(&handle_));
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    filterFormat = CUDNN_TENSOR_NCHW;)


    checkCudaErrors(cudaMallocManaged((void**)&input_desc, sizeof(TensorDesc)));
    input_desc->init("nchw", in_shape);
    input_desc = new TensorDesc("nchw", in_shape);

    R(checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnIdesc));
    int inputStrideA[4]; generateStrides(in_shape.data(), inputStrideA, 4, filterFormat);
    checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnIdesc, CUDNN_DATA_FLOAT, 4, in_shape.data(), inputStrideA));)

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

    R(cudnnCreateFilterDescriptor(&cudnnFdesc);
    checkCudnnErr(cudnnSetFilterNdDescriptor(cudnnFdesc, CUDNN_DATA_FLOAT, filterFormat, 4, kernel_shape.data()));)

    cudaMallocManaged((void**)&kernel, kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]);
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


    output_desc = conv2d_forward_shape_infer(input_desc, kernel_desc, conv_desc);

    R(checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnOdesc));
    outputDimA[0] = in_shape[0];
    outputDimA[1] = kernel_shape[0];
    outputDimA[2] = getFwdConvOutputDim(in_shape[2], convPadA[0], kernel_shape[2], convStrideA[0], convDilationA[0]);
    outputDimA[3] = getFwdConvOutputDim(in_shape[3], convPadA[1], kernel_shape[3], convStrideA[1], convDilationA[1]);
    R(generateStrides(outputDimA, outinputStrideA, 4, CUDNN_TENSOR_NCHW));
    R(checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnOdesc, CUDNN_DATA_FLOAT, 4, outputDimA, outinputStrideA));))
    
    checkCudaErrors(cudaMallocManaged((void**)&output, output_desc->shape[0] * output_desc->shape[1] * output_desc->shape[2] * output_desc->shape[3] * sizeof(float)));
    R(checkCudaErrors(cudaMallocManaged((void**)&cudnn_output, outputDimA[0] * outputDimA[1] * outputDimA[2] * outputDimA[3] * sizeof(float)));)
};

test_conv2d_float::~test_conv2d_float(){
    delete input_desc;
    R(checkCudnnErr(cudnnDestroyTensorDescriptor(cudnnIdesc));)
    checkCudaErrors(cudaFree(input));

    delete kernel_desc;
    R(checkCudnnErr(cudnnDestroyFilterDescriptor(cudnnFdesc));)
    checkCudaErrors(cudaFree(kernel));

    checkCudaErrors(cudaFree(conv_desc));
    R(checkCudnnErr(cudnnDestroyConvolutionDescriptor(cudnnConvDesc));)


    delete output_desc;
    R(checkCudnnErr(cudnnDestroyTensorDescriptor(cudnnOdesc));)
    checkCudaErrors(cudaFree(output));
    R(checkCudaErrors(cudaFree(cudnn_output));)

    R(cudnnDestroy(handle_);)
}


INSTANTIATE_TEST_SUITE_P(
    general,
    test_conv2d_float,
    testing::Values(
        // ----------------------------------- padding ------------------------------------------------
        std::make_tuple(
            std::vector<int>{1, 1, 3, 3},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{1, 1, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]) % 5;},
            0, 1,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]) % 5;},
            0, 1,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            1, 1,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 1,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 1,
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
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 2,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 3,
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            std::vector<int>{1, 3, 5, 5},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{2, 3, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 4,
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
            dim3(2, 2, 2),
            dim3(2, 2, 2)
        ),
        std::make_tuple(
            std::vector<int>{3, 1, 244, 244},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{1, 1, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 2,
            dim3(1, 4, 4),
            dim3(1, 4, 4)
        ),
        std::make_tuple(  // cudnn wrong?
            std::vector<int>{1, 4, 2, 2},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{4, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 1,
            dim3(1, 1, 1),
            dim3(1, 1, 1)
        ),
        std::make_tuple(
            std::vector<int>{1, 4, 2, 2},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{4, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            2, 1,
            dim3(1, 1, 1),
            dim3(1, 1, 1)
        ),
        std::make_tuple(
            std::vector<int>{3, 4, 7, 7},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{4, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 2,
            dim3(2, 4, 4),
            dim3(2, 4, 4)
        ),
        std::make_tuple(
            std::vector<int>{3, 4, 32, 32},
            [](const std::vector<int>& i){return i[0] + i[1] + i[2] + i[3];},
            std::vector<int>{6, 4, 3, 3},
            [](const std::vector<int>& i){return (i[0] + i[1] + i[2] + i[3]);},
            3, 2,
            dim3(2, 4, 4),
            dim3(2, 4, 4)
        )
    )
);

TEST_P(test_conv2d_float, check_output_vs_cudnn){
    float alpha = 1.f, beta = 0.f;
    size_t workSpaceSize = 0;
    R(checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc, algo, &workSpaceSize));)
    void* workSpace = 0;
    if (workSpaceSize > 0) checkCudaErrors(cudaMalloc(&workSpace, workSpaceSize));

    R(checkCudnnErr(
        cudnnConvolutionForward(
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

    conv2d_forward_naive<<<grid, block/*, sharedmem_size * sizeof(float)*/>>>(
        input,
        input_desc,
        kernel,
        kernel_desc,
        conv_desc,
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
                    ASSERT_EQ(
                        output[n * output_desc->stride[0] + c * output_desc->stride[1] + h * output_desc->stride[2] + w * output_desc->stride[3]],
                        cudnn_output[n * outinputStrideA[0] + c * outinputStrideA[1] + h * outinputStrideA[2] + w * outinputStrideA[3]]
                    ) << "n" << n << ", c" << c << ", h" << h << ", w" << w;
                }
            }
        }
    })
}