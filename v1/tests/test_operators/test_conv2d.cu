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
#include"cudnn_error.cuh"

// class test_conv2d_float:
//     public testing::TestWithParam<
//         std::tuple<
//             std::vector<int>,  // input shape
//             std::function<float(const std::vector<int>&)>,  // input generator
//             std::vector<int>,  // kernel shape
//             std::function<float(const std::vector<int>&)>,  // kernel generator


//             dim3,  // grid
//             dim3  //block
//         >
//     >
// {
//   public:
//     int len = 0;
//     std::function<float(int)> gen;
//     dim3 grid, block;

//     std::vector<float> input;
//     std::vector<float> host_output;
//     std::vector<float> fetch_output;

//     float* device_input;
//     float* device_output;


    // test_relu_float_1d_input();
    // ~test_relu_float_1d_input();
// };

// test_relu_float_1d_input::test_relu_float_1d_input(){
//     std::tie(len, gen, grid, block) = GetParam();

//     input.resize(len); for (int i = 0; i < len; i++) input[i] = gen(i);
//     host_output.resize(len); for (int i = 0; i < len; i++) host_output[i] = input[i] > 0 ? input[i] : 0;
//     fetch_output.resize(len);

//     checkCudaErrors(cudaMalloc((void**)&device_input, len * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&device_output, len * sizeof(float)));
// };

// test_relu_float_1d_input::~test_relu_float_1d_input(){
//     checkCudaErrors(cudaFree(device_input));
//     checkCudaErrors(cudaFree(device_output));
// }


// INSTANTIATE_TEST_SUITE_P(
//     general,
//     test_conv2d_float,
//     testing::Values(
//         std::make_tuple(
//             1,
//             [](int i){
//                 return 1;
//             },
//             dim3(1),
//             dim3(1)
//         ),
//         std::make_tuple(
//             10,
//             [](int i){
//                 return -5 + i;
//             },
//             dim3(1),
//             dim3(1)
//         ),
//         std::make_tuple(
//             128,
//             [](int i){
//                 return -0.5 + static_cast<float>(rand()) / RAND_MAX;
//             },
//             dim3(1),
//             dim3(128)
//         ),
//         std::make_tuple(
//             128,
//             [](int i){
//                 return -std::numeric_limits<float>::max() / 2 + rand() / static_cast<float>(RAND_MAX / std::numeric_limits<float>::max());
//             },
//             dim3(1),
//             dim3(128)
//         ),
//         std::make_tuple(
//             128,
//             [](int i){
//                 return -std::numeric_limits<float>::max() / 2 + rand() / static_cast<float>(RAND_MAX / std::numeric_limits<float>::max());
//             },
//             dim3(2, 3, 4),
//             dim3(5, 6, 7)
//         )
//     )
// );

// TEST_P(test_conv2d_float, check_output_vs_cpu){

// }


// Generate uniform numbers [0,1)
static void
initImage(float* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10;  // 2^-32
    }
}

TEST(linkcudnn, test) {
    cudnnHandle_t handle_;
    checkCudnnErr(cudnnCreate(&handle_));
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

TEST(convforward, cudnn_test) {
    sleep(10);
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