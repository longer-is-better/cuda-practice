#include<tuple>
#include<limits>
#include<functional>
#include<vector>
#include<gtest/gtest.h>
#include"relu.cuh"
#include"helper_cuda.h"

// todo type as param?

class test_relu_float_1d_input:
    public testing::TestWithParam<
        std::tuple<
            int,  // input len
            std::function<float(int)>,  // data generator
            dim3,  // grid
            dim3  //block
        >
    >
{
  public:
    int len = 0;
    std::function<float(int)> gen;
    dim3 grid, block;

    std::vector<float> input;
    std::vector<float> host_output;
    std::vector<float> fetch_output;

    float* device_input;
    float* device_output;


    test_relu_float_1d_input();
    ~test_relu_float_1d_input();
};

test_relu_float_1d_input::test_relu_float_1d_input(){
    std::tie(len, gen, grid, block) = GetParam();

    input.resize(len); for (int i = 0; i < len; i++) input[i] = gen(i);
    host_output.resize(len); for (int i = 0; i < len; i++) host_output[i] = input[i] > 0 ? input[i] : 0;
    fetch_output.resize(len);

    checkCudaErrors(cudaMalloc((void**)&device_input, len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&device_output, len * sizeof(float)));
};

test_relu_float_1d_input::~test_relu_float_1d_input(){
    checkCudaErrors(cudaFree(device_input));
    checkCudaErrors(cudaFree(device_output));
}


INSTANTIATE_TEST_SUITE_P(
    general,
    test_relu_float_1d_input,
    testing::Values(
        std::make_tuple(
            1,
            [](int i){
                return 1;
            },
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            10,
            [](int i){
                return -5 + i;
            },
            dim3(1),
            dim3(1)
        ),
        std::make_tuple(
            128,
            [](int i){
                return -0.5 + static_cast<float>(rand()) / RAND_MAX;
            },
            dim3(1),
            dim3(128)
        ),
        std::make_tuple(
            128,
            [](int i){
                return -std::numeric_limits<float>::max() / 2 + rand() / static_cast<float>(RAND_MAX / std::numeric_limits<float>::max());
            },
            dim3(1),
            dim3(128)
        ),
        std::make_tuple(
            128,
            [](int i){
                return -std::numeric_limits<float>::max() / 2 + rand() / static_cast<float>(RAND_MAX / std::numeric_limits<float>::max());
            },
            dim3(2, 3, 4),
            dim3(5, 6, 7)
        )
    )
);

TEST_P(test_relu_float_1d_input, check_output_vs_cpu){
    checkCudaErrors(cudaMemcpy(device_input, input.data(), len * sizeof(float), cudaMemcpyHostToDevice));
    // relu_forward<<<grid, block>>>(device_input, device_output, len, 1, 1);
    relu_forward<float><<<grid, block>>>(device_input, device_output, len, 1, 1);
    checkCudaErrors(cudaMemcpy(fetch_output.data(), device_output, len * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < len; i++) {
        EXPECT_FLOAT_EQ(host_output[i], fetch_output[i]) << i << " th, host_output: " <<  host_output[i] << ", fetch_output " << fetch_output[i] << std::endl;
    }
}