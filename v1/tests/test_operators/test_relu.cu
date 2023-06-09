#include<tuple>
#include <limits>
#include<functional>
#include<gtest/gtest.h>
#include"relu.cuh"
#include"helper_cuda.h"

// todo type as param?

class test_relu_float_1d_input:
    public testing::TestWithParam<
        std::tuple<
            int,  // input len
            std::function<float(int)>  // data generator
        >
    > {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
};

INSTANTIATE_TEST_SUITE_P(
    general,
    test_relu_float_1d_input,
    testing::Values(
        std::make_tuple(
            1,
            [](int i){
                return static_cast<float>(i) / 23 % 17;
            }
        )
        // ,
        // std::make_tuple(
        //     128,
        //     [](int i){
        //         return -std::numeric_limits<float>::max() + rand() / static_cast<float>(RAND_MAX / std::numeric_limits<float>::max());
        //     }
        // )
    )
);

TEST_P(test_relu_float_1d_input, check_output_vs_cpu){
    std::cout << "@!!!!!!!!!!!!!!!!!!!!";
    int len;
    std::function<float(int)> gen;
    std::tie(len, gen) = GetParam();

    relu_forward<float, 4, 1, 1><<<1, 2>>>(nullptr, nullptr);
    checkCudaErrors(cudaDeviceSynchronize());
}