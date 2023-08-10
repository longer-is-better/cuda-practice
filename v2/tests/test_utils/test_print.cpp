#include<vector>
#include<numeric>
#include<gtest/gtest.h>

#include"print_tensor.h"

TEST(test_print, tensor){
    std::vector<float> ts(100);
    std::iota(ts.begin(), ts.end(), 0);
    std::vector<int> dim1{1, 3, 5, 5};
    PrintTensor(ts.data(), dim1.data(), dim1.size(), "ts1");

    std::vector<int> dim2{2, 2};
    PrintTensor(ts.data(), dim2.data(), dim2.size(), "ts2");

    std::vector<int> dim3{1, 1, 2, 2, 1};
    PrintTensor(ts.data(), dim3.data(), dim3.size(), "ts3");
}