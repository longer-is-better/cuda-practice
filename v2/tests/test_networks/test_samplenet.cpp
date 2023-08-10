#include <vector>
#include <iostream>
#include <random>
#include <gtest/gtest.h>

TEST(train, smoke) {
    int n = 10;
    float piece[n][2];
    float ans[n];


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < n * 2; i++) ((float*)piece)[i] = dis(gen);
    for (int i = 0; i < n; i++) ans[n] = piece[n][0] + piece[n][1];


    Network
}