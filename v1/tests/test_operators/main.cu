#include<iostream>
#include<gtest/gtest.h>
#include"log.h"

int main(int argc, char **argv) {
    std::cout << "waiting for attach..." << std::endl;
    D(sleep(5);)
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}