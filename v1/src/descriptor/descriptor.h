#pragma once

#include<string>
#include<vector>


struct TensorDesc {
    char* layOut = nullptr;
    int* shape = nullptr;
    TensorDesc(std::string lo, std::vector<int> sp);
    ~TensorDesc();
};


struct Conv2dDesc {
    int stride;
    int padding;
};