#pragma once

#include<string>
#include<vector>


struct TensorDesc {
    int* dim_n = 0;
    char* layout = nullptr;
    int* shape = nullptr;
    int* stride = nullptr;
    TensorDesc(){};
    TensorDesc(const std::string& layout, const std::vector<int>& shape);
    TensorDesc(const TensorDesc& td) = delete;
    TensorDesc(TensorDesc&& td);

    TensorDesc& operator = (const TensorDesc& tensordesc) = delete;
    TensorDesc& operator = (TensorDesc&& tensordesc);

    ~TensorDesc();

    void init(const std::string& layout, const std::vector<int>& shape);

    void* operator new(std::size_t size);
    void operator delete(void *ptr);
};


struct Conv2dDesc {
    int stride;
    int padding;
};