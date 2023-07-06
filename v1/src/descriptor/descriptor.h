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


struct Pool2dDesc {
    int* dim_n = nullptr;
    int* window_shape = nullptr;
    int* stride = nullptr;
    int* padding = nullptr;

    Pool2dDesc() = delete;
    Pool2dDesc(Pool2dDesc&& pd) = delete;
    Pool2dDesc(const Pool2dDesc& pd) = delete;
    Pool2dDesc& operator = (const Pool2dDesc& Pool2dDesc) = delete;

    Pool2dDesc(const std::vector<int>& window_shape, const std::vector<int>& padding, const std::vector<int>& stride);
    Pool2dDesc& operator = (Pool2dDesc&& Pool2dDesc);

    ~Pool2dDesc();

    void* operator new(std::size_t size);
    void operator delete(void *ptr);
};

typedef enum {
    SIGMOID      = 0,
    RELU         = 1,
    TANH         = 2,
    CLIPPED_RELU = 3,
    ELU          = 4,
    IDENTITY     = 5,
    SWISH        = 6
} ActivationMode;