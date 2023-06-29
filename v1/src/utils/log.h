#include"stdio.h"
#include<iostream>

#define LOGERR(msg) {std::cerr << __FILE__ << ": " << __LINE__ << msg << std::endl;}


#define DR

#if !defined(__CUDACC_DEBUG__) && defined(DR)
    #define D(...)
#else
    #define D(...) __VA_ARGS__
#endif

#if defined(__CUDACC_DEBUG__) && defined(DR)
    #define R(...)
#else
    #define R(...) __VA_ARGS__
#endif