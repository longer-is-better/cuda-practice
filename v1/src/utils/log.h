#include<iostream>

#define LOGERR(msg) {std::cerr << __FILE__ << ": " << __LINE__ << msg << std::endl;}

#if defined(__CUDACC_DEBUG__) || !defined(NDEBUG)
#define D(...) __VA_ARGS__
#else 
#define D(...)
#endif

#if !defined(__CUDACC_DEBUG__) || defined(NDEBUG)
#define R(...) __VA_ARGS__
#else 
#define R(...)
#endif