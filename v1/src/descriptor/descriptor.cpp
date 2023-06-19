#include<string>
#include<string.h>
#include<vector>
#include"descriptor.h"



TensorDesc::TensorDesc(std::string lo, std::vector<int> sp) {
    if (!(layOut = static_cast<char*>(malloc(lo.size() + 1)))) exit(-1);
    memcpy(layOut, lo.c_str(), lo.size() + 1);

    if (!(shape = static_cast<int*>(malloc(sp.size() * sizeof(int))))) exit(-1);
    memcpy(shape, sp.data(), sp.size() * sizeof(int));
}
TensorDesc::~TensorDesc(){
    free(layOut);
    free(shape);
};