
#include<string>
#include<string.h>
#include<vector>
#include<iostream>



int main() {
    std::string lo("abc");
    std::cout << lo.size() << std::endl;
    
    char* layOut = static_cast<char*>(malloc(lo.size() + 1));

    std::cout << *layOut << std::endl;

    memcpy(layOut, lo.c_str(), lo.size());


};