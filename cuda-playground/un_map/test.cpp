#include<iostream>
#include<string>
#include<map>
#include<unordered_map>

int main(){
    std::unordered_map<std::string, int> t1;
    t1["b"] = 2;
    t1["a"] = 1;
    t1["c"] = 3;
    t1["d"] = 4;
    for (auto a: t1) std::cout << a.first << " " << a.second << std::endl;
}