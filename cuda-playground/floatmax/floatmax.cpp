#include<limits>
#include<iostream>
#include<math.h>

int main(){
    float fmin = std::numeric_limits<float>::min();
    float fmax = std::numeric_limits<float>::max();
    double ans = static_cast<double>(fmax) - static_cast<double>(fmin);
    std::cout << fmax << std::endl;
    std::cout << 2 * fmax << std::endl;
    std::cout << fmin + fmax << std::endl;
    std::cout << ans << std::endl;
    std::cout << static_cast<double>(fmax) - static_cast<double>(fmin) << std::endl;
    return 0;
}