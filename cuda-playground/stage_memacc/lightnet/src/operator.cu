#include <vector>

#include "operator.cuh"


Operator::Operator(const Operator& op) {
    _name = op._name;
}


void Operator::set_cudastream(cudaStream_t cudastream) {
    _cudastream = cudastream;
}

int Operator::indegree() {
    int ans = 0;
    for (std::pair<Operator *const, bool>& p: _prevoperators) {
        ans += p.second;
    }
    return ans;
}
