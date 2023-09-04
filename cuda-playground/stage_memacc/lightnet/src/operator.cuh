#pragma once

#include <map>
#include <utility>


class Operator {
public:
    std::string _name;
    bool _end_of_graph = false;
    cudaStream_t _cudastream = cudaStreamDefault;

    std::map<Operator*, bool> _prevoperators = {};  // bool: exist for topologicalSort
    std::map<Operator*, bool> _nextoperators = {};  // bool: exist


    Operator(){};
    Operator(bool end_of_graph): _end_of_graph(end_of_graph){
        ;
    };
    Operator(const Operator &op);




    virtual int indegree();
    virtual void set_cudastream(cudaStream_t cudastream);
    virtual std::string type_str() = 0;
    virtual Operator* copy() = 0;
    virtual void infer_shape() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;



    Operator(Operator &&op) = delete;
    Operator& operator = (const Operator &op) = delete;
    Operator& operator = (Operator &&op) = delete;
};