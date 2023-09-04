#include "operator_reduce.cuh"
#include "kernel_reduce.cuh"

Reduce::Reduce(REDUCE_OP op, bool end_of_graph): Operator(end_of_graph), _reduce_op(op) {}

Reduce::Reduce(Tensor* A, REDUCE_OP op)
    : Operator({A}, {new Tensor()}), _reduce_op(op) {
  ;
}

std::string Reduce::type_str() { return std::string("Reduce"); }

Reduce* Reduce::copy() {
    return new Reduce(_reduce_op, _end_of_graph);
}

void Reduce::infer_shape() {
    _output_tensors[0]->set_shape({});
}


void Reduce::forward() {
    dim3 GRID;
    dim3 BLOCK = 512;
    size_t shared_mem = BLOCK.x * sizeof(float);


    size_t work_n = _input_tensors[0]->_element_count;
    float *work_space = nullptr;
    while (work_n != 1){
        kreduce<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
            _input_tensors[0]->_element_count,
            work_n,
            work_space,
            work_space,
            _reduce_op
        );
        work_n = GRID.x;
    }
}


void Reduce::backward() {
    dim3 BLOCK;
    dim3 GRID;
    size_t shared_mem;

    BLOCK = dim3(32);
    shared_mem = 0;

    float alpha;
    switch (_reduce_op) {
        case REDUCE_OP::SUM:
            alpha = 1.f;
            break;
        // case REDUCE_OP::AVG:
        //     alpha = 1.f / _input_tensors[0]->_element_count;
        //     break;
        
        default:
            break;
    }



    Tensor s = _input_tensors[0]->grad();
    s.to(cudaMemoryTypeHost);
}