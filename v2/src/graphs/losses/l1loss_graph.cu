#include "l1loss_graph.cuh"
#include "operators.h"

L1LossGraph::L1LossGraph(REDUCE_OP op)
{
    _input_tensors.push_back(new Tensor());  // predict
    _input_tensors.push_back(new Tensor());  // target
    Operator *sub = new ElementWise(_input_tensors[0], _input_tensors[1], ELE_OP::SUB);
    Operator *abs = new Map(sub->_output_tensors[0], MAP_OP::ABS);
    Operator *red = new Reduce(abs->_output_tensors[0], op);
}

L1LossGraph::~L1LossGraph()
{
}