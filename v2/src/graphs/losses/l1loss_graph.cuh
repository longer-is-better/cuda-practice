#pragma once

#include "compute_graph.h"
#include "kreduce.cuh"


class L1LossGraph: public ComputeGraph
{
public:
    L1LossGraph(REDUCE_OP op = REDUCE_OP::AVG);
    ~L1LossGraph();
};
