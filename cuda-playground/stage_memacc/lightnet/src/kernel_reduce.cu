#include <iostream>

#include "kernel_reduce.cuh"

std::ostream& operator<<(std::ostream& os, const REDUCE_OP &op) {
    switch (op) {
        case REDUCE_OP::SUM:
            os << "REDUCE_OP::SUM";
            break;

        default:
            break;
    }
    return os;
}



__global__ void kreduce_realdbbf(size_t total_n, float *I, float *O, REDUCE_OP op) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(total_n % grid.size() == 0); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 2; // Pipeline with two stages
    // Two batches must fit in shared memory:
    extern __shared__ float double_partial[];
    double_partial[2 * block.size()] = 0.f;
    size_t offset[stages_count] = { 0, block.size() }; // Offsets to each batch

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    size_t batch = 0;
    size_t stage_index = 0;

    if (block_batch(batch) < total_n) {
        pipeline.producer_acquire();
        cuda::memcpy_async(
            block,
            double_partial + offset[stage_index],
            I + block_batch(batch),
            sizeof(float) * block.size(),
            pipeline
        );
        pipeline.producer_commit();
    }

    while (batch * grid.size() <= total_n) {
        // preload
        size_t next_batch = batch + 1;
        size_t next_stage_index = next_batch % stages_count;
        if (block_batch(next_batch) < total_n) {
            pipeline.producer_acquire();
            cuda::memcpy_async(
                block,
                double_partial + offset[next_stage_index],
                I + block_batch(next_batch),
                sizeof(float) * block.size(),
                pipeline
            );
            pipeline.producer_commit();
        }

        // reduce current batch
        pipeline.consumer_wait();
        float *partial = double_partial + offset[stage_index];
        for (int s = block.size() / 2; s > 16; s >>= 1) {
            if (threadIdx.x < s) {
                partial[threadIdx.x] += partial[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            warpReduceSum(partial, threadIdx.x);
        }

        if (threadIdx.x == 0) {
            double_partial[2 * block.size()] += partial[0];
        }
        pipeline.consumer_release();
        batch = next_batch;
        stage_index = next_stage_index;
    }
    if (threadIdx.x == 0) {
        atomicAdd(O, double_partial[2 * block.size()]);
    }
}