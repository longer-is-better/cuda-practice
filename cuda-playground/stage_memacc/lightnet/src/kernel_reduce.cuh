#pragma once
#include <stdio.h>
#include <iostream>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>

enum class REDUCE_OP {
    SUM = 0
};

std::ostream& operator<<(std::ostream& os, const REDUCE_OP &op);

template <typename T>
__device__ void warpReduceSum(volatile T* shmem_ptr, int t) {
    // shmem_ptr[t] += shmem_ptr[t + 32];
    shmem_ptr[t] += shmem_ptr[t + 16];
    shmem_ptr[t] += shmem_ptr[t + 8];
    shmem_ptr[t] += shmem_ptr[t + 4];
    shmem_ptr[t] += shmem_ptr[t + 2];
    shmem_ptr[t] += shmem_ptr[t + 1];
}

template <typename T>
__global__ void kreduce(size_t total_n, size_t current_n, T *I, T *O, REDUCE_OP op) {
    extern __shared__ T partial[];

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T front = 0, back = 0;
    if (i < current_n) front = I[i];
    if (i + blockDim.x < current_n) back = I[i + blockDim.x];
    partial[threadIdx.x] = front + back;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 16; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 16) {
        warpReduceSum(partial, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        O[blockIdx.x] = partial[0];
    }
}


// bad implement
template <typename T>
__global__ void kreduce_dbbf(size_t total_n, T *I, T *O, REDUCE_OP op) {
    extern __shared__ T double_partial[];
    size_t partial_size = blockDim.x;
    double_partial[2 * partial_size] = 0.f;
    size_t gridSize = gridDim.x * blockDim.x;

    size_t gridIdx = 0;
    size_t partialIdx = gridIdx % 2;
    size_t I_Idx = gridIdx * gridSize + blockIdx.x * blockDim.x + threadIdx.x;
    double_partial[partialIdx * partial_size + threadIdx.x] = I_Idx < total_n ? I[I_Idx] : 0;

    while (gridIdx * gridSize <= total_n) {
        // preload
        size_t next_gridIdx = gridIdx + 1;
        size_t next_partialIdx = next_gridIdx % 2;
        I_Idx = next_gridIdx * gridSize + blockIdx.x * blockDim.x + threadIdx.x;
        double_partial[next_partialIdx * partial_size + threadIdx.x] = I_Idx < total_n ? I[I_Idx] : 0;

        // reduce current gridIdx
        T *partial = double_partial + partialIdx * partial_size;
        for (int s = blockDim.x / 2; s > 16; s >>= 1) {
            if (threadIdx.x < s) {
                partial[threadIdx.x] += partial[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            warpReduceSum(partial, threadIdx.x);
        }

        if (threadIdx.x == 0) {
            double_partial[2 * partial_size] += partial[0];
        }
        gridIdx++;
        partialIdx = next_partialIdx;
    }
    if (threadIdx.x == 0) {
        atomicAdd(O, double_partial[2 * partial_size]);
    }
}


template <typename T>
__global__ void kreduce_realdbbf(size_t total_n, T *I, T *O, REDUCE_OP op) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(total_n % grid.size() == 0); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 2; // Pipeline with two stages
    // Two batches must fit in shared memory:
    extern __shared__ T double_partial[];
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
            sizeof(T) * block.size(),
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
                sizeof(T) * block.size(),
                pipeline
            );
            pipeline.producer_commit();
        }

        // reduce current batch
        pipeline.consumer_wait();
        T *partial = double_partial + offset[stage_index];
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