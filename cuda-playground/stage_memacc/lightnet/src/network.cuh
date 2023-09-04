#pragma once


class Network 
{
public:
    static Network *_trianer;
    cudaStream_t _cudastream;



    Network();
    ~Network();

    void to(cudaMemoryType type);
    void backward();
    void update_weights(float alpha);


    Network(const Network& network) = delete;
    Network(Network&& network) = delete;
    Network& operator=(const Network& network) = delete;
    Network& operator=(Network&& network) = delete;
};
