#include <glog/logging.h>

#include "network.h"
#include "tensor.h"


#include "elementwise.cuh"
// Network* Network::_trianer = nullptr;

// Network::Network(
//     ComputeGraph *computegraph,
//     bool train,
//     cudaStream_t cudastream
// ):
//     _cudastream(cudastream),
//     _train(train)
// {
//     // if (train) {
//         // if (_trianer) {
//         //     LOG(FATAL) << "train is singleton";
//         //     delete this;
//         //     return;
//         // } else {
//             // computegraph->copy(_input_tensors, _weight_tensors);
//             // _trianer = this;
//         // }
//     // } else {
//         computegraph->copy(_input_tensors, _weight_tensors);
//     // }
// }

// Network::~Network()
// {
//     // if (_trianer = this) _trianer = nullptr;
// }


// void Network::to(cudaMemoryType type){
//     for (Tensor* input_tensor: _input_tensors) {
//         input_tensor->to(type);
//     }
//     for (Tensor* weight_tensor: _weight_tensors) {
//         weight_tensor->to(type);
//     }

//     for (Operator* op: this->get_op_seq()) {
//         for (Tensor* output_tensor: op->_output_tensors) {
//             output_tensor->to(type);
//         }
//     }
// }

// /// @brief infer tensor shape, tensor malloc data, tensor malloc gradient if train
// /// @param sample_inputs 
// /// @param weight_path 
// void Network::init(std::vector<Tensor*> sample_inputs, std::string weight_path) {
//     CHECK_EQ(_input_tensors.size(), sample_inputs.size());
//     for (int i = 0; i < _input_tensors.size(); i++) {
//         *_input_tensors[i] = *sample_inputs[i];
//         if (_train) _input_tensors[i]->malloc_gradient();
//     }
//     for (Tensor* weight_tensor: _weight_tensors) {
//         if (_train) weight_tensor->malloc_gradient();
//     }
//     for (Operator *op: get_op_seq()) {
//         op->infer_shape();
//         for (Tensor *tensor: op->_output_tensors) {
//             tensor->malloc_data();
//             if (_train) tensor->malloc_gradient();
//         }
//     }
// }


// std::vector<Tensor*> Network::forward(std::vector<Tensor*> input_tensors){

//     CHECK_EQ(_input_tensors.size(), input_tensors.size());
//     for (int i = 0; i < _input_tensors.size(); i++) {
//         *_input_tensors[i] = *input_tensors[i];
//     }
//     for (Operator* op: get_op_seq()) {
//         op->forward();
//     }
//     checkCudaErrors(cudaDeviceSynchronize());
//     return get_output_tensors();
// }



// void Network::backward() {
//     for (auto it = get_op_seq().rbegin(); it != get_op_seq().rend(); ++it) {
//         (*it)->backward();
//     }
// }


// void Network::update_weights(float alpha) {
//     for (Tensor* weight_tensor: _weight_tensors) {
//         weight_tensor->update_weights(alpha, _cudastream);
//     }

//     float wtf[4] = {5, 5, 4, 2};
//     float *www;
//     checkCudaErrors(cudaMalloc(&www, 16));
//     checkCudaErrors(cudaMemcpy(www, wtf, 16, cudaMemcpyHostToDevice));
//     kelementwise_inplace<<<1, 32>>>(
//         4,
//         www,
//         1.f,
//         www,
//         ELE_OP::MULTIPLY
//     );
//     checkCudaErrors(cudaDeviceSynchronize());
//     float *back_www = new float[4];
//     checkCudaErrors(cudaMemcpy(back_www, www, 16, cudaMemcpyDeviceToHost));
//     int a = 1;
// }