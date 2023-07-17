#include <string>
// #include <iostream>
// #include <memory>
// #include <filesystem>
#include <unordered_map>
// #include <unistd.h>

// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui.hpp>

// #include <torch/script.h> //one-stop header
#include <torch/torch.h>
#include<descriptor.h>

class Alexnet                                                                                                                                                              
{
  public:
    Conv2dDesc *Conv2d_0, *Conv2d_3, *Conv2d_6, *Conv2d_8, *Conv2d_10;
    Conv2dDesc *Linear_1, *Linear_4, *Linear_6;
    Pool2dDesc *MaxPool2d_2, *MaxPool2d_5, *MaxPool2d_12, *AvgPool2d;

    float *input = nullptr,
          *Conv2d_0_output, *MaxPool2d_2_output,
          *Conv2d_3_output, *MaxPool2d_5_output,
          *Conv2d_6_output,
          *Conv2d_8_output,
          *Conv2d_10_output, *MaxPool2d_12_output,
          *Linear_1_output,
          *Linear_4_output,
          *Linear_6_output,
          *AvgPool2d_output;
    TensorDesc *input_desc = nullptr,
               *Conv2d_0_output_desc, *MaxPool2d_2_output_desc,
               *Conv2d_3_output_desc, *MaxPool2d_5_output_desc,
               *Conv2d_6_output_desc,
               *Conv2d_8_output_desc,
               *Conv2d_10_output_desc, *MaxPool2d_12_output_desc,
               *Linear_1_output_desc,
               *Linear_4_output_desc,
               *Linear_6_output_desc,
               *AvgPool2d_output_desc,
               *flatten_output_desc;

    std::unordered_map<std::string, std::pair<TensorDesc*, float*>> weights;
    Alexnet(torch::jit::script::Module module, TensorDesc *input_desc);
    ~Alexnet();
    std::pair<TensorDesc*, float*> forward(float *input);
};