#include <string>
#include <iostream>
#include <memory>
#include <filesystem>
#include <unordered_map>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <torch/script.h> //one-stop header
#include <torch/torch.h>

class Alexnet
{
private:
    std::unordered_map<std:: string, float*> weights;
public:
    Alexnet(std::string model_apath);
    ~Alexnet();
};