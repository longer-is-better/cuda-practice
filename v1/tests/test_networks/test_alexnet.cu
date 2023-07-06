#include <string>
#include <iostream>
// #include <memory>
// #include <unistd.h>

#include"gtest/gtest.h"

#include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui.hpp>

#include <torch/script.h> //one-stop header
#include <torch/torch.h>


TEST (torch_alexnet, smoke_1_image) {
    torch::jit::script::Module module;
    try{
        // Deserialize the scriptmodule from a file using torch::jit::load().
        // run cuda-practice/v1/tests/test_networks/test_alexnet/save_model.py for traced_alexnet_model.pt
        module = torch::jit::load("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/traced_alexnet_model.pt");
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
	    GTEST_FAIL();
    }
    std::cout << "model load ok\n";

    // 提取权重
    std::map<std::string, torch::Tensor> state_dict;
    for (const auto& pair : module.named_parameters()) {
        state_dict[pair.name] = pair.value.clone();
    }

    // 打印权重
    for (const auto& pair : state_dict) {
        std::cout << pair.first << "   ";
        // 获取张量的形状
        torch::IntArrayRef shape = pair.second.sizes();

        std::cout << "Tensor Shape: ";
        for (int64_t dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        if (pair.first == "features.0.bias") std::cout << pair.second;
        std::cout << std::endl;
    }

    // load image with opencv and transform.
    // 1. read image
    cv::Mat image;

	char *buffer;
	//也可以将buffer作为输出参数
	if((buffer = getcwd(NULL, 0)) == NULL) perror("getcwd error");
	else {
        printf("%s\n", buffer);
		free(buffer);
	}

    image = cv::imread("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/dog.png", cv::IMREAD_COLOR);
    // 2. convert color space, opencv read the image in BGR
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    // convert to float format
    image.convertTo(img_float, CV_32F, 1.0/255);
    // 3. resize the image for resnet101 model
    cv::resize(img_float, img_float, cv::Size(224, 224),cv::INTER_AREA);
    // 4. transform to tensor
    auto img_tensor = torch::from_blob(img_float.data, {1,224,224,3},torch::kFloat32);
    // in pytorch, batch first, then channel
    img_tensor = img_tensor.permute({0,3,1,2}); 
    // 5. Removing mean values of the RGB channels
    // the values are from following link.
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202
    img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
    
    // Create vectors of inputs.
    std::vector<torch::jit::IValue> inputs1, inputs2;
    inputs1.push_back(torch::ones({1,3,224,224}));
    inputs2.push_back(img_tensor);
    
    // 6. Execute the model and turn its output into a tensor
    at::Tensor output = module.forward(inputs2).toTensor();
    std::cout << "output.sizes() " << output.sizes() << std::endl;
    std::cout << output.slice(/*dim=*/1,/*start=*/0,/*end=*/3) << '\n';

    // 7. Load labels
    std::string label_file = "/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/synset_words.txt";
    std::ifstream rf(label_file.c_str());
    CHECK(rf) << "Unable to open labels file" << label_file;
    std::string line;
    std::vector<std::string> labels;
    while(std::getline(rf, line)){labels.push_back(line);}
    
    // 8. print predicted top-3 labels
    std::tuple<torch::Tensor, torch::Tensor> result = output.sort(-1, true);
    torch::Tensor top_scores = std::get<0>(result)[0];
    torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
    
    auto top_scores_a = top_scores.accessor<float, 1>();
    auto top_idxs_a = top_idxs.accessor<int, 1>();
    for(int i=0; i<3;i++){
        int idx = top_idxs_a[i];
	    std::cout << "top-" << i+1 << " label: ";
	    std::cout << labels[idx] << ",score: " << top_scores_a[i] << std::endl;
    }
}

