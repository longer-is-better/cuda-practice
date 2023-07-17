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

#include "log.h"
#include "helper_cuda.h"
#include "alexnet.cuh"
#include "print_tensor.h"
#include "conv2d.cuh"


TEST (torch_alexnet, smoke) {
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
    std::cout << "output: \n" <<  output << '\n';

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


TEST (alexnet, smoke) {
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

    TensorDesc *input_desc = new TensorDesc("nchw", {1, 3, 224, 224});
    Alexnet alxnt = Alexnet(module, input_desc);

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
    cv::Mat input = cv::dnn::blobFromImage(img_float);
    // 5. Removing mean values of the RGB channels
    // the values are from following link.
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202

    for (int i = 0 * 224 * 224; i < 1 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.485) / 0.229;
    for (int i = 1 * 224 * 224; i < 2 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.456) / 0.224;
    for (int i = 2 * 224 * 224; i < 3 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.406) / 0.225;
    
    // 6. Execute the model and turn its output into a tensor
    auto res = alxnt.forward(reinterpret_cast<float*>(input.data));

    auto res_torch = torch::from_blob(res.second, {1,1000},torch::kFloat32);
    std::cout << "res_torch.sizes() " << res_torch.sizes() << std::endl;
    std::cout << "res_torch: \n" <<  res_torch << '\n';

    // 7. Load labels
    std::string label_file = "/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/synset_words.txt";
    std::ifstream rf(label_file.c_str());
    CHECK(rf) << "Unable to open labels file" << label_file;
    std::string line;
    std::vector<std::string> labels;
    while(std::getline(rf, line)){labels.push_back(line);}
    
    // 8. print predicted top-3 labels
    std::tuple<torch::Tensor, torch::Tensor> result = res_torch.sort(-1, true);
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


TEST (alexnet, vs_torch) {
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

    TensorDesc *input_desc = new TensorDesc("nchw", {1, 3, 224, 224});
    Alexnet alxnt = Alexnet(module, input_desc);

    // load image with opencv and transform.
    // 1. read image
    cv::Mat image = cv::imread("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/dog.png", cv::IMREAD_COLOR);
    // 2. convert color space, opencv read the image in BGR
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    // convert to float format
    image.convertTo(img_float, CV_32F, 1.0/255);
    // 3. resize the image for resnet101 model
    cv::resize(img_float, img_float, cv::Size(224, 224),cv::INTER_AREA);
    // 4. transform to tensor
    cv::Mat input = cv::dnn::blobFromImage(img_float);
    auto img_tensor = torch::from_blob(input.data, {1,3,224,224},torch::kFloat32);
    // 5. Removing mean values of the RGB channels
    // the values are from following link.
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202

    for (int i = 0 * 224 * 224; i < 1 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.485) / 0.229;
    for (int i = 1 * 224 * 224; i < 2 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.456) / 0.224;
    for (int i = 2 * 224 * 224; i < 3 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.406) / 0.225;

    // 6. Execute the model and turn its output into a tensor
    auto res = alxnt.forward(reinterpret_cast<float*>(input.data));


    // Create vectors of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);
    
    // 6. Execute the model and turn its output into a tensor
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << "torch output[0]: \n" << output[0] << '\n';

    // // 7. Load labels
    // std::string label_file = "/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/synset_words.txt";
    // std::ifstream rf(label_file.c_str());
    // CHECK(rf) << "Unable to open labels file" << label_file;
    // std::string line;
    // std::vector<std::string> labels;
    // while(std::getline(rf, line)){labels.push_back(line);}
    
    // // 8. print predicted top-3 labels
    // std::tuple<torch::Tensor, torch::Tensor> result = output.sort(-1, true);
    // torch::Tensor top_scores = std::get<0>(result)[0];
    // torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
    
    // auto top_scores_a = top_scores.accessor<float, 1>();
    // auto top_idxs_a = top_idxs.accessor<int, 1>();
    // for(int i=0; i<3;i++){
    //     int idx = top_idxs_a[i];
	//     std::cout << "top-" << i+1 << " label: ";
	//     std::cout << labels[idx] << ",score: " << top_scores_a[i] << std::endl;
    // }
}

TEST (alexnet_features, vs_torch) {
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



    torch::jit::script::Module FAC01;
    try{
        // Deserialize the scriptmodule from a file using torch::jit::load().
        // run cuda-practice/v1/tests/test_networks/test_alexnet/save_model.py for traced_alexnet_model.pt
        FAC01 = torch::jit::load("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/traced_alexnet_model_features.pt");
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
	    GTEST_FAIL();
    }

    std::cout << "model load ok\n";

    TensorDesc *input_desc = new TensorDesc("nchw", {1, 3, 224, 224});
    Alexnet alxnt = Alexnet(module, input_desc);

    // load image with opencv and transform.
    // 1. read image
    cv::Mat image = cv::imread("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/dog.png", cv::IMREAD_COLOR);
    // 2. convert color space, opencv read the image in BGR
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    // convert to float format
    image.convertTo(img_float, CV_32F, 1.0/255);
    // 3. resize the image for resnet101 model
    cv::resize(img_float, img_float, cv::Size(224, 224),cv::INTER_AREA);
    // 4. transform to tensor
    cv::Mat input = cv::dnn::blobFromImage(img_float);
    // 5. Removing mean values of the RGB channels
    // the values are from following link.
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202

    // TensorDesc input_before_norm("nchw", {1, 3, 224, 224});

    // PrintTensor(
    //     (float*)input.data,
    //     input_before_norm.shape,
    //     input_before_norm.dim_n[0], 
    //     "input before norm"
    // );

    for (int i = 0 * 224 * 224; i < 1 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.485) / 0.229;
    for (int i = 1 * 224 * 224; i < 2 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.456) / 0.224;
    for (int i = 2 * 224 * 224; i < 3 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.406) / 0.225;

    // 6. Execute the model and turn its output into a tensor
    auto res = alxnt.forward(reinterpret_cast<float*>(input.data));


    auto img_tensor = torch::from_blob(input.data, {1,3,224,224},torch::kFloat32);

    // Create vectors of inputs.
    std::vector<torch::jit::IValue> inputs1, inputs2;
    inputs1.push_back(torch::ones({1,3,224,224}));
    inputs2.push_back(img_tensor);
    
    // 6. Execute the model and turn its output into a tensor
    at::Tensor output = FAC01.forward(inputs2).toTensor();
    std::cout << "output.sizes() " << output.sizes() << std::endl;
    std::vector<float> tensorArray(output.data<float>(), output.data<float>() + output.numel());
    // std::cout << output.slice(/*dim=*/1,/*start=*/0,/*end=*/3) << '\n';
    std::cout << "torch output[0][0]: \n" << output[0][0] << '\n';

    // 7. Load labels
    // std::string label_file = "/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/synset_words.txt";
    // std::ifstream rf(label_file.c_str());
    // CHECK(rf) << "Unable to open labels file" << label_file;
    // std::string line;
    // std::vector<std::string> labels;
    // while(std::getline(rf, line)){labels.push_back(line);}
    
    // // 8. print predicted top-3 labels
    // std::tuple<torch::Tensor, torch::Tensor> result = output.sort(-1, true);
    // torch::Tensor top_scores = std::get<0>(result)[0];
    // torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
    
    // auto top_scores_a = top_scores.accessor<float, 1>();
    // auto top_idxs_a = top_idxs.accessor<int, 1>();
    // for(int i=0; i<3;i++){
    //     int idx = top_idxs_a[i];
	//     std::cout << "top-" << i+1 << " label: ";
	//     std::cout << labels[idx] << ",score: " << top_scores_a[i] << std::endl;
    // }
}


TEST (alexnet_FAF, vs_torch) {
    torch::jit::script::Module FAF;
    try{
        // Deserialize the scriptmodule from a file using torch::jit::load().
        // run cuda-practice/v1/tests/test_networks/test_alexnet/save_model.py for traced_alexnet_model.pt
        FAF = torch::jit::load("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/traced_alexnet_model_FAF.pt");
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
	    GTEST_FAIL();
    }

    std::cout << "model load ok\n";

    TensorDesc *input_desc = new TensorDesc("nchw", {1, 3, 224, 224});
    Alexnet alxnt = Alexnet(FAF, input_desc);

    // load image with opencv and transform.
    // 1. read image
    cv::Mat image = cv::imread("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/dog.png", cv::IMREAD_COLOR);
    // 2. convert color space, opencv read the image in BGR
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    // convert to float format
    image.convertTo(img_float, CV_32F, 1.0/255);
    // 3. resize the image for resnet101 model
    cv::resize(img_float, img_float, cv::Size(224, 224),cv::INTER_AREA);
    // 4. transform to tensor
    cv::Mat input = cv::dnn::blobFromImage(img_float);
    // 5. Removing mean values of the RGB channels
    // the values are from following link.
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202

    // TensorDesc input_before_norm("nchw", {1, 3, 224, 224});

    // PrintTensor(
    //     (float*)input.data,
    //     input_before_norm.shape,
    //     input_before_norm.dim_n[0], 
    //     "input before norm"
    // );

    for (int i = 0 * 224 * 224; i < 1 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.485) / 0.229;
    for (int i = 1 * 224 * 224; i < 2 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.456) / 0.224;
    for (int i = 2 * 224 * 224; i < 3 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.406) / 0.225;

    // 6. Execute the model and turn its output into a tensor
    auto res = alxnt.forward(reinterpret_cast<float*>(input.data));


    auto img_tensor = torch::from_blob(input.data, {1,3,224,224},torch::kFloat32);

    // Create vectors of inputs.
    std::vector<torch::jit::IValue> inputs1, inputs2;
    inputs1.push_back(torch::ones({1,3,224,224}));
    inputs2.push_back(img_tensor);
    
    // 6. Execute the model and turn its output into a tensor
    at::Tensor output = FAF.forward(inputs2).toTensor();
    std::cout << "output.sizes() " << output.sizes() << std::endl;
    std::vector<float> tensorArray(output.data<float>(), output.data<float>() + output.numel());
    // std::cout << output.slice(/*dim=*/1,/*start=*/0,/*end=*/3) << '\n';
    std::cout << "torch output[0]: \n" << output[0] << '\n';




    for (int i = 0; i < 9216; i++) EXPECT_FLOAT_EQ(res.second[i], output[0][i].item<float>());
    // 7. Load labels
    // std::string label_file = "/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/synset_words.txt";
    // std::ifstream rf(label_file.c_str());
    // CHECK(rf) << "Unable to open labels file" << label_file;
    // std::string line;
    // std::vector<std::string> labels;
    // while(std::getline(rf, line)){labels.push_back(line);}
    
    // // 8. print predicted top-3 labels
    // std::tuple<torch::Tensor, torch::Tensor> result = output.sort(-1, true);
    // torch::Tensor top_scores = std::get<0>(result)[0];
    // torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
    
    // auto top_scores_a = top_scores.accessor<float, 1>();
    // auto top_idxs_a = top_idxs.accessor<int, 1>();
    // for(int i=0; i<3;i++){
    //     int idx = top_idxs_a[i];
	//     std::cout << "top-" << i+1 << " label: ";
	//     std::cout << labels[idx] << ",score: " << top_scores_a[i] << std::endl;
    // }
}


TEST (alexnet_FAFC02_mod, vs_torch) {
    torch::jit::script::Module FAFC02_mod;
    try{
        // Deserialize the scriptmodule from a file using torch::jit::load().
        // run cuda-practice/v1/tests/test_networks/test_alexnet/save_model.py for traced_alexnet_model.pt
        FAFC02_mod = torch::jit::load("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/traced_alexnet_model_FAFC02_mod.pt");
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
	    GTEST_FAIL();
    }

    std::cout << "model load ok\n";

    TensorDesc *input_desc = new TensorDesc("nchw", {1, 3, 224, 224});
    Alexnet alxnt = Alexnet(FAFC02_mod, input_desc);

    // load image with opencv and transform.
    // 1. read image
    cv::Mat image = cv::imread("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/dog.png", cv::IMREAD_COLOR);
    // 2. convert color space, opencv read the image in BGR
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    // convert to float format
    image.convertTo(img_float, CV_32F, 1.0/255);
    // 3. resize the image for resnet101 model
    cv::resize(img_float, img_float, cv::Size(224, 224),cv::INTER_AREA);
    // 4. transform to tensor
    cv::Mat input = cv::dnn::blobFromImage(img_float);
    // 5. Removing mean values of the RGB channels
    // the values are from following link.
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202

    // TensorDesc input_before_norm("nchw", {1, 3, 224, 224});

    // PrintTensor(
    //     (float*)input.data,
    //     input_before_norm.shape,
    //     input_before_norm.dim_n[0], 
    //     "input before norm"
    // );

    for (int i = 0 * 224 * 224; i < 1 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.485) / 0.229;
    for (int i = 1 * 224 * 224; i < 2 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.456) / 0.224;
    for (int i = 2 * 224 * 224; i < 3 * 224 * 224; i ++) ((float*)input.data)[i] = (((float*)input.data)[i] - 0.406) / 0.225;

    // 6. Execute the model and turn its output into a tensor
    auto res = alxnt.forward(reinterpret_cast<float*>(input.data));


    auto img_tensor = torch::from_blob(input.data, {1,3,224,224},torch::kFloat32);

    // Create vectors of inputs.
    std::vector<torch::jit::IValue> inputs1, inputs2;
    inputs1.push_back(torch::ones({1,3,224,224}));
    inputs2.push_back(img_tensor);
    
    // 6. Execute the model and turn its output into a tensor
    at::Tensor output = FAFC02_mod.forward(inputs2).toTensor();
    std::cout << "output.sizes() " << output.sizes() << std::endl;
    std::vector<float> tensorArray(output.data<float>(), output.data<float>() + output.numel());
    // std::cout << output.slice(/*dim=*/1,/*start=*/0,/*end=*/3) << '\n';
    std::cout << "torch output[0]: \n" << output[0] << '\n';

    // 7. Load labels
    // std::string label_file = "/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/synset_words.txt";
    // std::ifstream rf(label_file.c_str());
    // CHECK(rf) << "Unable to open labels file" << label_file;
    // std::string line;
    // std::vector<std::string> labels;
    // while(std::getline(rf, line)){labels.push_back(line);}
    
    // // 8. print predicted top-3 labels
    // std::tuple<torch::Tensor, torch::Tensor> result = output.sort(-1, true);
    // torch::Tensor top_scores = std::get<0>(result)[0];
    // torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
    
    // auto top_scores_a = top_scores.accessor<float, 1>();
    // auto top_idxs_a = top_idxs.accessor<int, 1>();
    // for(int i=0; i<3;i++){
    //     int idx = top_idxs_a[i];
	//     std::cout << "top-" << i+1 << " label: ";
	//     std::cout << labels[idx] << ",score: " << top_scores_a[i] << std::endl;
    // }
}




TEST (CAF, torch_jit_trace) {
    torch::jit::script::Module caf;
    try{
        caf = torch::jit::load("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/traced_alexnet_model_CAF.pt");
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
	    GTEST_FAIL();
    }

    TensorDesc *weight_desc = new TensorDesc("oihw", {1, 9216, 1, 1});
    float *weight = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&weight, weight_desc->tensor_size()));
    TensorDesc *bias_desc = new TensorDesc("o", {1});
    float *bias = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&bias, bias_desc->tensor_size()));
    for (const auto& pair : caf.named_parameters()) {
        if (pair.name == "fc.weight") {
            std::cout << pair.name << std::endl;
            checkCudaErrors(
                cudaMemcpy(
                    weight,
                    pair.value.contiguous().data_ptr<float>(),
                    pair.value.numel() * pair.value.itemsize(),
                    cudaMemcpyHostToDevice
                )
            );
        } else if (pair.name == "fc.bias") {
            std::cout << pair.name << std::endl;
            checkCudaErrors(
                cudaMemcpy(
                    bias,
                    pair.value.contiguous().data_ptr<float>(),
                    pair.value.numel() * pair.value.itemsize(),
                    cudaMemcpyHostToDevice
                )
            );
        } else LOGERR("unrecognized layer" + pair.name);
    }


    std::cout << *weight_desc << std::endl;
    PrintTensor(weight, weight_desc->shape, *weight_desc->dim_n, "weight");

    std::cout << *bias_desc << std::endl;
    PrintTensor(bias, bias_desc->shape, *bias_desc->dim_n, "bias");

    TensorDesc *input_desc = new TensorDesc("nchw", {1, 9216, 1, 1});
    float *input = nullptr;
    checkCudaErrors(
        cudaMallocManaged(
            (void**)&input,
            input_desc->tensor_size()
        )
    );
    for (int i = 0; i < 9216; i++) input[i] = i;

    at::Tensor torch_input = torch::from_blob(input, {1,9216},torch::kFloat32);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_input);



    std::cout << *input_desc << std::endl;
    PrintTensor(input, input_desc->shape, *input_desc->dim_n, "input");


    Conv2dDesc *conv_desc = nullptr;
    checkCudaErrors(
        cudaMallocManaged(
            (void**)&conv_desc,
            sizeof(Conv2dDesc)
        )
    );
    conv_desc->padding = 0;
    conv_desc->stride = 1;

    TensorDesc *output_desc = conv2d_forward_shape_infer(input_desc, weight_desc, conv_desc);
    float *output = nullptr;
    checkCudaErrors(
        cudaMallocManaged(
            (void**)&output,
            output_desc->tensor_size()
        )
    );

    at::Tensor torch_output = caf.forward(inputs).toTensor();
    std::cout << "torch_output: " << torch_output << std::endl;

    conv2d_bias_active_forward_naive<<<dim3(1, 1, 1), dim3(4, 1, 1)>>>(
        input,
        input_desc,
        weight,
        weight_desc,
        conv_desc,
        bias,
        bias_desc,
        SIGMOID,
        output,
        output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << *output_desc << std::endl;
    PrintTensor(output, output_desc->shape, *output_desc->dim_n, "output");
}