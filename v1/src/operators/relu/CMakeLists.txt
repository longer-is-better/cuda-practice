add_library(relu SHARED relu.cu)
target_include_directories(relu PUBLIC .)
set_target_properties(relu PROPERTIES CUDA_ARCHITECTURES "86")
# target_include_directories(multiply PUBLIC include)