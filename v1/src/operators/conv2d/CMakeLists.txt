add_library(conv2d SHARED conv2d.cu)
target_link_libraries(
    conv2d
    utils
    descriptor
)
target_include_directories(conv2d PUBLIC .)
set_target_properties(conv2d PROPERTIES CUDA_ARCHITECTURES "86")