find_package(CUDA REQUIRED)
add_library(cuda_tools INTERFACE)
target_link_libraries(
    cuda_tools
    INTERFACE
    ${glog_LIBRARIES}
)
target_include_directories(cuda_tools INTERFACE cuda ${CUDA_LIBRARIES})
set_target_properties(cuda_tools PROPERTIES CUDA_ARCHITECTURES "86")


add_library(cudnn_tools INTERFACE)
target_link_libraries(
    cudnn_tools
    INTERFACE
    ${glog_LIBRARIES}
)
target_include_directories(cudnn_tools INTERFACE cudnn)


# add_library(common_tools INTERFACE)
# target_link_libraries(
#     common_tools
#     INTERFACE
#     ${glog_LIBRARIES}
# )
# target_include_directories(common_tools INTERFACE common)


add_library(common_tools SHARED common/common_tools.cpp)
target_link_libraries(
    common_tools
    ${glog_LIBRARIES}
)
target_include_directories(common_tools PUBLIC common)
# set_target_properties(tensor PROPERTIES CUDA_ARCHITECTURES "86")