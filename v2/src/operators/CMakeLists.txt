file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
add_library(operators SHARED ${srcs})
target_link_libraries(
    operators
    tensor
    kernels
    common_tools
    ${glog_LIBRARIES}
)
target_include_directories(operators PUBLIC .)
set_target_properties(operators PROPERTIES CUDA_ARCHITECTURES "86")