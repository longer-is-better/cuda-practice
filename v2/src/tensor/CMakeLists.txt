# add_library(pool INTERFACE)
# target_include_directories(pool INTERFACE .)
# target_link_libraries(pool INTERFACE descriptor)

file(GLOB srcs *.cu)
add_library(tensor SHARED ${srcs})
target_link_libraries(
    tensor
    cuda_tools
    # operators
    kernels
    gflags
    ${glog_LIBRARIES}
)
target_include_directories(tensor PUBLIC .)
set_target_properties(tensor PROPERTIES CUDA_ARCHITECTURES "86")