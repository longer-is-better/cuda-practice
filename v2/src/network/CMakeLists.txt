find_package(CUDA REQUIRED)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/compute_graph.cu)
add_library(network SHARED ${srcs})
target_link_libraries(
    network
    tensor
    operators
    ${glog_LIBRARIES}
)
target_include_directories(network PUBLIC . ${CUDA_INCLUDE_DIRS})
set_target_properties(network PROPERTIES CUDA_ARCHITECTURES "86")