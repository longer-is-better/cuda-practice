find_package(CUDA REQUIRED)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)
add_library(loss SHARED ${srcs})
target_link_libraries(
    loss
    network
    kernels
    ${glog_LIBRARIES}
)
target_include_directories(loss PUBLIC . ${CUDA_INCLUDE_DIRS})
set_target_properties(loss PROPERTIES CUDA_ARCHITECTURES "86")