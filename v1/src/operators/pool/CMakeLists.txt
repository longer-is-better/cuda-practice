# add_library(pool INTERFACE)
# target_include_directories(pool INTERFACE .)
# target_link_libraries(pool INTERFACE descriptor)

file(GLOB srcs *.cu)
add_library(pool SHARED ${srcs})
target_link_libraries(
    pool
    # utils
    descriptor
)
target_include_directories(pool PUBLIC .)
set_target_properties(pool PROPERTIES CUDA_ARCHITECTURES "86")