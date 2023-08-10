add_library(
    kernels
    SHARED
    all_kernels.cu
)

# target_link_libraries(
#     kernels
# )

target_include_directories(kernels PUBLIC .)

set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES "86")