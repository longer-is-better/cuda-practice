find_package(PythonInterp REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd_party/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(alexnet SHARED alexnet.cu)
target_link_libraries(
    alexnet
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    conv2d
    pool
    relu
)

target_include_directories(alexnet PUBLIC . ${PROJECT_SOURCE_DIR}/Common)
set_target_properties(alexnet PROPERTIES CUDA_ARCHITECTURES "86")