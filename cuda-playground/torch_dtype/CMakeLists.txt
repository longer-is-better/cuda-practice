cmake_minimum_required(VERSION 3.16)
project(dtype LANGUAGES CXX CUDA)

find_package(PythonInterp REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/../libtorch/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(test main.cpp)
target_link_libraries(
    test
    ${TORCH_LIBRARIES}
    # ${OpenCV_LIBS}
)

# target_include_directories(alexnet PUBLIC . ${PROJECT_SOURCE_DIR}/Common)
# set_target_properties(alexnet PROPERTIES CUDA_ARCHITECTURES "86")