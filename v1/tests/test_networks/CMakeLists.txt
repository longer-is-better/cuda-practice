file(GLOB test_network_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu)


find_package(PythonInterp REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd_party/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(test_networks main.cu ${test_network_files})
include_directories(${PROJECT_SOURCE_DIR}/Common)
target_link_libraries(
  test_networks
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  alexnet
  # pool
  # relu
  # conv2d
  # cudnn
  utils
  GTest::gtest_main
)
set_target_properties(test_networks PROPERTIES CUDA_ARCHITECTURES "86")

include(GoogleTest)
gtest_discover_tests(test_networks PROPERTIES DISCOVERY_TIMEOUT 60)