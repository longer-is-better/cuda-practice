file(GLOB test_op_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu)
add_executable(test_operators main.cu ${test_op_files})
include_directories(${PROJECT_SOURCE_DIR}/Common)
target_link_libraries(
  test_operators
  pool
  relu
  conv2d
  cudnn
  utils
  GTest::gtest_main
)

set_target_properties(test_operators PROPERTIES CUDA_ARCHITECTURES "86")
include(GoogleTest)
gtest_discover_tests(test_operators PROPERTIES DISCOVERY_TIMEOUT 60)