file(GLOB test_op_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu)
add_executable(test_operators main.cu ${test_op_files})
include_directories(${PROJECT_SOURCE_DIR}/Common)
target_link_libraries(
  test_operators
  relu
  conv2d
  cudnn
  utils
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_operators PROPERTIES DISCOVERY_TIMEOUT 60)