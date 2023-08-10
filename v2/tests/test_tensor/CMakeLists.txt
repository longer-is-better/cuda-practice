find_package(CUDA REQUIRED)
file(GLOB test_tensor_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.c*)
add_executable(test_tensor main.cpp ${test_tensor_files})
target_link_libraries(
  test_tensor
  tensor
  gflags
  GTest::gtest
  ${CUDA_LIBRARIES}
)
include(GoogleTest)
gtest_discover_tests(test_tensor PROPERTIES DISCOVERY_TIMEOUT 60)