find_package(CUDA REQUIRED)
file(GLOB test_network_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.c*)
add_executable(test_network main.cpp ${test_network_files})
target_link_libraries(
  test_network
  network
  loss
  gflags
  GTest::gtest
  ${CUDA_LIBRARIES}
)
include(GoogleTest)
gtest_discover_tests(test_network PROPERTIES DISCOVERY_TIMEOUT 60)