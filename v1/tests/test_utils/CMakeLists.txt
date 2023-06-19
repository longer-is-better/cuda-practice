file(GLOB test_util_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
add_executable(test_utils ${test_util_files})
include_directories(
  ${PROJECT_SOURCE_DIR}/Common
)
target_link_libraries(
  test_utils
  cudnn
  utils
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_utils)