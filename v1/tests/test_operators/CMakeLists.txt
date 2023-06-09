

# 将下载下来的googletest编译成库 使用IMPORTED关键字
add_library(GTest::GTest INTERFACE IMPORTED)
# 链接gtest
target_link_libraries(GTest::GTest INTERFACE gtest_main)


# 将我们写的tests编译成可执行文件
add_executable(test_relu test_relu.cu)

# 把GTest和被测函数都链接到 可执行文件 上
target_link_libraries(test_relu
 PRIVATE
  GTest::GTest
  relu)

# 增加测试
add_test(gtest_operators test_relu)