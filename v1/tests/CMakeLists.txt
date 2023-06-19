include(FetchContent)

# 将Googletest下载到本地的build当中，作为本地项目的一部分。所有googletest的目标都将可见。
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(test_operators)
add_subdirectory(test_utils)
# add_subdirectory(networks)
# add_subdirectory(test_operators_copy)