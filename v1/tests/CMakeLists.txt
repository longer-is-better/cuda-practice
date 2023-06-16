include(FetchContent)

# 将Googletest下载到本地的build当中，作为本地项目的一部分。所有googletest的目标都将可见。
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_subdirectory(test_operators)
# add_subdirectory(networks)