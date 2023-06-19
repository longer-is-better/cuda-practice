project(project_utils)
cmake_minimum_required(VERSION 3.16)

# file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# add_library(utils SHARED ${SRCS})
add_library(utils INTERFACE)
# target_link_libraries(utils INTERFACE cudnn)
target_include_directories(utils INTERFACE . ${PROJECT_SOURCE_DIR}/Common)