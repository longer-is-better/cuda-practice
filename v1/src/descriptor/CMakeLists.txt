file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)
add_library(descriptor SHARED ${SRCS})
target_link_libraries(descriptor utils)
target_include_directories(descriptor INTERFACE . ${PROJECT_SOURCE_DIR}/Common)