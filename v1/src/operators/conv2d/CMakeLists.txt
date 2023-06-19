add_library(conv2d INTERFACE)
target_include_directories(conv2d INTERFACE .)
target_link_libraries(conv2d INTERFACE descriptor)