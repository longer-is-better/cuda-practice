add_library(pool INTERFACE)
target_include_directories(pool INTERFACE .)
target_link_libraries(pool INTERFACE descriptor)