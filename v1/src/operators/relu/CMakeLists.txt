add_library(relu SHARED relu.cu)
target_include_directories(relu PUBLIC .)

# target_include_directories(multiply PUBLIC include)