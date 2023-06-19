clear; cmake -S . -B build && cmake --build build && pushd build ; ctest ; pushd
clear; cmake -S . -B build && cmake --build build && build/tests/test_operators/test_operators --gtest_filter=*conv*
clear; cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build && gdb -ex "break test_conv2d.cu:196" --args build/tests/test_operators/test_operators --gtest_filter=*conv*
clear; cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build && gdb -ex "break cuda-practice/v1/src/utils/print_tensor.h:24" --args build/tests/test_operators/test_utils