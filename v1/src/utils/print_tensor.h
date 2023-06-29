#pragma once
#include<iostream>
#include<string>


int NC_MAX = 3;
int HW_MAX = 32;

template<typename T>
void PrintVector(const T* ptr, int num) {
    for (int n = 0; n < num; n++) {
        if (n > HW_MAX) {
            std::cout << "row(" << num << ") too large, print " << HW_MAX << " only.";
            break;
        }
        std::cout << ptr[n] << " ";
    }
    std::cout << std::endl;
}


template<typename T>
void PrintMatrix(const T* ptr, int row_num, int col_num, std::string title = "") {
    std::cout << title << std::endl;
    for (int r = 0; r < row_num; r++) {
        if (r > HW_MAX) {
            std::cout << "row(" << row_num << ") too large, print " << HW_MAX << " only." << std::endl;
            break;
        }
        PrintVector(ptr + r * col_num, col_num);
    }
    std::cout << std::endl;
}

template<typename T>
void PrintTensor(const T* const ptr, const int* const dimA, const int dimN, const std::string title) {
    if (dimN > 2) {
        int inner_size = 1; for (int i = 1; i < dimN; i++) inner_size *= dimA[i];
        for (int n = 0; n < dimA[0]; n++) {
            if (n > NC_MAX) {
                std::cout << "dim(" << dimA[0] << ") too large, print partly." << std::endl;
                break;
            }
            PrintTensor(ptr + inner_size * n, dimA + 1, dimN - 1, title + " " + std::to_string(n));
        }
    } else if (dimN == 2) {
        PrintMatrix(ptr, dimA[0], dimA[1], title);
    } else if (dimN == 1) {
        PrintMatrix(ptr, 1, dimA[0], title);
    } else {
        std::cerr << "err at" << __FILE__ << __LINE__ << std::endl;
    }
}
