cmake_minimum_required(VERSION 3.12)
project(cnn_verification_project)

# Tìm Python và Pybind11
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# Thêm thư mục include của DUT
include_directories(dut_cpp/include)

# Tạo một thư viện tĩnh cho logic DUT
add_library(cnn_dut STATIC
    dut_cpp/src/cnn_dut.cpp
)

# Tạo module Pybind11
pybind11_add_module(cnn_dut_module
    pybind_wrapper/wrapper.cpp
)

# Liên kết module với thư viện DUT
target_link_libraries(cnn_dut_module PRIVATE cnn_dut)