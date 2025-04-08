cmake_minimum_required(VERSION 3.10)
project(feelpp_interface)

find_package(pybind11 REQUIRED)

# Build the Python module "feelpp_interface" from bindings.cpp.
pybind11_add_module(feelpp_interface bindings.cpp)
