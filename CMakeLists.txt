cmake_minimum_required(VERSION 3.14)
project(PINN_FEM_Coupling)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Obtain the pybind11 cmake directory from pip installation.
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})

# Add subdirectories for the Feel++ module and the bindings module.
add_subdirectory(src/feelpp)
add_subdirectory(src/bindings)
