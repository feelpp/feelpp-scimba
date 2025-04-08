#!/bin/bash

# Step 1: Build the C++ project (FEM solver and PyBind11 module).
mkdir -p build && cd build
cmake -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) ..
make -j$(nproc)
cd ..

# Step 2: Run the PINN training script (this will train or load the network).
python3 python/pinn/train_pinn.py

# Step 3: Execute the interface script to pass PINN data to the FEM module.
python3 python/interfaces/get_interpolation_data.py

# Step 4: Run the FEM solver executable.
./build/src/feelpp/feelpp_solver
