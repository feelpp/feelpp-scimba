import sys
# Add the build directory to the Python module search path.
# Adjust this path if your module is generated in a subdirectory.
sys.path.append("/feel/build")
import feelpp_interface

import numpy as np
import torch

def get_pinn_data_from_network(network, points):
    """
    Given a set of points (Nx2 numpy array), use the PINN network to produce predictions.
    """
    with torch.no_grad():
        pts_tensor = torch.tensor(points, dtype=torch.float32)
        u_pred = network(pts_tensor).numpy().flatten()
    return u_pred.tolist()

def send_data_to_feelpp(u_values, points):
    """
    Pass the PINN predicted values and the corresponding points to the C++ FEM module via the PyBind11 interface.
    """
    feelpp_interface.set_interpolation_data(u_values, points.tolist())

if __name__ == "__main__":
    # For demonstration, use a Dummy network.
    class DummyNetwork:
        def __call__(self, x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            return torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)
    network = DummyNetwork()
    points = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    u_values = get_pinn_data_from_network(network, points)
    print("Predicted values from PINN:", u_values)
    send_data_to_feelpp(u_values, points)
