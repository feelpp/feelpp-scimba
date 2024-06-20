from pathlib import Path

import scimba.nets.training_tools as training_tools
import scimba.pinns.pinn_losses as pinn_losses
import scimba.pinns.pinn_x as pinn_x
import scimba.pinns.training_x as training_x
import scimba.sampling.sampling_parameters as sampling_parameters
import scimba.sampling.sampling_pde as sampling_pde
import scimba.sampling.uniform_sampling as uniform_sampling

import torch
import pyvista as pv 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll

from scimba.equations import domain, pdes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846
ELLIPSOID_A = 4 / 3
ELLIPSOID_B = 1 / ELLIPSOID_A

#___________________________________________________________________________________________________________

class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self, space_domain, 
                 rhs = '8*pi*pi*sin(2*pi*x) * sin(2*pi*y)',
                 diff='(1,0,0,1)', 
                 g ='0',
                 u_exact = 'sin(2*pi*x) * sin(2*pi*y)'):
        
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=1,
            parameter_domain=[[1.0000, 1.00001]],
        )
        
        self.diff = diff
        self.g = g
        self.rhs = rhs
        self.u_exact = u_exact

        self.first_derivative = True
        self.second_derivative = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        u = self.get_variables(w)
        x1, x2 = x.get_coordinates()
        g = eval(self.u_exact, {'x': x1, 'y': x2, 'pi': PI, 'sin' : torch.sin, 'cos': torch.cos})

        return  g

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")

        diff = eval(self.diff, {'x': x1, 'y': x2, 'pi': PI, 'sin' : torch.sin, 'cos': torch.cos})
        f = eval(self.rhs, {'x': x1, 'y': x2, 'pi': PI, 'sin': torch.sin, 'cos': torch.cos})
        
        return u_xx* diff[0] + u_yy* diff[3] + f
    
    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        g = eval(self.u_exact, {'x': x1, 'y': x2, 'pi': PI, 'sin' : torch.sin, 'cos': torch.cos})
        return g * w
  
    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        alpha = self.get_parameters(mu)
        return alpha*eval(self.u_exact, {'x': x1, 'y': x2, 'pi': PI, 'sin': torch.sin, 'cos': torch.cos})

#___________________________________________________________________________________________________________

class PoissonDisk2D(pdes.AbstractPDEx):
    def __init__(self, space_domain):
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=1,
            parameter_domain=[[1.0000, 1.00001]],
        )

        self.first_derivative = True
        self.second_derivative = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        return self.get_variables(w)

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = self.get_parameters(mu)
        return u_xx + u_yy + f

    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        x1_0, x2_0 = self.space_domain.large_domain.center
        f = self.get_parameters(mu)
        return 0.25 * f * (1 - (x1 - x1_0) ** 2 - (x2 - x2_0) ** 2)
    
#___________________________________________________________________________________________________________

def Run_laplacian2D(pde, epoch=100, bc_loss_bool=True, w_bc=10, w_res=10):
   
    # Initialize samplers
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = "test.pth"
    new_training = True

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    # Define network architecture and losses
    tlayers = [20, 20, 20, 20, 20]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="sine")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=bc_loss_bool, w_res=w_res, w_bc=w_bc
    )
    optimizers = training_tools.OptimizerData(learning_rate=1.2e-2, decay=0.99)

    # Initialize TrainerPINNSpace
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=5000,
    )

    # Perform training
    if not bc_loss_bool:
        if new_training:
            trainer.train(epochs=epoch, n_collocation=5000, n_data=0)
    else:
        if new_training:
            trainer.train(
                epochs=epoch, n_collocation=5000, n_bc_collocation=1000, n_data=0
            )

    # Plot and print the coordinates and values of u
    n_visu = 20000
    reference_solution = True
    trainer.plot(n_visu, reference_solution=True)
    u = trainer.network.setup_w_dict

    return u

if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn

    xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))
    print(xdomain)
    pde = Poisson_2D(xdomain)
    u = Run_laplacian2D(pde)
    print(u)
    # Example points to evaluate u
    points = [[0.1, 0.2], [0.5, 0.7], [1.0, 1.0]]
    points = torch.tensor(points, dtype=torch.float32)
    labels = torch.zeros(len(points))  # Assuming all points have label 0    
    data = domain.SpaceTensor(points, labels, boundary=True)
    points = points.to(torch.float32)
    mu = torch.ones((len(points), 1), dtype=torch.float32)

    u_values = u(data, mu)
    for point, u_value in zip(points, u_values):
        print(f"Point: {point}, u value: {u_value}")