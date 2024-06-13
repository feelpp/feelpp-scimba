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
                 rhs='-2 * (-6 + x * (2 - 6 * y) + 3 * y - 8 * y*y + 2 * y*y*y + x*x * (-2 + 6 * y))',
                 diff='(1,0,0,1)', 
                 g ='y*y*y * (1 - y) - 2 * y*y * ((y - 1) * x * (1 - x)) + 6 * y * (1 - y)',
                 u_exact = 'y*y*y * (1 - y) - 2 * y*y * ((y - 1) * x * (1 - x)) + 6 * y * (1 - y)'):
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

        return u - g

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



def Run_laplacian2D(pde, epoch = 100, bc_loss_bool=True, w_bc=10, w_res=10):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = "test.pth"
    #new_training = False
    new_training = True

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    tlayers = [20, 20, 20, 20, 20]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="sine")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=bc_loss_bool, w_res=w_res, w_bc=w_bc
    )
    optimizers = training_tools.OptimizerData(learning_rate=1.2e-2, decay=0.99)
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=5000,
    )

    if not bc_loss_bool:
        if new_training:
            trainer.train(epochs=epoch, n_collocation=5000, n_data=0)
    else:
        if new_training:
            trainer.train(
                epochs=epoch, n_collocation=5000, n_bc_collocation=1000, n_data=0
            )

    trainer.plot(20000, reference_solution=True)
    # trainer.plot_derivative_mu(n_visu=20000)
    return network, pde


if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn
    xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))
    print(xdomain)
    pde = Poisson_2D(xdomain)
    network, pde = Run_laplacian2D(pde)

    u_exact = 'y + (x*(1-x) + y*(1-y)/4) '
    pde = Poisson_2D(xdomain, rhs='5/2', g='y', u_exact = u_exact)
    network, pde = Run_laplacian2D(pde)
    """
    u_exact = 'sin(2*pi*x) * sin(2*pi*y)'
    rhs = '8*pi*pi*sin(2*pi*x) * sin(2*pi*y)'
    pde = Poisson_2D(xdomain, rhs=rhs, g='0', u_exact = u_exact)
    network, pde = Run_laplacian2D(pde)

    # Laplacian on circle with nn
    xdomain = domain.SpaceDomain(2, domain.DiskBasedDomain(2, center=[0.0, 0.0], radius=1.0))
    
    pde = PoissonDisk2D(xdomain)
    network, pde = Run_laplacian2D(pde)
    
    u_exact =  '1/4*(1 - x*x - y*y)'
    rhs = '1'
    pde_disk = Poisson_2D(xdomain,  rhs= rhs, g= '0', u_exact=u_exact)
    network, pde = Run_laplacian2D(pde_disk)

    u_exact =  'sin(pi*(x*x + y*y))'
    rhs = '4*pi*sin(pi*(x*x + y*y)) - 4*pi*pi*(x*x + y*y)*cos(pi*(x*x + y*y))'
    pde_disk = Poisson_2D(xdomain,  rhs= rhs, g= '0', u_exact=u_exact)
    network, pde = Run_laplacian2D(pde_disk)
    """

    u_scimba = network.forward

    # File path to the .case file
    file_path = '/workspaces/2024-stage-feelpp-scimba/feelppdb/feelpp_cfpde/np_1/cfpdes-2d-p1.exports/Export.case'

    # Read the .case file using PyVista
    data = pv.read(file_path)

    # Iterate over each block in the dataset to find coordinates
    coordinates = None
    for i, block in enumerate(data):
        if block is None:
            continue

        print(f"Block {i}:")
        print(block)
        
        # Extract the mesh points (coordinates)
        coordinates = block.points

    # Ensure coordinates are found
    if coordinates is None:
        raise ValueError("No coordinates found in the mesh blocks.")

    # Print the first few coordinates to understand their structure
    print("First few coordinates:")
    print(coordinates[:5])

    # Determine the number of features
    num_features = coordinates.shape[1]
    print(f"Number of features in coordinates: {num_features}")

    # If there are more features than expected, strip the extra ones
    if num_features > 2:
        coordinates = coordinates[:, :2]  # Keep only the first two features (x, y)

    # Convert coordinates to a PyTorch tensor
    input_tensor = torch.tensor(coordinates, dtype=torch.double)
    print(f"Shape of input tensor (coordinates): {input_tensor.shape}")

    # Create the mu tensor with correct shape
    mu_value = 1  # Example value for mu
    mu = torch.full((input_tensor.size(0), 1), mu_value, dtype=torch.double)
    print(f"Shape of mu tensor: {mu.shape}")

    # Pass input tensor and mu tensor separately to the network
    solution_tensor = u_scimba(input_tensor, mu)

    # Convert the tensor to a NumPy array
    solution_array = solution_tensor.detach().numpy()

    # Print solution array
    print("Solution array:")
    print(solution_array)