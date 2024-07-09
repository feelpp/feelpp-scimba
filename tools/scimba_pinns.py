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
import sympy as sp
import matplotlib.collections as mcoll

from scimba.equations import domain, pdes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846
ELLIPSOID_A = 4 / 3
ELLIPSOID_B = 1 / ELLIPSOID_A

def differentiate_elements(diff_matrix_str, variables=['x', 'y']):
    # Define the symbols (variables) used in the expression
    symbols = sp.symbols(variables)
    
    # Parse the diff matrix string into a list of sympy expressions
    diff_matrix_str = diff_matrix_str.replace('(', '').replace(')', '')
    expressions_str_list = diff_matrix_str.split(',')
    expressions = [sp.sympify(expr) for expr in expressions_str_list]

    # Compute the derivatives of each expression with respect to each variable
    derivatives = []
    for var in symbols:
        derivs = [sp.diff(expr, var) for expr in expressions]
        derivatives.append(derivs)

    return derivatives

#___________________________________________________________________________________________________________

class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self, space_domain, 
                 rhs = '8*pi*pi*sin(2*pi*x) * sin(2*pi*y)',
                 diff='(1,0,0,1)', 
                 g ='0',
                 u_exact = 'sin(2*pi*x) * sin(2*pi*y)', 
                 grad_u_exact = '(2*pi*cos(2*pi*x) * sin(2*pi*y), 2*pi*sin(2*pi*x) * cos(2*pi*y))'):
        
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=1,
            parameter_domain=[[1.0000, 1.000001]],
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
        g = eval(self.u_exact, {'x': x1, 'y': x2, 'pi': PI, 'sin' : torch.sin, 'cos': torch.cos, 'exp': torch.exp})

        return  u - g


    def residual(self, w, x, mu, **kwargs):
        # Ensure x1 and x2 are PyTorch tensors with requires_grad=True
        x1, x2 = x.get_coordinates()

        # Ensure diff and f are computed correctly
        diff = eval(self.diff, {'x': x1, 'y': x2, 'pi': PI, 'sin': torch.sin, 'cos': torch.cos, 'exp': torch.exp})
        f = eval(self.rhs, {'x': x1, 'y': x2, 'pi': PI, 'sin': torch.sin, 'cos': torch.cos, 'exp': torch.exp})
        
        u_x = self.get_variables(w, "w_x")
        u_y = self.get_variables(w, "w_y")

        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        
        variables = ['x', 'y']
        derivatives = differentiate_elements(self.diff, variables)
        #print(derivatives)
        
        # Evaluate derivatives
        derivatives_eval = []
        for i in range(len(derivatives)):
            row_eval = []
            for j in range(len(derivatives[i])):
                derivative_expr = str(derivatives[i][j])
                eval_expr = eval(derivative_expr, {'x': x1, 'y': x2, 'pi': PI, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp})
                row_eval.append(eval_expr)
            derivatives_eval.append(row_eval)

        # Compute div_grad_u (assuming derivatives_eval and diff are properly aligned)
        div_grad_u_up = (derivatives_eval[0][0]*u_x + diff[0]*u_xx) + (derivatives_eval[0][1]*u_y + diff[1]*u_yy)
        div_grad_u_down = (derivatives_eval[1][0]*u_x + diff[2]*u_xx) + (derivatives_eval[1][1]*u_y + diff[3]*u_yy)
        div_grad_u = div_grad_u_up + div_grad_u_down
        
        return div_grad_u + f
        
        
    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        g = eval(self.u_exact, {'x': x1, 'y': x2, 'pi': PI, 'sin' : torch.sin, 'cos': torch.cos, 'exp': torch.exp})
        return g * w
  
    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        alpha = self.get_parameters(mu)
        return alpha*eval(self.u_exact, {'x': x1, 'y': x2, 'pi': PI, 'sin': torch.sin, 'cos': torch.cos, 'exp': torch.exp})

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

def Run_Poisson2D(pde, epoch=600, bc_loss_bool=True, w_bc=10, w_res=10):
   
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
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers) #, activation_type="sine")
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
    trainer.plot_derivative_mu(n_visu)
    trainer.plot_derivative_xmu(n_visu)

    u = pinn.get_w

    return u

if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn

    xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))
    print(xdomain)
    pde = Poisson_2D(xdomain)
    u = Run_Poisson2D(pde)
    print(u)

    u_exact = 'x*x/(1+x) + y*y/(1+y)'
    rhs = '-(4 + 2*x + 2*y) / ((1+x)*(1+y))'
    pde = Poisson_2D(xdomain, rhs=rhs, diff='(1+x,0,0,1+y)', g='x*x/(1+x) + y*y/(1+y)', u_exact = u_exact )
    u = Run_Poisson2D(pde)
    print(u)

    u_exact = 'x*x + y*y'
    pde = Poisson_2D(xdomain, rhs='-4*x -4*y', diff='(x,y,-y,x+y)', g=u_exact, u_exact = u_exact)
    u = Run_Poisson2D(pde)
    print(u)

    # Example points to evaluate u
    points = [[0.1, 0.2], [0.5, 0.7], [1.0, 1.0]]
    points = torch.tensor(points, dtype=torch.float64)
    labels = torch.zeros(len(points))  # Assuming all points have label 0    
    data = domain.SpaceTensor(points, labels, boundary=True)
    points = points.to(torch.float32)
    mu = torch.ones((len(points), 1), dtype=torch.float64)

    u_values = u(data, mu)
    for point, u_value in zip(points, u_values):
        print(f"u( {point[0:]} ) = {u_value[0]}")

    