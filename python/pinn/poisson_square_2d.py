import torch
from scimba.equations import pdes

class PoissonSquare2D(pdes.AbstractPDEx):
    def __init__(self, space_domain):
        super().__init__(
            nb_unknowns=1,         # Single unknown u(x,y)
            space_domain=space_domain,
            nb_parameters=0        # No extra parameters for f
        )
        # Enable automatic differentiation for first and second derivatives.
        self.first_derivative = True
        self.second_derivative = True

    def residual(self, w, x, mu=None, **kwargs):
        """Strong form residual: -Δu - f(x,y) = 0, where f=2*pi^2*sin(pi*x)*sin(pi*y)"""
        x1, x2 = x.get_coordinates()
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = 2 * torch.pi**2 * torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)
        return -(u_xx + u_yy) - f

    def bc_residual(self, w, x, mu=None, **kwargs):
        """Boundary residual: u = 0"""
        return self.get_variables(w)

    def reference_solution(self, x, mu=None):
        """Analytical solution: sin(pi*x)*sin(pi*y)"""
        x1, x2 = x.get_coordinates()
        return torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)
