# heat1d_pinn.py

import torch
from scimba.equations.domain import SpaceDomain, SquareDomain, SpaceTensor
from scimba.equations.pdes import AbstractPDEtx
from scimba.sampling import sampling_pde, sampling_ode, sampling_parameters, uniform_sampling
from scimba.pinns import pinn_tx, pinn_losses, training_tx
from scimba.nets import training_tools

import numpy as np

# --------------------------------------------------
# ParamHeatPDE: defines the 1D heat equation for PINN training
# --------------------------------------------------
class ParamHeatPDE(AbstractPDEtx):
    """1D Heat PDE: u_t = α u_xx, with time‐varying Dirichlet on left and zero on right."""

    def __init__(self, xmin, xmax, t_bc, u_bc,
                 tdomain=(0.0, 0.03),   # time interval [t0, t_final]
                 p_domain=((0.4, 0.6),) # α ∈ [0.4, 0.6]
                 ):
        # spatial bounds
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.length = self.xmax - self.xmin

        # build the space domain object
        xdomain = SpaceDomain(1, SquareDomain(1, [[self.xmin, self.xmax]]))
        super().__init__(
            nb_unknowns=1,           # u(x,t) only
            time_domain=list(tdomain),
            space_domain=xdomain,
            nb_parameters=1,         # thermal diffusivity α
            parameter_domain=p_domain
        )

        # request derivatives needed in residuals
        self.first_derivative_t  = True   # need u_t
        self.second_derivative_t = False  # no u_tt
        self.first_derivative_x  = True   # need u_x
        self.second_derivative_x = True   # need u_xx

        # set up boundary‐condition data
        self.update_bc(t_bc, u_bc)

    def residual(self, w, t, x, mu, **kwargs):
        """
        PDE residual: u_t - α u_xx == 0
        w: solution dict containing derivatives
        mu: parameter tensor for α
        """
        alpha = self.get_parameters(mu)
        u_t   = self.get_variables(w, "w_t")
        u_xx  = self.get_variables(w, "w_xx")
        return u_t - alpha * u_xx

    def bc_residual(self, w, t, x, mu, **kwargs):
        """
        Boundary residual mixes Dirichlet at xmin and homogeneous at xmax:
          u(xmin,t) = f(t)   and   u(xmax,t) = 0
        We mask points at each boundary.
        """
        coords = x.get_coordinates().view(-1)
        u_val  = self.get_variables(w).view(-1)

        # identify left / right boundary points
        mask_L = ((coords - self.xmin).abs() < 1e-8).float()
        mask_R = ((coords - self.xmax).abs() < 1e-8).float()

        # build target f(t) at xmin
        f_t = torch.zeros_like(u_val)
        if mask_L.any():
            t_flat = t.view(-1)
            idxs   = mask_L.bool()
            f_t[idxs] = self._interp_f(t_flat[idxs])

        # residual = mask_L*(u - f) + mask_R*(u - 0)
        res = mask_L * (u_val - f_t) + mask_R * u_val
        return res.view(-1, 1)

    @torch.no_grad()
    def _interp_f(self, t_q):
        """
        Linear interpolation of boundary data (t_bc, u_bc):
        clamp queries, find bracketing indices, and lerp.
        """
        t_q = t_q.clamp(self.t_bc[0], self.t_bc[-1])
        idx  = torch.searchsorted(self.t_bc, t_q)

        idx1 = idx.clamp(1, len(self.t_bc) - 1)
        idx0 = idx1 - 1

        t0, t1 = self.t_bc[idx0], self.t_bc[idx1]
        u0, u1 = self.u_bc[idx0], self.u_bc[idx1]
        w = (t_q - t0) / (t1 - t0 + 1e-12)  # avoid /0
        return u0 + w * (u1 - u0)

    def initial_condition(self, x, mu, **kwargs):
        """Initial temp: sin(π x) over the domain."""
        coords = x.get_coordinates()
        return torch.sin(torch.pi * coords).view(-1, 1)

    def post_processing(self, t, x, mu, w):
        """
        Enforce BCs analytically:
          u = base(x,t) + bump(x)*NN(x,t)
        base satisfies Dirichlet at xmin and default at xmax.
        bump vanishes at both ends.
        """
        coords = x.get_coordinates().view(-1, 1)
        t_flat = t.view(-1, 1)
        L      = self.length

        # weight → 1 at xmin, 0 at xmax
        weight_left = (self.xmax - coords) / L

        # interpolate f(t) at left
        f_t = self._interp_f(t_flat.view(-1)).view(-1, 1)
        f0  = self.u_bc[0]  # initial left‐BC value

        # base func: sin(πx) + weight*(f(t)-f0)
        base = torch.sin(torch.pi * coords) + weight_left * (f_t - f0)

        # bump = 0 at xmin & xmax
        bump = (coords - self.xmin) * (self.xmax - coords)

        return base + bump * w

    def reference_solution(self, t, x, mu):
        """
        Analytical solution for verification (0–1 domain, sin(π x) IC):
          u = exp(-α π² t) sin(π x)
        """
        coords = x.get_coordinates().view(-1)
        alpha  = self.get_parameters(mu).view(-1)
        u = torch.exp(-alpha * torch.pi**2 * t.view(-1)) * torch.sin(torch.pi * coords)
        return u.view(-1, 1)

    def update_bc(self, t_bc, u_bc):
        """Sort and store BC arrays as tensors."""
        t_bc = torch.as_tensor(t_bc, dtype=torch.get_default_dtype())
        u_bc = torch.as_tensor(u_bc, dtype=torch.get_default_dtype())
        idx  = torch.argsort(t_bc)
        self.t_bc, self.u_bc = t_bc[idx], u_bc[idx]


# --------------------------------------------------
# PinnSolver: wraps training/prediction for interface u and flux q
# --------------------------------------------------
class PinnSolver:
    """Trainable PINN that returns u(xmin,t) and q=-∂u/∂x there."""

    def __init__(self, xmin=0.5, xmax=1.0,
                 alpha_range=(0.4, 0.6),
                 n_colloc=4000, epochs=100):
        # PDE object with placeholder BC
        self.pde = ParamHeatPDE(
            xmin, xmax,
            t_bc=[0.0], u_bc=[0.0],
            tdomain=(0.0, 0.03),
            p_domain=(alpha_range,)
        )

        # network: MLP with sinusoidal activations
        net = pinn_tx.MLP_tx(
            pde=self.pde,
            layer_sizes=[20, 80, 80, 80, 20, 10],
            activation_type="sine"
        )
        self.pinn = pinn_tx.PINNtx(net, self.pde).float()

        # samplers for residual, BC, initial
        t_s  = sampling_ode.TSampler(uniform_sampling.UniformSampling, ode=self.pde)
        x_s  = sampling_pde.XSampler(pde=self.pde)
        mu_s = sampling_parameters.MuSampler(uniform_sampling.UniformSampling, model=self.pde)
        sampler = sampling_pde.PdeTXCartesianSampler(t_s, x_s, mu_s)

        # loss weights
        losses = pinn_losses.PinnLossesData(
            w_res=1.0, init_loss_bool=True,
            w_init=1.0, w_bc=1.0, bc_loss_bool=True
        )

        # optimizer settings
        optim = training_tools.OptimizerData(learning_rate=9e-3, decay=0.99)

        # trainer config
        self.trainer = training_tx.TrainerPINNSpaceTime(
            pde=self.pde,
            network=self.pinn,
            losses=losses,
            optimizers=optim,
            sampler=sampler,
            file_name="pinn_nd.pth",
            batch_size=n_colloc
        )
        self.trainer.pre_training = False

        # store training settings
        self.n_colloc, self.epochs = n_colloc, epochs

    def set_interface_u(self, t_arr, u_arr):
        """Override interface Dirichlet BC u(xmin,t)."""
        self.pde.update_bc(t_arr, u_arr)

    def fit(self):
        """Train PINN on PDE, IC, BC losses only."""
        self.trainer.train(
            epochs=self.epochs,
            n_collocation=self.n_colloc,
            n_bc_collocation=max(1, self.n_colloc // 10),
            n_init_collocation=max(1, self.n_colloc // 10),
            n_data=0
        )

    def _prepare_tensors(self, t_arr, x_val, mu_val=None):
        """Build time, space, and parameter tensors for prediction."""
        device = next(self.pinn.parameters()).device

        t_np = np.asarray(t_arr, dtype=np.float32).ravel()
        n    = t_np.size
        ts   = torch.from_numpy(t_np).view(n, 1).to(device)

        x_space = self.trainer.sampler.sampling_x(n)
        x_space.x = torch.full((n,1), float(x_val),
                               dtype=torch.float32, device=device)

        if mu_val is None:
            mu_val = float(sum(self.pde.parameter_domain[0]) / 2.0)
        mus = torch.full((n,1), mu_val,
                         dtype=torch.float32, device=device)

        return ts, x_space, mus

    def predict_interface_u(self, t_arr, mu_val=None):
        """
        Compute u(xmin,t) via PINN + post‐processing.
        Returns NumPy array of u at interface.
        """
        ts, x_space, mus = self._prepare_tensors(t_arr, self.pde.xmin, mu_val)
        with torch.no_grad():
            w_dict = self.trainer.network.setup_w_dict(ts, x_space, mus)
            u_pred = self.pde.post_processing(ts, x_space, mus, w_dict["w"])[:,0]
        return u_pred.cpu().numpy()

    def predict_interface_q(self, t_arr, mu_val=None, eps=1e-10):
        """
        Approximate flux q = -∂u/∂x using forward FD:
          q ≈ (u(x+ε)-u(x)) / ε
        """
        u0 = self.predict_interface_u(t_arr, mu_val)

        # shift evaluation point by eps
        orig = self.pde.xmin
        self.pde.xmin = orig + eps
        u1 = self.predict_interface_u(t_arr, mu_val)
        self.pde.xmin = orig

        return (u1 - u0) / eps

    def predict_region(self, x_arr, t_scalar, mu_val=None):
        """
        Get u(x,t_scalar) over x_arr:
          returns sorted (x_arr, u_arr) for plotting.
        """
        import numpy as _np, torch as _t

        N      = len(x_arr)
        device = next(self.pinn.parameters()).device

        ts = _t.full((N,1), float(t_scalar),
                     dtype=_t.float32, device=device)

        x_space = self.trainer.sampler.sampling_x(N)
        coords  = _t.from_numpy(_np.asarray(x_arr, np.float32)).view(-1,1).to(device)
        x_space.x = coords

        if mu_val is None:
            mu_val = float(sum(self.pde.parameter_domain[0]) / 2.0)
        mus = _t.full((N,1), mu_val,
                     dtype=_t.float32, device=device)

        with _t.no_grad():
            w_dict = self.trainer.network.setup_w_dict(ts, x_space, mus)
            u_pred = self.pde.post_processing(ts, x_space, mus, w_dict["w"])[:,0]

        x_np = coords.detach().cpu().numpy().ravel()
        u_np = u_pred.detach().cpu().numpy().ravel()
        idx  = _np.argsort(x_np)
        return x_np[idx], u_np[idx]
