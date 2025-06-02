# feelpp-scimba

Integration of [ScimBa](https://pypi.org/project/scimba/) and [Feel++](https://docs.feelpp.org/) to streamline data exchange and combine machine-learning workflows with high-performance finite-element PDE solving.

---

## Table of Contents

- [feelpp-scimba](#feelpp-scimba)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation with uv](#installation-with-uv)
  - [Using the Docker Image](#using-the-docker-image)
  - [Quickstart Example](#quickstart-example)
  - [Mixed FEM–PINN Schwarz solver](#mixed-fempinn-schwarz-solver)
  - [Project Status](#project-status)
  - [Contact](#contact)

---

## Prerequisites

- **Feel++** installed on your host (e.g. `apt install libfeelpp-all-dev` on Ubuntu).  
- **Python 3.9+** (uv can download and manage Python for you).  
- **uv** (a drop-in replacement for `pip`, `virtualenv`, and more):  
  ```bash
  # via pip
  pip install --user uv
  # or via the official install script
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```  
- **Git**  

---

## Installation with uv

1. **Clone the repository**  
   ```bash
   git clone https://github.com/feelpp/feelpp-scimba.git
   cd feelpp-scimba
   ```

2. **Create a virtual environment** (with access to system-wide Feel++ packages)  
   ```bash
   uv venv --system-site-packages .venv
   ```

3. **Activate the environment**  
   ```bash
   source .venv/bin/activate
   ```

4. **Install the Python package and its dependencies**  
   ```bash
   uv pip install -e .
   ```

5. **Install all dependencies including for testing**
   ```bash
   uv pip install -e .[test]
   ```

6. **Run tests**
   ```bash
   pytest
   ```

---

## Using the Docker Image

If you prefer containerization, you can build and run our Dockerfile:

```bash
# Build
docker build -t feelpp_scimba:latest .

# Run interactively
docker run --rm -it feelpp_scimba:latest
```

Inside the container you’ll have Feel++, ScimBa, and all Python dependencies ready to go.

---

## Quickstart Example

```python
import sys
import feelpp.core as fppc
import feelpp.toolboxes.core as tb
from feelpp.scimba.Poisson import Poisson

# Initialize Feel++
sys.argv = ["feelpp_app"]
env = feelpp.Environment(
    sys.argv,
    opts=tb.toolboxes_options("coefficient-form-pdes", "cfpdes"),
    config=fppc.localRepository("feelpp_cfpde")
)

# Solve a 2D Poisson problem
P = Poisson(dim=2)
P(
    h=0.05, 
    order=1,
    name="u",
    rhs="8*pi*pi*sin(2*pi*x)*sin(2*pi*y)",
    diff="{1,0,0,1}",
    g="0",
    shape="Rectangle",
    plot=1,
    solver="feelpp",
    u_exact="sin(2*pi*x)*sin(2*pi*y)",
    grad_u_exact="{2*pi*cos(2*pi*x)*sin(2*pi*y),2*pi*sin(2*pi*x)*cos(2*pi*y)}"
)
```

---

## Mixed FEM–PINN Schwarz solver

This script demonstrates the use of our mixed FEM–PINN Schwarz solver to compute the solution of the one-dimensional heat conduction equation:

$$
\frac{\partial u(x,t)}{\partial t}
- \alpha\,\frac{\partial^2 u(x,t)}{\partial x^2}
= 0,\quad
x \in [0,1],\; t \in [0,T],
$$
subject to
- **Dirichlet boundary conditions**: $u(0,t)=0$, $u(1,t)=0$,
- **Initial condition**: $u(x,0)=\sin(\pi x)$.

The domain $[0,1]$ is split at an interface $x_m$:
- **Left subdomain** $[0, x_m]$: solved by a FEM (Feel++) solver.
- **Right subdomain** $[x_m, 1]$: solved by a PINN (SciMBA) solver.

These two solvers are coupled via a Neumann–Dirichlet Schwarz iteration.

**Key Parameters**

| Parameter  | Description                                                                                               |
|------------|-----------------------------------------------------------------------------------------------------------|
| `xm`       | Interface location $x_m$ dividing the domain: FEM on $[0,x_m]$, PINN on $[x_m,1]$.                  |
| `ω`        | Relaxation factor ($0<\omega<1$) used to stabilize and accelerate the Schwarz iteration.               |
| `max_iter` | Maximum number of Schwarz iterations to perform before stopping.                                          |
| `tol`      | Convergence tolerance: stops when $\frac{\|u_I^{(k+1)} - u_I^{(k)}\|_{L^2}}{\|u_I^{(k)}\|_{L^2}} < \text{tol}$.                    |
| `alpha`    | Thermal diffusivity $\alpha$ in the heat conduction equation.                                           |

```python
import sys
import feelpp.core as fppc
import feelpp.toolboxes.core as tb
from feelpp.scimba.heat1d_solver import FEM_PINN_Solver

sys.argv = ["Heat1D"]
env = fppc.Environment(
    sys.argv, 
    opts=tb.toolboxes_options("coefficient-form-pdes", "cfpdes"), 
    config=fppc.localRepository("Heat1D-repo")
)

solver = FEM_PINN_Solver(xm=0.4, ω=0.8, max_iter=5, tol=1e-4, alpha=0.5)
t, u = solver.solve()
```

---

## Project Status

This project is under active development. Contributions and feedback are very welcome!

---

## Contact

- **Christophe Prud’homme** — <christophe.prudhomme@cemosis.fr>  
- **Rayen Tlili** — <rayen.tlili@etu.unistra.fr>  
- **Repository** — https://github.com/feelpp/feelpp-scimba
