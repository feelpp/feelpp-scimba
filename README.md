# feelpp-scimba

Integration of [ScimBa](https://pypi.org/project/scimba/) and [Feel++](https://docs.feelpp.org/) to streamline data exchange and combine machine-learning workflows with high-performance finite-element PDE solving.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation with uv](#installation-with-uv)  
3. [Using the Docker Image](#using-the-docker-image)  
4. [Quickstart Example](#quickstart-example)  
5. [Project Status](#project-status)  
6. [Contact](#contact)  

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
import feelpp
import feelpp.toolboxes.core as tb
from feelpp_scimba.Poisson import Poisson

# Initialize Feel++
sys.argv = ["feelpp_app"]
env = feelpp.Environment(
    sys.argv,
    opts=tb.toolboxes_options("coefficient-form-pdes", "cfpdes"),
    config=feelpp.localRepository("feelpp_cfpde")
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

## Project Status

This project is under active development. Contributions and feedback are very welcome!

---

## Contact

- **Christophe Prud’homme** — <christophe.prudhomme@cemosis.fr>  
- **Rayen Tlili** — <rayen.tlili@etu.unistra.fr>  
- **Repository** — https://github.com/feelpp/feelpp-scimba
