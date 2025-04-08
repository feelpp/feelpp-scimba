
# feelpp-scimba


## 📁 Project Structure

```
project_root/
├── src/
│   ├── feelpp/              # FEM module (mock-up using Feel++ headers)
│   └── bindings/            # PyBind11 interface to expose C++ ↔ Python data transfer
├── python/
│   ├── pinn/                # PINN training and definition using SciMBA
│   ├── interfaces/          # Data exchange: export PINN prediction to C++
│   └── data_processing/     # Postprocessing, visualization, utilities
├── examples/                # Full pipeline launcher script
├── build/                   # Build artifacts (CMake-generated)
```

---

## 📌 Project Goals

- ✅ Use **SciMBA** to train a PINN that solves the 2D Poisson equation
- ✅ Export the trained model's predictions directly to C++ via **PyBind11** for in-memory integration
- ✅ Build a Lagrange interpolant of the PINN solution on a predefined FEM mesh
- ⚠️ Currently compare against a **mock FEM solution** (e.g., a zero vector) — no real finite element solve yet
- 🔜 Lay the foundation for full integration of a **real Feel++ FEM solver**, using variational formulations (e.g., via `coefficientformpde`)


---

## 🔍 Problem Setup

We solve the Poisson equation over the unit square:

$$
-\Delta u = f(x, y) = 2\pi^2 \sin(\pi x)\sin(\pi y), \quad u = 0 \text{ on } \partial\Omega
$$

**Analytical solution:**
$$
u(x, y) = \sin(\pi x)\sin(\pi y)
$$

---

## 🔁 Workflow Overview

```mermaid
graph TD
  A[Train PINN in SciMBA] --> B[Predict on FEM mesh]
  B --> C[Pass prediction via PyBind11]
  C --> D[Construct interpolation in FEM module]
  D --> E[Compare against FEM solution (mock)]
```

---

## 🔧 Technologies

| Task             | Stack |
|------------------|-------|
| PINN Training    | Python + SciMBA + PyTorch |
| FEM Module       | C++ (Feel++ headers ready) |
| Data Interface   | PyBind11 (Python ↔ C++) |
| Build System     | CMake |
| Visualization    | Plotly / Matplotlib |
| Environment      | Docker (based on [Feel++ container](https://github.com/feelpp/docker)) |

---

## 🚀 Build & Run

### 🐳 1. Build the image (optional)

If you're using a custom Docker container:

```bash
docker build -t feelpp-scimba .
```

(Dockerfile should be based on: `ghcr.io/feelpp/feelpp:jammy` and install `scimba`, `pybind11`, `torch`, etc.)

---

### ▶️ 2. Run the full pipeline

In the project root:

```bash
./examples/run_full_example.sh
```

Example output:
```
load network: /feel/networks/poisson_square.pth
network loaded
L2 error between FEM and PINN interpolation: 0.45
```
