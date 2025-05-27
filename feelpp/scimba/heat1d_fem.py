import numpy as np
import pyvista as pv
import feelpp.core as fppc
from feelpp.toolboxes.cfpdes import *

# --- Geometry ---

def _generate_geo(fname: str, x0: float, x1: float, eps: float, h: float):
    """Make a thin 2D rectangle for 1D heat conduction."""
    geo = f"""SetFactory("OpenCASCADE");
h = {h};
Rectangle(1)={{ {x0}, {-eps/2}, 0, {x1-x0}, {eps}, 0}};
Characteristic Length{{ PointsOf{{ Surface{{1}}; }} }} = h;
Physical Curve("Gamma_L")={{4}};   // x = x0
Physical Curve("Gamma_I")={{2}};   // x = x1
Physical Surface("Omega")={{1}};
"""
    with open(fname, "w") as f:
        f.write(geo)


# --- JSON helper ---

def _make_json(order: int, alpha: float, expr_q: str | None):
    """Build Feel++ JSON: u=0 on left, flux q(t) on interface."""
    dirichlet = [
        {"variable": "u", "markers": ["Gamma_L"], "expr": "0."},
    ]

    neumann_expr = expr_q if expr_q is not None else "0."
    neumann = [
        {"variable": "u", "markers": ["Gamma_I"], "expr": neumann_expr}
    ]

    return {
        "Name": "Heat1D",
        "Models": {
            "cfpdes-2d": {"equations": "AdvDiff"},
            "AdvDiff": {
                "setup": {
                    "unknown":   {"basis": f"Pch{order}", "name": "u", "symbol": "u"},
                    "coefficients": {
                        "d": "1.",       # diffusion
                        "c": f"{alpha}", # reaction
                        "f": "0."        # source
                    },
                }
            },
        },
        "Materials":   {"Omega": {"markers": ["Omega"]}},
        "BoundaryConditions": {
            "AdvDiff": {"Dirichlet": dirichlet, "Neumann": neumann}
        },
        "InitialConditions": {
            "AdvDiff": {
                "u": {
                    "Expression": {
                        "initial": {"markers": ["Omega"], "expr": "sin(pi*x):x:y"}
                    }
                }
            }
        },
        "PostProcess": {"cfpdes-2d": {"Exports": {"fields": ["all"]}}},
    }


# --- FEM solver class ---

class FemSolver:
    """Left subdomain [0, xm] with heat flux q(t) at the interface."""

    def __init__(self, xm=0.5, h=0.005, eps=0.01, order=1, alpha=0.5):
        self.xm, self.h, self.eps = xm, h, eps
        self.order, self.alpha = order, alpha
        self._geo = "fem.geo"
        self._cfg = "fem.cfg"

    def _write_cfg(self, dt=0.002, T=0.03):
        """Write the Feel++ config file."""
        cfg = f"""
case.dimensions=2

[cfpdes]
pc-type=gamg
reuse-prec=1

[cfpdes.AdvDiff.bdf]
order=1

[ts]
time-step={dt}
time-final={T}
restart.at-last-save=true
"""
        with open(self._cfg, "w") as f:
            f.write(cfg)

    def solve(self, expr_q: str | None = "0"):
        """
        Run the heat equation with interface flux expr_q.
        Returns times and interface temperatures.
        """
        # prep files
        _generate_geo(self._geo, 0.0, self.xm, self.eps, self.h)
        self._write_cfg()
        fppc.Environment.setConfigFile(self._cfg)

        # init model
        model = cfpdes(dim=2, keyword="cfpdes-2d")
        mesh  = fppc.load(fppc.mesh(dim=2, realdim=2), self._geo, self.h)
        model.setMesh(mesh)
        model.setModelProperties(_make_json(self.order, self.alpha, expr_q))

        model.init(buildModelAlgebraicFactory=True)
        model.exportResults()
        model.startTimeStep()

        while not model.timeStepBase().isFinished():
            model.solve()
            model.exportResults()
            model.updateTimeStep()

        # read interface temp
        reader = pv.get_reader("cfpdes-2d.exports/Export.case")
        times = np.array(reader.time_values)
        uI = []

        for t in times:
            reader.set_active_time_value(t)
            block = reader.read()[0]
            data = block.point_data[list(block.point_data.keys())[0]]
            pts  = block.points[:, 0]
            uI.append(data[np.isclose(pts, self.xm)].mean())

        return times, np.array(uI)
