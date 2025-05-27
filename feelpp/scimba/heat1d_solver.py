import sys
from pathlib import Path

root_dir = Path.cwd().parent
sys.path.insert(0, str(root_dir))

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from feelpp.scimba.heat1d_fem import FemSolver
from feelpp.scimba.heat1d_pinn import PinnSolver

# --- Interface Data Manager ---
class InterfaceData:
    """Holds the shared interface time, temperature (u), and flux (q)."""

    def __init__(self):
        self.t = None    # time points
        self.u = None    # Dirichlet temperature values
        self.q = None    # Neumann flux values

    def update_u(self, t, u, ω=1.0):
        """Update interface temperature u(t) with relaxation factor ω."""
        t, u = np.array(t), np.array(u)
        if self.u is None:
            self.t, self.u = t, u
        else:
            self.u = ω * u + (1 - ω) * self.u

    def update_q(self, t, q, ω=1.0):
        """Update interface flux q(t) with relaxation factor ω."""
        q = np.array(q)
        if self.q is None:
            self.q = q
        else:
            self.q = ω * q + (1 - ω) * self.q

    def poly_expr_q(self, deg=3):
        """Fit a polynomial of degree deg to (t, q) and return a Feel++ expression."""
        coef = np.polyfit(self.t, self.q, deg)[::-1]
        terms = [
            f"{coef[i]:.8g}{('*t^'+str(i)) if i>0 else ''}"
            for i in range(len(coef))
        ]
        return "+".join(terms) + ":t"


# --- Schwarz Coupling Solver ---
class FEM_PINN_Solver:
    """
    Performs Neumann–Dirichlet Schwarz iterations between FEM (left subdomain)
    and PINN (right subdomain) for the 1D heat equation.
    """

    def __init__(self, xm=0.5, ω=0.8, max_iter=6, tol=1e-4,
                 alpha=0.5, export_path="cfpdes-2d.exports/Export.case"):
        # coupling parameters
        self.xm, self.ω, self.max_iter, self.tol = xm, ω, max_iter, tol
        self.alpha = alpha

        # left: FEM solver, right: PINN solver
        self.fem = FemSolver(xm=xm, alpha=alpha)
        self.pinn = PinnSolver(xmin=xm, alpha_range=(alpha-0.1, alpha+0.1))
        self.iface = InterfaceData()
        self.export_path = export_path

        # error history
        self.error_L2 = []
        self.error_H1 = []

    def solve(self):
        """Run Schwarz iterations until convergence or max_iter."""
        print("[Init] 🔄 Starting Neumann–Dirichlet Schwarz iterations")

        # Step 0: initial FEM solve with a large constant flux
        t0, u0 = self.fem.solve(expr_q="10")
        self.iface.update_u(t0, u0, ω=1.0)
        print(f"[Init] FEM provided u_I(t), mean={u0.mean():.3e}")

        # pick 5 representative times for plotting and error metrics
        idx = np.linspace(0, len(t0)-1, 5, dtype=int)
        self.plot_times = t0[idx]

        for k in range(1, self.max_iter+1):
            print(f"\n=== Iteration {k} ===")

            # 1) PINN step: set Dirichlet BC, train, then predict Neumann flux
            self.pinn.set_interface_u(self.iface.t, self.iface.u)
            self.pinn.fit()
            q_pin = self.pinn.predict_interface_q(self.iface.t)
            self.iface.update_q(self.iface.t, q_pin, ω=self.ω)

            # 2) FEM step: use predicted flux as Neumann BC and solve
            expr_q = self.iface.poly_expr_q()
            t_fem, u_fem = self.fem.solve(expr_q=expr_q)
            self.iface.update_u(t_fem, u_fem, ω=self.ω)

            # 3) plot full-domain solution and record L2/H1 errors
            L2_list, H1_list = self._record_and_plot(k)
            print(f" Iter {k}  L2 errors = {np.round(L2_list,6)}")
            print(f" Iter {k}  H1 errors = {np.round(H1_list,6)}")

            # 4) check global convergence against analytical interface solution
            u_ex_I = np.exp(-self.alpha*np.pi**2*self.iface.t) * np.sin(np.pi*self.xm)
            rel_err = np.linalg.norm(self.iface.u - u_ex_I) / np.linalg.norm(u_ex_I)
            print(f" Iter {k}  Interface relative error = {rel_err:.2e}")
            if rel_err < self.tol:
                print("✅ Converged based on interface values.")
                break
        else:
            print("⚠️ Reached max_iter without meeting tolerance.")

        # plot error trends after completion
        self._plot_error_trends()
        return self.iface.t, self.iface.u

    def _read_fem_slice(self, τ):
        """Read FEM solution at time τ on the left domain (x ≤ xm)."""
        reader = pv.get_reader(self.export_path)
        times = np.array(reader.time_values)
        idx = np.argmin(np.abs(times - τ))
        reader.set_active_time_value(float(times[idx]))
        block = reader.read()[0]
        pts  = block.points[:,0]
        vals = block.point_data[list(block.point_data.keys())[0]]
        mask = pts <= self.xm + 1e-8
        xL, uL = pts[mask], vals[mask]
        order = np.argsort(xL)
        return xL[order], uL[order]

    def _record_and_plot(self, iter_id):
        """Plot full-domain solution for this iteration and compute L2/H1 errors."""
        L2_list, H1_list = [], []

        plt.figure(figsize=(8,4))
        for τ in self.plot_times:
            xL, uL = self._read_fem_slice(τ)
            xR_dense = np.linspace(self.xm, 1.0, 500)
            xR, uR = self.pinn.predict_region(xR_dense, τ)

            # merge FEM and PINN solutions, remove duplicates, sort
            x_all = np.hstack([xL, xR])
            u_all = np.hstack([uL, uR])
            uniq_x, idx_uniq = np.unique(x_all, return_index=True)
            order = np.argsort(idx_uniq)
            x_all, u_all = uniq_x[order], u_all[idx_uniq[order]]

            # exact full-domain solution and error metrics
            u_ex = np.exp(-self.alpha*np.pi**2*τ) * np.sin(np.pi*x_all)
            L2 = np.sqrt(np.trapz((u_all - u_ex)**2, x_all))

            du_num = np.gradient(u_all, x_all)
            du_ex  = np.exp(-self.alpha*np.pi**2*τ) * np.pi * np.cos(np.pi*x_all)
            H1 = np.sqrt(np.trapz((du_num - du_ex)**2, x_all))

            L2_list.append(L2)
            H1_list.append(H1)

            plt.plot(x_all, u_all, label=f"t={τ:.3f} (Iter {iter_id})")

        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"Iteration {iter_id}: solutions at selected times")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.error_L2.append(np.array(L2_list))
        self.error_H1.append(np.array(H1_list))
        return np.array(L2_list), np.array(H1_list)

    def _plot_error_trends(self):
        """Plot L2 and H1 error vs. iteration number."""
        its = np.arange(1, len(self.error_L2) + 1)

        # L2 error trend
        plt.figure(figsize=(8,4))
        for i, t_val in enumerate(self.plot_times):
            series = [e[i] for e in self.error_L2]
            plt.plot(its, series, "o-", label=f"t={t_val:.3f}")
        plt.xlabel("Iteration")
        plt.ylabel("L2 error")
        plt.title("L2 error vs. iteration")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # H1 error trend
        plt.figure(figsize=(8,4))
        for i, t_val in enumerate(self.plot_times):
            series = [e[i] for e in self.error_H1]
            plt.plot(its, series, "s--", label=f"t={t_val:.3f}")
        plt.xlabel("Iteration")
        plt.ylabel("H1 error")
        plt.title("H1 error vs. iteration")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
