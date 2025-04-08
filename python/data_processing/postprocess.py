import numpy as np
import plotly.graph_objects as go

def compute_l2_error(u_ref, u_pred):
    """Compute the L2 error between two arrays."""
    return np.sqrt(np.mean((u_ref - u_pred)**2))

def plot_solution(points, solution, title="Solution"):
    """Plot the solution using Plotly (colors indicate the solution value)."""
    x = points[:, 0]
    y = points[:, 1]
    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=5,
            color=solution,
            colorscale='Viridis',
            colorbar=dict(title="Value")
        )
    )])
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
    fig.show()

if __name__ == "__main__":
    pts = np.random.rand(100,2)
    sol = np.sin(np.pi * pts[:,0]) * np.sin(np.pi * pts[:,1])
    error = compute_l2_error(sol, sol + np.random.randn(100) * 0.1)
    print("L2 error:", error)
    plot_solution(pts, sol, title="Example Solution")
