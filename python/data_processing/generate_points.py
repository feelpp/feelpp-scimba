import numpy as np

def generate_interpolation_points(nx=50, ny=50):
    """Generate an nx x ny grid of points inside the unit square [0,1]^2."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    return points

if __name__ == "__main__":
    pts = generate_interpolation_points()
    print("Number of interpolation points:", pts.shape[0])
