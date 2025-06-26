import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


def draw_square_pyramid(scale=1.0, edge_values=None):
    """Draw a square pyramid with optional edge labels.

    Parameters
    ----------
    scale : float, optional
        Scaling factor for the pyramid size (default 1.0).
    edge_values : list of int, optional
        List of 8 integers between -3 and 3 representing labels for the
        edges in the following order:
        base edges (0-1, 1-2, 2-3, 3-0) followed by side edges
        (0-4, 1-4, 2-4, 3-4).
    """

    if edge_values is None:
        edge_values = [0] * 8

    if len(edge_values) != 8:
        raise ValueError("edge_values must contain exactly 8 integers")

    if not all(isinstance(v, int) and -3 <= v <= 3 for v in edge_values):
        raise ValueError("edge_values must be integers between -3 and 3")

    # Define vertices of a unit square pyramid centered at the origin
    base = np.array([
        [-0.5, -0.5, 0],
        [ 0.5, -0.5, 0],
        [ 0.5,  0.5, 0],
        [-0.5,  0.5, 0]
    ]) * scale
    top = np.array([[0, 0, 1]]) * scale
    vertices = np.vstack([base, top])

    # Edge definitions
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # base edges
        (0, 4), (1, 4), (2, 4), (3, 4)   # side edges
    ]

    # Prepare lines for plotting
    lines = []
    for start, end in edges:
        lines.append([vertices[start], vertices[end]])
    lc = Line3DCollection(lines, colors="b")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(lc)

    # Annotate edges
    for value, (start, end) in zip(edge_values, edges):
        midpoint = (vertices[start] + vertices[end]) / 2
        ax.text(midpoint[0], midpoint[1], midpoint[2], str(value), color="red")

    # Set limits to nicely fit the pyramid
    limit = scale / 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0, limit * 2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


if __name__ == "__main__":
    # Example usage
    example_values = [0, 1, -2, 3, -1, 2, -3, 1]
    draw_square_pyramid(scale=1.0, edge_values=example_values)
