import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider

# FCC lattice points within a unit cube
def generate_fcc_lattice(n=1):
    # Basic FCC unit cell points
    corner = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])
    face_center = np.array([[0.5, 0.5, 0],
                            [0.5, 0.5, 1],
                            [0.5, 0, 0.5],
                            [0.5, 1, 0.5],
                            [0, 0.5, 0.5],
                            [1, 0.5, 0.5]])

    all_points = []

    for x in range(n):
        for y in range(n):
            for z in range(n):
                shift = np.array([x, y, z])
                all_points.extend(corner + shift)
                all_points.extend(face_center + shift)

    return np.array(all_points)

# Plotting function
def plot_fcc(ax, points):
    ax.clear()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, n_cells)
    ax.set_ylim(0, n_cells)
    ax.set_zlim(0, n_cells)
    
    # 눈금 제거
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Calculate number of points per unit cell
    points_per_cell = 8 + 6  # 8 corner points + 6 face center points
    total_cells = len(points) // points_per_cell

    # Separate corner points and face center points
    corner_points = []
    face_center_points = []
    
    for i in range(total_cells):
        start_idx = i * points_per_cell
        corner_points.extend(points[start_idx:start_idx+8])
        face_center_points.extend(points[start_idx+8:start_idx+points_per_cell])

    corner_points = np.array(corner_points)
    face_center_points = np.array(face_center_points)

    # Plot points with different colors
    ax.scatter(corner_points[:, 0], corner_points[:, 1], corner_points[:, 2], 
              c='blue', s=50, label='Corner points')
    ax.scatter(face_center_points[:, 0], face_center_points[:, 1], face_center_points[:, 2], 
              c='red', s=50, label='Face center points')

    # Connect corner points to nearest face center points
    for corner in corner_points:
        # Calculate distances to all face centers
        distances = np.sqrt(np.sum((face_center_points - corner)**2, axis=1))
        # Find the three nearest face centers (in FCC, each corner connects to three face centers)
        nearest_indices = np.argsort(distances)[:3]
        
        for idx in nearest_indices:
            face_center = face_center_points[idx]
            ax.plot([corner[0], face_center[0]], 
                   [corner[1], face_center[1]], 
                   [corner[2], face_center[2]], 
                   'k-', alpha=0.3)  # black lines with transparency

    # Connect face center points to their nearest neighbors
    for i, face_center in enumerate(face_center_points):
        distances = np.sqrt(np.sum((face_center_points - face_center)**2, axis=1))
        # Get 4 nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances)[1:5]  # Skip index 0 (self)
        
        for idx in nearest_indices:
            neighbor = face_center_points[idx]
            ax.plot([face_center[0], neighbor[0]], 
                   [face_center[1], neighbor[1]], 
                   [face_center[2], neighbor[2]], 
                   'r--', alpha=0.3)  # red dashed lines with transparency
    
    ax.legend()

# Initialize figure and sliders
n_cells = 2  # Default number of unit cells per axis
points = generate_fcc_lattice(n_cells)

fig = plt.figure(figsize=(10, 8))
# Create main axes for the plot
ax = fig.add_subplot(111, projection='3d')

# Add slider axes
slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
n_slider = Slider(
    ax=slider_ax,
    label='Number of Cells',
    valmin=1,
    valmax=5,
    valinit=n_cells,
    valstep=1
)

# Update function for the slider
def update(val):
    global points
    n_cells = int(val)
    points = generate_fcc_lattice(n_cells)
    plot_fcc(ax, points)
    fig.canvas.draw_idle()

n_slider.on_changed(update)
plot_fcc(ax, points)
plt.show()
