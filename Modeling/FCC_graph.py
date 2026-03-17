import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Slider, Button

def generate_fcc_lattice(nx=24, ny=24, nz=24, a=1.0):
    """
    Generates FCC lattice points using vectorized numpy operations.
    """
    # 1. Corners (0,0,0) type
    i, j, k = np.mgrid[0:nx+1, 0:ny+1, 0:nz+1]
    corners = np.vstack([i.ravel(), j.ravel(), k.ravel()]).T.astype(float)
    
    # 2. Face centers (0.5, 0.5, 0) type
    # XY face
    i, j, k = np.mgrid[0:nx, 0:ny, 0:nz+1]
    xy = np.vstack([i.ravel()+0.5, j.ravel()+0.5, k.ravel()]).T
    
    # YZ face
    i, j, k = np.mgrid[0:nx+1, 0:ny, 0:nz]
    yz = np.vstack([i.ravel(), j.ravel()+0.5, k.ravel()+0.5]).T
    
    # XZ face
    i, j, k = np.mgrid[0:nx, 0:ny+1, 0:nz]
    xz = np.vstack([i.ravel()+0.5, j.ravel(), k.ravel()+0.5]).T
    
    points = np.concatenate([corners, xy, yz, xz]) * a
    return np.unique(points, axis=0)

def find_neighbors_spatial_hash(points, a=1.0, tol=1e-5):
    """
    Efficiently find nearest neighbors using spatial hashing (O(N)).
    Distance is a/sqrt(2).
    """
    nn_dist = a / np.sqrt(2)
    bucket_size = nn_dist + tol
    
    # Map points to buckets
    buckets = {}
    for idx, p in enumerate(points):
        key = tuple((p / bucket_size).astype(int))
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(idx)
        
    neighbors = []
    # Only iterate through non-empty buckets
    for key, indices in buckets.items():
        # Check current and 26 adjacent buckets (3x3x3 cube)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if neighbor_key in buckets:
                        for i in indices:
                            for j in buckets[neighbor_key]:
                                if i < j: # Avoid double counting and self-pairing
                                    d = np.linalg.norm(points[i] - points[j])
                                    if abs(d - nn_dist) < tol:
                                        neighbors.append((i, j))
    return neighbors

class FCCVisualizer:
    def __init__(self, nx=12, ny=12, nz=12, a=1.0):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.a = a
        self.max_edges = 15000
        
        self.fig = plt.figure(figsize=(12, 9))
        # Leave space on the left for sliders
        self.ax = self.fig.add_axes([0.2, 0.1, 0.75, 0.85], projection='3d')
        
        # --- UI Element Setup ---
        ax_color = 'lightgoldenrodyellow'
        self.ax_nx = self.fig.add_axes([0.05, 0.7, 0.1, 0.03], facecolor=ax_color)
        self.ax_ny = self.fig.add_axes([0.05, 0.6, 0.1, 0.03], facecolor=ax_color)
        self.ax_nz = self.fig.add_axes([0.05, 0.5, 0.1, 0.03], facecolor=ax_color)
        self.ax_update = self.fig.add_axes([0.05, 0.4, 0.1, 0.05])
        
        self.s_nx = Slider(self.ax_nx, 'nx', 1, 32, valinit=nx, valstep=1)
        self.s_ny = Slider(self.ax_ny, 'ny', 1, 32, valinit=ny, valstep=1)
        self.s_nz = Slider(self.ax_nz, 'nz', 1, 32, valinit=nz, valstep=1)
        self.btn_update = Button(self.ax_update, 'Update', color=ax_color, hovercolor='0.975')
        
        self.btn_update.on_clicked(self.update_plot)
        self.update_plot(None)

    def update_plot(self, event):
        self.nx = int(self.s_nx.val)
        self.ny = int(self.s_ny.val)
        self.nz = int(self.s_nz.val)
        
        print(f"Update: {self.nx}x{self.ny}x{self.nz}")
        
        self.ax.cla()
        points = generate_fcc_lattice(self.nx, self.ny, self.nz, self.a)
        neighbors = find_neighbors_spatial_hash(points, self.a)
        
        # 1. Plot Atoms
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c='royalblue', s=1, alpha=0.3)
        
        # 2. Plot Edges
        draw_neighbors = neighbors
        if len(neighbors) > self.max_edges:
            indices = np.random.choice(len(neighbors), self.max_edges, replace=False)
            draw_neighbors = [neighbors[i] for i in indices]
            
        segments = [np.array([points[i], points[j]]) for i, j in draw_neighbors]
        lc = Line3DCollection(segments, colors='gray', linewidths=0.2, alpha=0.1)
        self.ax.add_collection(lc)
        
        # 3. Aesthetics
        self.ax.set_title(f"FCC 1-Skeleton ({self.nx}x{self.ny}x{self.nz} cells)")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        max_range = np.max(points.max(axis=0) - points.min(axis=0)) / 2.0
        mid = (points.max(axis=0) + points.min(axis=0)) * 0.5
        self.ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        self.ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        self.ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize FCC Nearest-Neighbor Graph with UI.")
    parser.add_argument("-nx", type=int, default=12, help="Initial nx.")
    parser.add_argument("-ny", type=int, default=12, help="Initial ny.")
    parser.add_argument("-nz", type=int, default=12, help="Initial nz.")
    parser.add_argument("-a", type=float, default=1.0, help="Lattice constant.")
    args = parser.parse_args()
    
    visualizer = FCCVisualizer(nx=args.nx, ny=args.ny, nz=args.nz, a=args.a)
    plt.show()

