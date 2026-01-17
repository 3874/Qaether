import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

def create_fcc_unit_cluster(sphere_radius=1.0):
    """
    중심 구체 1개와 그를 둘러싼 12개의 이웃 구체(FCC 배위수)만 생성합니다.
    """
    # FCC에서 인접한 두 점 사이의 거리 r_d = a / sqrt(2) = 2r
    # 따라서 half_a (a/2) = sqrt(2) * r
    half_a = np.sqrt(2) * sphere_radius
    
    # 1. 중심점
    points = [[0, 0, 0]]
    
    # 2. FCC의 12개 이웃 점 (퍼뮤테이션)
    # XY 평면 4개, YZ 평면 4개, XZ 평면 4개
    neighbors = [
        # XY Plane
        (half_a, half_a, 0), (half_a, -half_a, 0), (-half_a, half_a, 0), (-half_a, -half_a, 0),
        # YZ Plane
        (0, half_a, half_a), (0, half_a, -half_a), (0, -half_a, half_a), (0, -half_a, -half_a),
        # XZ Plane
        (half_a, 0, half_a), (half_a, 0, -half_a), (-half_a, 0, half_a), (-half_a, 0, -half_a)
    ]
    
    points.extend(neighbors)
    return np.array(points)

def verify_cluster_distances(points, target_dist=2.0):
    """
    중심점에서 12개 이웃까지의 거리가 정확히 target_dist인지 검증합니다.
    """
    center = points[0]
    neighbors = points[1:]
    
    print(f"\n--- FCC Unit Cluster Verification ---")
    distances = np.linalg.norm(neighbors - center, axis=1)
    
    for i, d in enumerate(distances):
        print(f"Neighbor {i+1:02d} distance: {d:.6f}")
    
    avg_dist = np.mean(distances)
    print(f"Average Distance: {avg_dist:.6f}")
    if np.allclose(distances, target_dist):
        print(f"Result: SUCCESS! Every neighbor touches the center at exactly one point (d={target_dist}).")
    print(f"--------------------------------------\n")

def get_sphere_mesh(center, radius, resolution=20):
    """
    구체의 3D 표면 데이터를 생성합니다.
    """
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]
    
    return x, y, z

def visualize_cosmic_universe(points, sphere_radius):
    """
    실제 3D 구체 메쉬를 렌더링하여 물리적인 접점(Contact Point)을 시각화합니다.
    """
    if len(points) == 0: return
    
    fig = go.Figure()
    
    # 정팔면체(Octahedron) 정의
    half_a = np.sqrt(2) * sphere_radius
    top_vertex = np.array([0, 2*half_a, 0])
    
    # 6개의 꼭짓점만 선택
    # 0: Bottom, 1,5,3,6: Middle Loop, 13: Top
    octa_indices = [0, 1, 5, 3, 6, 13]
    all_points = np.vstack([points, top_vertex])
    octahedron_points = all_points[octa_indices]
    
    # 6개의 서로 다른 색상
    colors = ['#FF4136', '#2ECC40', '#0074D9', '#FFDC00', '#B10DC9', '#FF851B']
    names = ['Bottom', 'Loop 1', 'Loop 2', 'Loop 3', 'Loop 4', 'Top']
    
    # 엣지 정의 (octahedron_points의 인덱스 기준: 0~5)
    # 0: Bottom, 1:L1, 2:L2, 3:L3, 4:L4, 5:Top
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),       # Bottom to Loop
        (1, 2), (2, 3), (3, 4), (4, 1),       # 中 Loop edges
        (5, 1), (5, 2), (5, 3), (5, 4)        # Top to Loop
    ]
    
    # 구체 시각화 (6개만)
    for i in range(6):
        p = octahedron_points[i]
        x, y, z = get_sphere_mesh(p, sphere_radius)
        color = colors[i]
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=0.3,
            name=names[i],
            hoverinfo='name'
        ))

    # 엣지 (라인) 그리기
    for start, end in edges:
        p1, p2 = octahedron_points[start], octahedron_points[end]
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color='black', width=6),
            showlegend=False
        ))

    fig.update_layout(
        title="<b>Isolated FCC Octahedron</b>",
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white'
        ),
        paper_bgcolor='white', font=dict(color='black'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    output_file = "cosmic_cluster.html"
    fig.write_html(output_file)
    print(f"Interactive 3D Transparent model saved to: {output_file}")

def visualize_matplotlib_universe(points, sphere_radius):
    """
    Matplotlib에서도 투명한 구체 표면을 렌더링합니다.
    """
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.set_box_aspect([1,1,1]) 

    half_a = np.sqrt(2) * sphere_radius
    top_vertex = np.array([0, 2*half_a, 0])
    octa_indices = [0, 1, 5, 3, 6, 13]
    all_points = np.vstack([points, top_vertex])
    octahedron_points = all_points[octa_indices]
    
    colors = ['#FF4136', '#2ECC40', '#0074D9', '#FFDC00', '#B10DC9', '#FF851B']

    for i in range(6):
        p = octahedron_points[i]
        x, y, z = get_sphere_mesh(p, sphere_radius)
        ax.plot_surface(x, y, z, color=colors[i], alpha=0.2, shade=True)

    # 옥타헤드론 엣지 연결
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
        (5, 1), (5, 2), (5, 3), (5, 4)
    ]
    
    for start, end in edges:
        p1, p2 = octahedron_points[start], octahedron_points[end]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=2)

    ax.set_axis_off()
    limit = 2.5
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    plt.title("Physical Contact Points (Zoom in to verify)", color='white')
    plt.show()

if __name__ == "__main__":
    R = 1.0     
    print(f"Generating Physical FCC Cluster (R={R})...")
    cluster_points = create_fcc_unit_cluster(R)
    verify_cluster_distances(cluster_points, target_dist=2.0*R)
    
    visualize_cosmic_universe(cluster_points, R)
    visualize_matplotlib_universe(cluster_points, R)
