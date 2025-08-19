import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

torch.manual_seed(0)

def plot_capped_cylinder(ax, center, radius, height, color='orange', alpha=0.25, n_theta=60, n_h=20):
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(-height/2.0, height/2.0, n_h)
    Theta, Z = np.meshgrid(theta, z)
    X = center[0] + radius*np.cos(Theta)
    Y = center[1] + radius*np.sin(Theta)
    Z = center[2] + Z
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=alpha, color=color)
    ax.plot(center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta),
            center[2] + height/2.0*np.ones_like(theta), color=color, alpha=0.8)
    ax.plot(center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta),
            center[2] - height/2.0*np.ones_like(theta), color=color, alpha=0.8)


def plot_box_aabb(ax, center, half_sizes, color='red', alpha=0.20):
    cx, cy, cz = center
    hx, hy, hz = half_sizes
    corners = np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz]
    ])
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    polys = [corners[idx] for idx in faces]
    pc = Poly3DCollection(polys, facecolors=color, edgecolors='k', linewidths=0.3, alpha=alpha)
    ax.add_collection3d(pc)


def plot_tube(ax, curve, radius, n_theta=18, color='C0', alpha=0.25):
    P = curve.shape[0]
    if P < 2: return
    T = np.diff(curve, axis=0)
    T = np.vstack([T[0], T])
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    N = T - (T @ ref)[:, None] * ref
    bad = np.linalg.norm(N, axis=1) < 1e-8
    N[bad] = T[bad] - (T[bad] @ np.array([1.0,0.0,0.0]))[:,None] * np.array([1.0,0.0,0.0])
    N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    B = np.cross(T, N)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=True)
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    quads = []
    for i in range(P):
        ring = curve[i] + radius * (np.outer(cos_t, N[i]) + np.outer(sin_t, B[i]))
        if i > 0:
            prev_ring = prev
            for k in range(len(theta)-1):
                quads.append([prev_ring[k], prev_ring[k+1], ring[k+1], ring[k]])
        prev = ring
    pc = Poly3DCollection(quads, facecolors=color, edgecolors='none', alpha=alpha)
    ax.add_collection3d(pc)


def plot_shape_sequence(
    all_rs,
    path_indices,
    obstacles=None,
    tube_radius=None,
    waypoints_indices=None,                    # list of (P,3) curves to highlight
    ax=None,
    path_color='C0',
    endpoint_color='black',
    waypoint_color='C3',
    path_linewidth=1.0,
    waypoint_linewidth=2.5,
    waypoint_marker=False,
    waypoint_marker_size=20,
):
    """
    shapes: list/array of (P,3) centerlines (e.g., a planned path)
    waypoints: optional list of (P,3) curves you explicitly provided to the planner
    """

    created_ax = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        created_ax = True

    # --- draw path ---
    shapes = all_rs[path_indices]
    for t, curve in enumerate(shapes):
        color = endpoint_color if (t == 0 or t == len(shapes) - 1) else path_color
        ax.plot(curve[:,0], curve[:,1], curve[:,2], color=color, linewidth=path_linewidth)
        if tube_radius is not None:
            plot_tube(ax, curve, tube_radius, n_theta=18, color=path_color, alpha=0.20)

    # --- draw obstacles (same as before) ---
    if obstacles:
        for obs in obstacles:
            if obs["type"] == "cylinder":
                plot_capped_cylinder(ax, obs["center"], obs["radius"], obs["height"],
                                     color=obs.get("color","orange"), alpha=obs.get("alpha",0.35))
            elif obs["type"] == "box":
                plot_box_aabb(ax, obs["center"], obs["half_sizes"],
                              color=obs.get("color","red"), alpha=obs.get("alpha",0.25))

    # --- draw explicit waypoints in different color ---
    if waypoints_indices:
        waypoints = all_rs[waypoints_indices]
        for wp in waypoints:
            ax.plot(wp[:,0], wp[:,1], wp[:,2], color=waypoint_color, linewidth=waypoint_linewidth)
            if tube_radius is not None:
                plot_tube(ax, wp, tube_radius, n_theta=18, color=waypoint_color, alpha=0.35)
            if waypoint_marker:
                ax.scatter(wp[0,0], wp[0,1], wp[0,2], s=waypoint_marker_size, c=waypoint_color, depthshade=False)
                ax.scatter(wp[-1,0], wp[-1,1], wp[-1,2], s=waypoint_marker_size, c=waypoint_color, depthshade=False)

    # --- equalize axes ---
    x_limits = ax.get_xlim(); y_limits = ax.get_ylim(); z_limits = ax.get_zlim()
    x_range = x_limits[1]-x_limits[0]; y_range = y_limits[1]-y_limits[0]; z_range = z_limits[1]-z_limits[0]
    max_range = max(x_range, y_range, z_range)
    x_mid = sum(x_limits)/2; y_mid = sum(y_limits)/2; z_mid = sum(z_limits)/2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

    if created_ax:
        plt.show(block=False)
    return ax


def plot_gammas(all_gammas, path_indices, waypoints_indices=None, waypoint_color='red', gamma_color='blue', waypoint_marker_size=50):
    """
    Plots the gamma values and optionally highlights waypoints.

    Parameters:
        gammas (array-like): Array of gamma values (N x 3).
        waypoints (array-like, optional): Array of waypoint indices to highlight.
        waypoint_color (str, optional): Color for the waypoints. Default is 'red'.
        gamma_color (str, optional): Color for the gamma points. Default is 'blue'.
        waypoint_marker_size (int, optional): Size of the waypoint markers. Default is 50.
    """
    gammas = all_gammas[path_indices]
    xvec = np.arange(1, gammas.shape[0] + 1)  # X-axis values for gamma indices
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))  # Create subplots for gamma1, gamma2, gamma3

    # Plot gamma values for each component
    ax[0].scatter(xvec, gammas[:, 0], label='gamma1', color=gamma_color)
    ax[1].scatter(xvec, gammas[:, 1], label='gamma2', color=gamma_color)
    ax[2].scatter(xvec, gammas[:, 2], label='gamma3', color=gamma_color)

    # Highlight waypoints if provided
    if waypoints_indices is not None:
        for i, a in enumerate(ax):
            for j in range(len(waypoints_indices)):
                wp = path_indices.index(waypoints_indices[j])
                a.scatter(wp+1, all_gammas[waypoints_indices[j], i],
                        color=waypoint_color, s=waypoint_marker_size)

    # Add legends and labels
    for i, a in enumerate(ax):
        # a.legend()
        a.set_ylabel(f'gamma{i+1}')
        a.grid(True)

    ax[2].set_xlabel('Index')  # Label for the x-axis
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot