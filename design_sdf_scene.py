import numpy as np
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *
import matplotlib.pyplot as plt

# Define orientation variables for all plots
elevation = 180
azimuth = 0


# Load data
r, R, gamma, N, P = get_data()

r0 = r[-1, :, :]
r3 = r[4, :, :]
r3inv = np.zeros_like(r3)
r3inv[:, 0] = -r3[:, 0]
r3inv[:, 1] = -r3[:, 1]
r3inv[:, 2] = r3[:, 2]
r4 = r[54576,:,:] # hook
r5 = r[1203,:,:]  # pretty straight! good




# === Scene 1 ===
shapes = [r0, r3, r4, r5, r0]
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
for r in shapes:
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color='C0', linewidth=2.0)

axis = np.array([0.0, 0.0, 1.0])
axis /= np.linalg.norm(axis)
cyl_center = np.array([-0.016, 0.016, 0.045])
cyl_radius = 0.002
cyl_height = 0.09
plot_obs = [
    {"type": "cylinder", "center": np.array([-0.016, 0.016, 0.045]), "radius": cyl_radius, "height": cyl_height, "color": "orange", "alpha": 0.35},
    {"type": "cylinder", "center": np.array([-0.002, -0.016, 0.045]), "radius": cyl_radius, "height": cyl_height, "color": "orange", "alpha": 0.35}
    ]
for obs in plot_obs:
    if obs["type"] == "cylinder":
        plot_capped_cylinder(ax, obs["center"], obs["radius"], obs["height"],
                             color=obs.get("color", "orange"), alpha=obs.get("alpha", 0.35))

equalize_axes(ax)
# ax1.view_init(elev=elevation, azim=azimuth)
plt.show()
input(" Press Enter to continue...")


# === Scene 2 ===
shapes = [r0, r4, r0]
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
for r in shapes:
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color='C0', linewidth=2.0)

cyl_center = np.array([0.02, 0.0, 0.045])
cyl_radius = 0.008
cyl_height = 0.040
box_center = np.array([-0.02, 0.0, 0.060])
box_half_sizes = np.array([0.015, 0.010, 0.015])
cyl = lambda X: sdf_capped_cylinder(X, center=cyl_center, axis=axis, radius=cyl_radius, height=cyl_height)
box = lambda X: sdf_box_aabb(X, center=box_center, half_sizes=box_half_sizes)
scene_sdf = lambda X: sdf_scene(X, [cyl, box], margin=0.0)

plot_obs = [
    # {"type": "cylinder", "center": cyl_center, "radius": cyl_radius, "height": cyl_height, "color": "orange", "alpha": 0.35},
    {"type": "box", "center": box_center, "half_sizes": box_half_sizes, "color": "red", "alpha": 0.25},
]
for obs in plot_obs:
    if obs["type"] == "cylinder":
        plot_capped_cylinder(ax, obs["center"], obs["radius"], obs["height"],
                             color=obs.get("color", "orange"), alpha=obs.get("alpha", 0.35))
    elif obs["type"] == "box":
        plot_box_aabb(ax, obs["center"], obs["half_sizes"],
                      color=obs.get("color", "red"), alpha=obs.get("alpha", 0.25))


equalize_axes(ax)
# ax.view_init(elev=elevation, azim=azimuth)
plt.show()
# input("Press Enter to continue...")

