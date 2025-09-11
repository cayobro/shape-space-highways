import numpy as np
import pickle
import scipy
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *


custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
save = False
export_for_experiment = False
r, R, gamma, N, P = get_data()

# k-NN in feature space (use R just for speed, real edge weights from shape_dist)
# nbrs, idxs = initialize_knn_graph(R, k=20)
nbrs, idxs = pickle.load(open(custom_dir + '/knn.pickle', 'rb')), pickle.load(open(custom_dir + '/idxs.pickle', 'rb'))

scenario = 'sdf_cyl' # 'basic' or 'sdf'
match scenario:
    case 'basic':
        plot_obs = None
        node_clearance = None
        sweep_ok = None
        valid_mask = None
        scene_sdf = None
    case 'sdf_cyl':
        axis = np.array([0.0, 0.0, 1.0]); axis /= np.linalg.norm(axis)
        cyl_center_1 = np.array([-0.016, 0.016, 0.045]); cyl_center_2 = np.array([-0.002, -0.016, 0.045])
        cyl_radius = 0.002; cyl_height = 0.09
        plot_obs = [
            {"type": "cylinder", "center": cyl_center_1, "radius": cyl_radius, "height": cyl_height, "color": "orange", "alpha": 0.35},
            {"type": "cylinder", "center": cyl_center_2, "radius": cyl_radius, "height": cyl_height, "color": "orange", "alpha": 0.35}
            ]
        cyl_1= lambda X: sdf_capped_cylinder(X, center=cyl_center_1, axis=axis, radius=cyl_radius, height=cyl_height)
        cyl_2= lambda X: sdf_capped_cylinder(X, center=cyl_center_2, axis=axis, radius=cyl_radius, height=cyl_height)
        obstacles = [cyl_1, cyl_2]
        scene_sdf = lambda X: sdf_scene(X, obstacles, margin=0.0)   

        node_clearance, valid_mask = node_clearance_mask(r, scene_sdf)
        sweep_ok = make_edge_sweep_checker(r, scene_sdf)
    case 'sdf_box':
        box_center = np.array([-0.02, 0.0, 0.060])
        box_half_sizes = np.array([0.015, 0.010, 0.015])
        plot_obs = [{"type": "box", "center": box_center, "half_sizes": box_half_sizes, "color": "red", "alpha": 0.25}]
        box = lambda X: sdf_box_aabb(X, center=box_center, half_sizes=box_half_sizes)
        obstacles = [box]
        scene_sdf = lambda X: sdf_scene(X, obstacles, margin=0.0)   

        node_clearance, valid_mask = node_clearance_mask(r, scene_sdf)
        sweep_ok = make_edge_sweep_checker(r, scene_sdf)


params = {
    "r": r,
    "gamma": gamma,
    "alpha": 1.0,   # geometric
    "beta":  1.0,   # activation magnitude
    "delta": 1.0,   # activation smoothness
    "lamb":  0.5,   # SDF penalty
    "eps":   0.01,  # SDF soft threshold
    'sdf_fn': scene_sdf,  # SDF function (or None)
    "node_clearance": node_clearance  # SDF clearance (or None)
}
w = make_edge_weight(params)
# adj = build_knn_graph(R, idxs, w, valid_mask=valid_mask, tau=None, collision_ok=sweep_ok)
adj_basic = pickle.load(open(custom_dir + '/adj_basic.pickle', 'rb'))

# adj_box = pickle.load(open(custom_dir + '/adj_sdf_box_a1_b0_d0_l05.pickle', 'rb'))
# adj_box_energy = pickle.load(open(custom_dir + '/adj_sdf_box_a1_b1_d1_l05.pickle', 'rb'))
adj_cyl = pickle.load(open(custom_dir + '/adj_sdf_cyl_a1_b0_d0_l05.pickle', 'rb'))
adj_cyl_energy = pickle.load(open(custom_dir + '/adj_sdf_cyl_a1_b1_d1_l05.pickle', 'rb'))


# Define waypoints
r0 = r[-1,:,:] 
r3 = r[4,:,:]
r4 = r[54576,:,:] # hook
r5 = r[1203,:,:]  # pretty straight! good
waypoints = [r0, r3, r4, r5, r0] # with sdf_cyl
# waypoints = [r0, r4, r0] # with sdf_box
nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

all_indices = []
gamma_seq = []
shape_seq = []
for adj in [adj_basic, adj_cyl, adj_cyl_energy]:
# for adj in [adj_basic, adj_box, adj_box_energy]:
# if True:
    wp_indices = waypoint_planner(waypoint_indices, adj, params)
    all_indices.append(wp_indices)
    gamma_seq.append(gamma[wp_indices,:])
    shape_seq.append(r[wp_indices,:,:])

for path_indices in all_indices:
    ax = plot_shape_sequence(all_rs=r, path_indices_list=[path_indices], waypoints_indices=waypoint_indices)
    for obs in plot_obs:
        if obs["type"] == "cylinder":
            plot_capped_cylinder(ax, obs["center"], obs["radius"], obs["height"],
                                color=obs.get("color", "orange"), alpha=obs.get("alpha", 0.35))
        elif obs["type"] == "box":
            plot_box_aabb(ax, obs["center"], obs["half_sizes"],
                      color=obs.get("color", "red"), alpha=obs.get("alpha", 0.25))
    # Try 2D plot
    fig, ax = plt.subplots(1,1)
    ax.plot(r[:,0,0], r[:,0,1], color='lightgray', alpha=0.5)
    ax.plot(r[:,1,0], r[:,1,1], color='lightgray', alpha=0.5)
    ax.plot(r[:,2,0], r[:,2,1], color='lightgray', alpha=0.5)

plot_gammas(all_gammas=gamma, path_indices_list=all_indices, waypoints_indices=waypoint_indices)


if save:
    custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
    file_knn = custom_dir + '/knn.pickle'
    file_adj = custom_dir + '/adj_sdf_box_a1_b1_d1_l05.pickle'
    file_idx = custom_dir + '/idxs.pickle'
    knnPickle = open(file_knn, 'wb') 
    pickle.dump(nbrs, knnPickle)  
    knnPickle.close()
    adjPickle = open(file_adj, 'wb') 
    pickle.dump(adj, adjPickle)  
    adjPickle.close()
    idxsPickle = open(file_idx, 'wb')
    pickle.dump(idxs, idxsPickle)
    idxsPickle.close()

export_data = False # full sequence
if export_data:
    save_dir = '/Users/cveil/Desktop/'
    idx = 2 # specify with adj

    wps = []
    for wp in waypoint_indices:
        wps.append(all_indices[0].index(wp))
    gamma_seq = np.array(gamma[all_indices[idx]])
    # Interpolate gamma to smooth a bit
    x_idx = np.linspace(0, gamma_seq.shape[0] - 1, num=gamma_seq.shape[0])
    gamma_seq_interp = np.zeros_like(gamma_seq)
    for dim in range(gamma.shape[1]):
        spline = scipy.interpolate.UnivariateSpline(x_idx, gamma_seq[:, dim], s=0.005)
        gamma_seq_interp[:, dim] = spline(x_idx)
    fig, ax = plt.subplots(3,1, sharex=True)
    for i in range(3):
        ax[i].scatter(x_idx, gamma_seq[:,i], marker='.', alpha=0.5)
        ax[i].plot(x_idx, gamma_seq_interp[:,i], color=get_color('cardinal'), linewidth=2.0)
    
    # Extract corresponding shapes
    r_seq = np.array(r[all_indices[idx]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(r_seq.shape[0]):
        color = get_color('cardinal')
        ax.plot(r_seq[i,:,0], r_seq[i,:,1], r_seq[i,:,2], color=color)

    ## save as pickle
    rPickle = open(save_dir + 'r_seq_sdf_energy.pickle', 'wb')
    pickle.dump(r_seq, rPickle)
    rPickle.close() 
    gammaPickle = open(save_dir + 'gamma_seq_sdf_energy.pickle', 'wb')
    pickle.dump(gamma_seq, gammaPickle)
    gammaPickle.close()


export_for_experiment = False
if export_for_experiment:
    save_dir = '/Users/cveil/Desktop/'
    idx = 1 # specify with adj

    wps = []
    for wp in waypoint_indices:
        wps.append(all_indices[0].index(wp))
    gamma_seq = np.array(gamma[all_indices[idx]])
    # Interpolate gamma to smooth a bit
    x_idx = np.linspace(0, gamma_seq.shape[0] - 1, num=gamma_seq.shape[0])
    x_exp = np.unique(np.concatenate((x_idx[::5], [12,74,139]))).astype(int)
    # x_exp = np.unique(np.concatenate((x_idx[::5], wps))).astype(int)
    gamma_seq_interp = np.zeros_like(gamma_seq); gamma_exp = np.zeros((x_exp.shape[0], gamma_seq.shape[1]))
    for dim in range(gamma.shape[1]):
        spline = scipy.interpolate.UnivariateSpline(x_idx, gamma_seq[:, dim], s=0.03)
        gamma_seq_interp[:, dim] = spline(x_idx)
        gamma_exp[:, dim] = spline(x_exp)
    fig, ax = plt.subplots(3,1, sharex=True)
    for i in range(3):
        ax[i].scatter(x_idx, gamma_seq[:,i], marker='.', alpha=0.5)
        ax[i].scatter(x_exp, gamma_exp[:,i], color=get_color('cardinal'), alpha=0.5)
        ax[i].plot(x_idx, gamma_seq_interp[:,i], color=get_color('cardinal'), linewidth=2.0)
    
    # Extract corresponding shapes
    r_seq = np.array(r[all_indices[idx]])
    r_exp = r_seq[x_exp,:,:] #np.zeros((x_exp.shape[0], r.shape[1], r.shape[2]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(r_seq.shape[0]):
        color = get_color('cardinal') if i in x_exp else 'gray'
        ax.plot(r_seq[i,:,0], r_seq[i,:,1], r_seq[i,:,2], color=color)
        # ax.plot(r_exp[i,:,0], r_exp[i,:,1], r_exp[i,:,2], color=get_color('cardinal'), linewidth=2.0)

    ## save as pickle
    rPickle = open(save_dir + 'r_exp.pickle', 'wb')
    pickle.dump(r_exp, rPickle)
    rPickle.close() 
    gammaPickle = open(save_dir + 'gamma_exp.pickle', 'wb')
    pickle.dump(gamma_exp, gammaPickle)
    gammaPickle.close()


input("Debug breakpoint. Press Enter to exit...")