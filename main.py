import numpy as np
import pickle
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

scenario = 'basic' # 'basic' or 'sdf'
match scenario:
    case 'basic':
        plot_obs = None
        node_clearance = None
        sweep_ok = None
        valid_mask = None
        scene_sdf = None
    case 'sdf':
        # # Scene SDF (near the centerline)
        # axis = np.array([0.0, 0.0, 1.0]); axis /= np.linalg.norm(axis)
        # cyl_center = np.array([0.02, 0.0, 0.045]); cyl_radius = 0.008; cyl_height = 0.040
        # box_center = np.array([-0.02, 0.0, 0.060]); box_half_sizes = np.array([0.015, 0.010, 0.015])
        # cyl = lambda X: sdf_capped_cylinder(X, center=cyl_center, axis=axis, radius=cyl_radius, height=cyl_height)
        # box = lambda X: sdf_box_aabb(X, center=box_center, half_sizes=box_half_sizes)
        # scene_sdf = lambda X: sdf_scene(X, [cyl, box], margin=0.0)
        # plot_obs = [{"type":"cylinder","center":cyl_center,"radius":cyl_radius,"height":cyl_height,"color":"orange","alpha":0.35},
        #     {"type":"box","center":box_center,"half_sizes":box_half_sizes,"color":"red","alpha":0.25},]
        # # two posts left/right of centerline
        # gap = 0.010  # 10 mm gap around x=0
        # post_off = 0.012  # center of posts
        # rad = 0.008        # post radius 8 mm
        # z_mid = 0.045
        # H = 0.050
        # axis = np.array([0,0,1.0]); axis /= np.linalg.norm(axis)
        # cyl_L = lambda X: sdf_capped_cylinder(X, center=np.array([-post_off, 0.0, z_mid]), axis=axis, radius=rad, height=H)
        # cyl_R = lambda X: sdf_capped_cylinder(X, center=np.array([+post_off, 0.0, z_mid]), axis=axis, radius=rad, height=H)
        # obstacles = [cyl_L, cyl_R]
        # scene_sdf = lambda X: sdf_scene(X, obstacles, margin=0.0)   
        # plot_obs = [
        #     {"type":"cylinder","center":np.array([-post_off,0,z_mid]),"radius":rad,"height":H,"color":"orange","alpha":0.35},
        #     {"type":"cylinder","center":np.array([+post_off,0,z_mid]),"radius":rad,"height":H,"color":"orange","alpha":0.35},
        #     ]
        # one obstacle fitted to r3 with idx4
        axis = np.array([0.0, 0.0, 1.0]); axis /= np.linalg.norm(axis)
        cyl_center = np.array([-0.016, 0.016, 0.045]); cyl_radius = 0.002; cyl_height = 0.09
        plot_obs = [{"type":"cylinder","center":cyl_center,"radius":cyl_radius,"height":cyl_height,"color":"orange","alpha":0.35}]


        cyl = lambda X: sdf_capped_cylinder(X, center=cyl_center, axis=axis, radius=cyl_radius, height=cyl_height)
        scene_sdf = lambda X: sdf_scene(X, [cyl], margin=0.0)

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
adj = build_knn_graph(R, idxs, w, valid_mask=valid_mask, tau=None, collision_ok=sweep_ok)


# Define waypoints
r0 = r[-1,:,:]  
r3 = r[4,:,:]
r4 = r[54576,:,:] # hook
r5 = r[1203,:,:]  # pretty straight! good
# waypoints = [r0, r3, r4, r5] # nice route
# waypoints = [r0, r3] # good for obstacle

waypoints = [r0, r3] # good for energy stuff because gamma has higher absolute magnitude than necessary

nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]
path_indices = waypoint_planner(waypoint_indices, adj, params)


plot_shape_sequence(all_rs=r, path_indices_list=[path_indices], waypoints_indices=waypoint_indices)
plot_gammas(all_gammas=gamma, path_indices_list=[path_indices], waypoints_indices=waypoint_indices)

if save:
    custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
    file_knn = custom_dir + '/knn.pickle'
    file_adj = custom_dir + '/adj_basic.pickle'
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

if export_for_experiment:
    save_dir = '/Users/cveil/Desktop/'
    # r1 = r[idx,:,:]
    # gamma1 = gamma[idx,:].reshape(1,3)
    # df = pd.DataFrame (r1, columns = ['x', 'y', 'z']); df.to_csv(save_dir + 'r_1.csv')
    # df = pd.DataFrame (gamma1, columns = ['act 1', 'act 2', 'act 3']); df.to_csv(save_dir + 'gamma_1.csv')


input("Debug breakpoint. Press Enter to exit...")