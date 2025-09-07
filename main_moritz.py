import numpy as np
import pickle
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *

save = False
r, R, gamma, N, P = get_data()

# k-NN in feature space (use R just for speed, real edge weights from shape_dist)
nbrs, idxs = initialize_knn_graph(R, k=20)

plot_obs = None
scenario = 'basic' # 'basic', 'energy', 'sdf'

match scenario:
    case 'basic':
        w_basic = make_edge_weight_basic(r)
        adj = build_knn_graph(R, idxs, w_basic, valid_mask=None, tau=None, collision_ok=None)
        # custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
        # nbrs, idxs = pickle.load(open(custom_dir + '/knn.pickle', 'rb')), pickle.load(open(custom_dir + '/idxs.pickle', 'rb'))
        # adj = pickle.load(open(custom_dir + '/adj_basic.pickle', 'rb'))
    case 'energy':
        w_energy = make_edge_weight_energy(r, gamma, alpha=1.0, beta=1.0, lam=1.0)
        adj = build_knn_graph(R, idxs, w_energy, valid_mask=None, tau=None, collision_ok=None)
    case 'sdf':
        # one obstacle fitted to r3 with idx4
        axis = np.array([0.0, 0.0, 1.0]); axis /= np.linalg.norm(axis)
        cyl_center = np.array([-0.016, 0.016, 0.045]); cyl_radius = 0.002; cyl_height = 0.09
        plot_obs = [{"type":"cylinder","center":cyl_center,"radius":cyl_radius,"height":cyl_height,"color":"orange","alpha":0.35}]
        cyl = lambda X: sdf_capped_cylinder(X, center=cyl_center, axis=axis, radius=cyl_radius, height=cyl_height)
        scene_sdf = lambda X: sdf_scene(X, [cyl], margin=0.0)
        node_clearance, valid_mask = node_clearance_mask(r, scene_sdf)
        
        w_sdf   = make_edge_weight_sdf(r, node_clearance, alpha=1.0, mu=0.5, eps=0.01)
        sweep_ok = make_edge_sweep_checker(r, scene_sdf)
        adj   = build_knn_graph(R, idxs, w_sdf,   valid_mask=valid_mask, tau=None, collision_ok=sweep_ok)

        

# Define different waypoints
r0 = r[-1,:,:]  
r3 = r[4,:,:]
r4 = r[54576,:,:] # hook
r5 = r[1203,:,:]  # pretty straight! good
# waypoints = [r0, r3, r4, r5] # nice route
# waypoints = [r0, r3] # good for obstacle
waypoints = [r0, r3] # good for energy stuff because gamma has higher absolute magnitude than necessary

# Find the indices of our waypoints in the shape library
nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

# Actual graph search
full_path_indices = waypoint_planner(waypoint_indices, adj=adj)


print(f'Number of intermediate shapes: {len(full_path_indices)}')
plot_shape_sequence(all_rs=r, path_indices_list=[full_path_indices], waypoints_indices=waypoint_indices)
plot_gammas(all_gammas=gamma, path_indices_list=[full_path_indices], waypoints_indices=waypoint_indices)

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

input("Debug breakpoint. Press Enter to exit...")