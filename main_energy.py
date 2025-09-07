import numpy as np
import pickle
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *


r, R, gamma, N, P = get_data()
params = {
    "r": r,
    "gamma": gamma,
    "alpha": 1.0,   # geometric
    "beta":  1.0,   # activation magnitude
    "delta": 1.0,   # activation smoothness
    "lamb":  0.5,   # SDF penalty
    "eps":   0.01,  # SDF soft threshold
    'sdf_fn': None,  # SDF function (or None)
    "node_clearance": None  # SDF clearance (or None)
}
# load benchmark
custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
nbrs, idxs = pickle.load(open(custom_dir + '/knn.pickle', 'rb')), pickle.load(open(custom_dir + '/idxs.pickle', 'rb'))
adj_basic = pickle.load(open(custom_dir + '/adj_basic.pickle', 'rb'))

# initialize energy weight graph
params['alpha'] = 1.0; params['beta'] = 1.0; params['delta'] = 1.0
w1 = make_edge_weight(params) 
adj1 = build_knn_graph(R, idxs, w1, valid_mask=None, tau=None, collision_ok=None)

params['alpha'] = 5000.0; params['beta'] = 1.0; params['delta'] = 1.0
w2 = make_edge_weight(params) 
adj2 = build_knn_graph(R, idxs, w2, valid_mask=None, tau=None, collision_ok=None)



# Define waypoints
r0 = r[-1,:,:]  
r3 = r[4,:,:]
waypoints = [r0, r3] # good for energy stuff because gamma has higher absolute magnitude than necessary
nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

labels = ['dist', 'high alpha', 'low alpha']
full_path_indices = []
for temp_adj in [adj_basic, adj2, adj1]:
    path_indices = waypoint_planner(waypoint_indices, adj=temp_adj, params=params)
    full_path_indices.append(path_indices)

plot_shape_sequence(all_rs=r, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices)
plot_gammas(all_gammas=gamma, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices, labels=labels)

if False:
    custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
    file_adj = custom_dir + '/adj_energy_a1_b1_d1.pickle'
    adjPickle = open(file_adj, 'wb') 
    pickle.dump(adj6, adjPickle)  
    adjPickle.close()



input("Debug breakpoint. Press Enter to exit...")