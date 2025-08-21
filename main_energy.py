import numpy as np
import pickle
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *


r, R, gamma, N, P = get_data()

# load benchmark
custom_dir = '/Users/cveil/Desktop/sim/shape_graphs'
nbrs, idxs = pickle.load(open(custom_dir + '/knn.pickle', 'rb')), pickle.load(open(custom_dir + '/idxs.pickle', 'rb'))
adj_basic = pickle.load(open(custom_dir + '/adj_basic.pickle', 'rb'))

# initialize energy weight graph
w1 = make_edge_weight_tbd(r, gamma, alpha=0.0, beta=1.0, lam=0.0) # only low activation magnitude
adj1 = build_knn_graph(R, idxs, w1, valid_mask=None, tau=None, collision_ok=None)

w2 = make_edge_weight_tbd(r, gamma, alpha=0.0, beta=0.0, lam=1.0) # only smooth
adj2 = build_knn_graph(R, idxs, w2, valid_mask=None, tau=None, collision_ok=None)

# w3 = make_edge_weight_tbd(r, gamma, alpha=1.0, beta=1.0, lam=0.0) # distance + low activation magnitude
# adj3 = build_knn_graph(R, idxs, w3, valid_mask=None, tau=None, collision_ok=None)

# w4 = make_edge_weight_tbd(r, gamma, alpha=1.0, beta=0.0, lam=1.0) # distance + smooth
# adj4 = build_knn_graph(R, idxs, w4, valid_mask=None, tau=None, collision_ok=None)

# w5 = make_edge_weight_tbd(r, gamma, alpha=0.0, beta=1.0, lam=1.0) # distance + smooth
# adj5 = build_knn_graph(R, idxs, w5, valid_mask=None, tau=None, collision_ok=None)

w6 = make_edge_weight_tbd_updated(r, gamma, alpha=0.0, beta=1.0, lam=0.0) # new only low activation magnitude
adj6 = build_knn_graph(R, idxs, w6, valid_mask=None, tau=None, collision_ok=None)

w7 = make_edge_weight_tbd_updated(r, gamma, alpha=0.0, beta=0.0, lam=1.0) # new only smooth
adj7 = build_knn_graph(R, idxs, w7, valid_mask=None, tau=None, collision_ok=None)

# Define waypoints
r0 = r[-1,:,:]  
r3 = r[4,:,:]
waypoints = [r0, r3] # good for energy stuff because gamma has higher absolute magnitude than necessary
nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

# labels = ['basic', 'low act', 'smooth', 'dist + low act', 'dist + smooth']
# full_path_indices = []
# for temp_adj in [adj_basic, adj1, adj2, adj3, adj4]:
#     path_indices = waypoint_planner(waypoint_indices, adj=temp_adj)
#     full_path_indices.append(path_indices)
# plot_shape_sequence(all_rs=r, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices)
# plot_gammas(all_gammas=gamma, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices, labels=labels)

labels = ['basic', 'low act', 'smooth', 'new low act', 'new smooth']
full_path_indices = []
for temp_adj in [adj_basic, adj1, adj2, adj6, adj7]:
    path_indices = waypoint_planner(waypoint_indices, adj=temp_adj)
    full_path_indices.append(path_indices)

plot_shape_sequence(all_rs=r, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices)
plot_gammas(all_gammas=gamma, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices, labels=labels)

input("Debug breakpoint. Press Enter to exit...")