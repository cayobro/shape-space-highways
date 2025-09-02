import numpy as np
import pickle
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *

save = True
plot_obs = None
r, R, gamma, N, P = get_data()

ks = [5, 10, 20, 30, 40, 50]

custom_dir = '/Users/cveil/Desktop/sim/shape_graphs/library_variations'


### Analyze different library sizes: Vary N
nhs = [100, 10, 1]
for nh in nhs:
    r_red, gamma_red = r[::nh,:,:], gamma[::nh,:]
    N_red = r_red.shape[0]
    print(f"Number of shapes: {N_red}")
    R_red = r_red.reshape(N_red, -1)
    nbrs, idxs = initialize_knn_graph(R, k=20)
    w_basic = make_edge_weight_basic(r)
    adj = build_knn_graph(R, idxs, w_basic, valid_mask=None, tau=None, collision_ok=None)
    if save:
        file_knn = custom_dir + f'/knn_N_{N_red}.pickle'
        file_adj = custom_dir + f'/adj_basic_N_{N_red}.pickle'
        file_idx = custom_dir + f'/idxs_N_{N_red}.pickle'
        knnPickle = open(file_knn, 'wb') 
        pickle.dump(nbrs, knnPickle)  
        knnPickle.close()
        adjPickle = open(file_adj, 'wb') 
        pickle.dump(adj, adjPickle)  
        adjPickle.close()
        idxsPickle = open(file_idx, 'wb')
        pickle.dump(idxs, idxsPickle)
        idxsPickle.close()
input("Debug breakpoint. Press Enter to continue...")
# custom_dir = '/Users/cveil/Desktop/sim/shape_graphs/library_variations'
# nbrs, idxs = pickle.load(open(custom_dir + '/knn.pickle', 'rb')), pickle.load(open(custom_dir + '/idxs.pickle', 'rb'))
# adj = pickle.load(open(custom_dir + '/adj_basic.pickle', 'rb'))



### Analyze different discretizations: Vary n_z







# Define waypoints
r0 = r[-1,:,:]  
r3 = r[4,:,:]
r4 = r[54576,:,:] # hook
r5 = r[1203,:,:]  # pretty straight! good
# waypoints = [r0, r3, r4, r5] # nice route
# waypoints = [r0, r3] # good for obstacle
# TODO DUENNES OBSTACLE von r0 zu r0

waypoints = [r0, r3] # good for energy stuff because gamma has higher absolute magnitude than necessary

nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

i = 0
full_path_indices = []
for temp_adj in [adj, adj]:
    if i == 0:
        path_indices = waypoint_planner(waypoint_indices, adj=temp_adj)
    else:
        path_indices = waypoint_planner_tbd(waypoint_indices, adj=temp_adj, gamma=gamma)
    full_path_indices.append(path_indices)
    i = i + 1

print(len(full_path_indices[0]))
print(len(full_path_indices[1]))
plot_shape_sequence(all_rs=r, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices)
plot_gammas(all_gammas=gamma, path_indices_list=full_path_indices, waypoints_indices=waypoint_indices)

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