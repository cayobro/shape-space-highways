import numpy as np
import torch
import time
import heapq
from sklearn.neighbors import NearestNeighbors
from src.edge_utils import *

torch.manual_seed(0)

def get_data(base_path='/Users/cveil/Desktop/stanford/data/gravitation/interpolated_dense_'):
    gamma_data = np.load(base_path + "gamma.npz")
    r_data = np.load(base_path + "r.npz", allow_pickle=True)
    z_data = np.load(base_path + "z.npz", allow_pickle=True)
    gamma = gamma_data["gamma"]
    r = r_data["r"]
    z_raw = z_data["z"]
    z = z_raw[0,:]
    r0 = np.zeros((r.shape[1], 3))
    r0[:,2] = z
    r = np.vstack([r, r0.reshape(1, *r0.shape)])        # add zero config
    gamma = np.vstack([gamma, np.zeros((1, gamma.shape[1]))])
    N, P, _ = r.shape
    R = r.reshape(N, -1)
    return r, R, gamma, N, P


def initialize_knn_graph(centerlines, k=20):
    """
    Initializes a k-NN graph for a set of points.

    Parameters:
        centerlines (array-like): The input points for which the k-NN graph is built.
        k (int): The number of nearest neighbors to consider.

    Returns:
        nbrs: The NearestNeighbors model fitted to the input points.
        idxs: The indices of the k+1 nearest neighbors for each point.
    """
    tic = time.time()  #
    # Fit a k-NN model to the input points
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(centerlines)
    # Get the indices of the k+1 nearest neighbors (including the point itself)
    _, idxs = nbrs.kneighbors(centerlines, return_distance=True)
    toc = time.time()  
    print(f"k-NN graph initialized in {toc - tic:.3f} seconds.")
    return nbrs, idxs


def build_knn_graph(R, idxs, weight_fn, valid_mask=None, tau=None, collision_ok=None):
    """
    Builds an adjacency list for a k-NN graph.

    Parameters:
        R (array-like): The input points (centerlines)
        idxs (array-like): Indices of the k+1 nearest neighbors for each point.
        weight_fn (function): A function to compute the weight of an edge between two points.
        valid_mask (array-like, optional): A boolean mask indicating valid points.
        tau (float, optional): A threshold for edge weights; edges with weights > tau are excluded.
        collision_ok (function, optional): A function to check if an edge is collision-free (needed for sdf)

    Returns:
        adj (list of lists): The adjacency list representation of the graph.
    """
    tic = time.time()  # Start timing
    N = R.shape[0]  # Number of points
    adj = [[] for _ in range(N)]  # Initialize an empty adjacency list

    # Iterate over all points
    for i in range(N):
        # Skip invalid points if a valid_mask is provided
        if valid_mask is not None and not valid_mask[i]:
            continue
        # Iterate over the neighbors of point i (excluding itself)
        for j in idxs[i][1:]:
            # Skip invalid neighbors if a valid_mask is provided
            if valid_mask is not None and not valid_mask[j]:
                continue
            # Skip edges that fail the collision check
            if collision_ok is not None and not collision_ok(i, j):
                continue
            # Compute the weight of the edge
            w = weight_fn(i, j)
            # Add the edge to the adjacency list if it satisfies the weight threshold
            if tau is None or w <= tau:
                adj[i].append((j, w))
    toc = time.time()  # End timing
    print(f"Adjacency matrix with {weight_fn} built in {toc - tic:.3f} seconds.")
    return adj


def make_nearest_index_fn(r):
    """
    Creates a function to find the nearest index to a given sample.

    Parameters:
        r (array-like): A list or array of points.

    Returns:
        nearest_index_to (function): A function that finds the nearest index to a given sample.
    """
    def nearest_index_to(sample):
        # Compute the difference between the sample and all points in r
        diffs = r - sample
        # Compute the mean squared distance for each point
        d = np.sqrt(np.mean(np.sum(diffs**2, axis=2), axis=1))
        # Return the index of the point with the smallest distance
        return int(np.argmin(d))
    return nearest_index_to

def dijkstra_shortest_path(start, goal, adj):
    """
    Finds the shortest path between two nodes in a graph using Dijkstra's algorithm.

    Parameters:
        start (int): The starting node.
        goal (int): The goal node.
        adj (list of lists): The adjacency list representation of the graph.

    Returns:
        path (list): The shortest path from start to goal as a list of node indices.
    """
    tic = time.time()  # Start timing
    N = len(adj)  # Number of nodes in the graph
    dist = [np.inf]*N  # Initialize distances to infinity
    prev = [-1]*N  # Initialize previous nodes to -1 (undefined)
    dist[start] = 0.0  # Distance to the start node is 0
    pq = [(0.0, start)]  # Priority queue for Dijkstra's algorithm (min-heap)

    # Main loop of Dijkstra's algorithm
    while pq:
        d, i = heapq.heappop(pq)  # Get the node with the smallest distance
        if i == goal:  # Stop if the goal node is reached
            break
        if d > dist[i]:  # Skip if the current distance is not optimal
            continue
        # Iterate over neighbors of the current node
        for j, w in adj[i]:
            nd = d + w  # Compute the new distance to neighbor j
            if nd < dist[j]:  # Update if the new distance is smaller
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd, j))  # Push the neighbor into the priority queue

    # Reconstruct the shortest path from start to goal
    path = []
    i = goal
    while i != -1:  # Follow the previous nodes until the start node is reached
        path.append(int(i))
        i = prev[i]
    toc = time.time()  # End timing
    print(f"Shortest path from {start} to {goal} found in {toc - tic:.3f} seconds.")

    # Check if a valid path was found
    if not path:
        print(f"No path found from {start} to {goal}.")
        return []
    return path[::-1]  # Return the path in the correct order (start to goal)

def waypoint_planner(waypoint_indices, adj, params):
    full_path_indices = []
    for start_idx, goal_idx in zip(waypoint_indices[:-1], waypoint_indices[1:]):
        segment = dijkstra_shortest_path(start_idx, goal_idx, adj)
        if full_path_indices:
            # Avoid repeating the first node of the segment
            full_path_indices.extend(segment[1:])
        else:
            full_path_indices.extend(segment)

    m = metrics(full_path_indices, params)
    metrics_table(m, title="Path metrics")

    return full_path_indices

def metrics(path_idx, params):
    gamma = params["gamma"]
    r     = params["r"]
    shape_dist = make_shape_dist(r)

    rs = r[path_idx]          # (m+1, P, 3)
    gs = gamma[path_idx]      # (m+1, f)
    m  = len(path_idx) - 1    # number of edges

    # --- effort (activation space) ---
    E_l2    = np.sum(np.sum(gs**2, axis=1))
    E_l2bar = E_l2 / (m+1)

    # --- smoothness (activation space) ---
    TV    = np.sum(np.linalg.norm(np.diff(gs, axis=0), axis=1))
    TVbar = TV / max(m,1)

    # --- tip path length (workspace) ---
    tips = rs[:, -1, :]                      # (m+1,3), last point = tip
    tip_segs = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    L_tip    = np.sum(tip_segs)
    L_tipbar = L_tip / max(m,1)

    # --- shape-space length (RMS metric) ---
    shape_segs = [shape_dist(path_idx[k], path_idx[k+1]) for k in range(m)]
    shape_segs = np.asarray(shape_segs, float)
    L_shape    = float(np.sum(shape_segs))
    L_shapebar = L_shape / max(m,1)

    out = {
        "num_nodes": int(m+1),
        # "num_edges": m,
        # energy-like
        "E_l2 (quadratic effort = energy)": E_l2,
        # "E_l2_per_node": E_l2bar,
        # smoothness
        "TV (total variation: smoothness of path)": TV,
        # "TV_per_edge": TVbar,
        # tip-space metrics
        "L_tip (workspace tip path length)": L_tip,
        # "L_tip_per_edge": L_tipbar,
        # shape-space metrics
        "L_shape (RMS shape-space path length)": L_shape,
        # "L_shape_per_edge": L_shapebar,
    }

    return out