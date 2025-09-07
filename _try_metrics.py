import numpy as np
import pandas as pd
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

r0 = r[-1,:,:] ; r3 = r[4,:,:]
waypoints = [r0, r3] # good for energy stuff because gamma has higher absolute magnitude than necessary
nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

path_indices = waypoint_planner(waypoint_indices, adj=adj_basic)

def path_pairs(path_idx):
    return list(zip(path_idx[:-1], path_idx[1:]))

def path_metrics(path_idx, r, gamma, shape_dist_fn, alpha, beta, delta,
                 sdf_fn=None):
    pairs = path_pairs(path_idx)

    # component terms
    shape_terms   = np.array([shape_dist_fn(i,j) for i,j in pairs])
    mag_terms     = np.array([0.5*(gamma[i]@gamma[i] + gamma[j]@gamma[j]) for i,j in pairs])
    smooth_terms  = np.array([np.linalg.norm(gamma[j]-gamma[i]) for i,j in pairs])

    # totals (the objective you optimized)
    C_shape   = shape_terms.sum()
    C_mag     = mag_terms.sum()
    C_smooth  = smooth_terms.sum()
    C_total   = alpha*C_shape + beta*C_mag + delta*C_smooth

    # worst jump for diagnostics
    max_jump  = smooth_terms.max() if len(smooth_terms) else 0.0

    # clearance metrics (if you have an SDF)
    c_min = None; c_mean = None; collided = None
    if sdf_fn is not None:
        # evaluate SDF on all centerline points along the path
        traj_pts = r[path_idx].reshape(-1, 3)           # (n*P, 3)
        sdf_vals = sdf_fn(traj_pts)
        c_min  = float(sdf_vals.min())
        c_mean = float(sdf_vals.mean())
        collided = c_min < 0.0

    # per-step normalization (optional, useful for fair comparisons)
    steps = max(len(pairs), 1)
    per_step = {
        "shape_per_step":   C_shape/steps,
        "mag_per_step":     C_mag/steps,
        "smooth_per_step":  C_smooth/steps,
        "total_per_step":   C_total/steps
    }

    return {
        "total_cost": C_total,
        "shape_cost": C_shape,
        "mag_cost": C_mag,
        "smooth_cost": C_smooth,
        "max_activation_jump": max_jump,
        "num_nodes": len(path_idx),
        "num_edges": len(pairs),
        "per_step": per_step,
        "clearance_min": c_min,
        "clearance_mean": c_mean,
        "collided": collided
    }

def metrics_table(metrics_dict, title=None, floatfmt=".3f"):
    """
    Pretty-print path metrics as a pandas DataFrame.

    Parameters
    ----------
    metrics_dict : dict
        Output of path_metrics() or similar dictionary of metrics.
    title : str, optional
        Title string to print above the table.
    floatfmt : str
        Format string for floats, e.g. ".3f".

    Returns
    -------
    df : pandas.DataFrame
    """
    # flatten nested 'per_step' dict if present
    flat = metrics_dict.copy()
    if "per_step" in flat:
        for k, v in flat["per_step"].items():
            flat[f"{k}"] = v
        del flat["per_step"]

    # make DataFrame with one row
    df = pd.DataFrame([flat])

    # format floats nicely
    with pd.option_context('display.float_format', lambda x: f"{x:{floatfmt}}"):
        if title:
            print(f"\n=== {title} ===")
        print(df.T.rename(columns={0: "Value"}))  # transpose for vertical view
    return df


metrics = path_metrics(path_indices, r, gamma,
                       shape_dist_fn=make_shape_dist(r),
                       alpha=1.0, beta=0.0, delta=0.0,
                       sdf_fn=None)

metrics_table(metrics, title="Path Metrics")