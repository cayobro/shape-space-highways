import numpy as np
import torch
import pandas as pd

torch.manual_seed(0)

def make_shape_dist(r):
    """
    Creates a function to compute the shape distance between two points.

    Parameters:
        r (array-like): A list or array of points.

    Returns:
        shape_dist (function): A function that computes the distance between two points i and j.
    """
    def shape_dist(i, j):
        a = r[i]  # Point i
        b = r[j]  # Point j
        # Compute the mean squared distance between corresponding points in a and b
        return np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))
    return shape_dist


def make_edge_weight(params):
    r = params["r"]
    gamma = params["gamma"]
    alpha = params["alpha"]; beta = params["beta"]; delta = params["delta"]; 
    lamb = params["lamb"]; eps = params["eps"]
    node_clearance = params.get("node_clearance", None)  

    shape_dist = make_shape_dist(r)  # Create the shape distance function
    
    def clearance_penalty(c):
        # Compute a penalty for low clearance values
        return 0.0 if c >= eps else 1.0 / (eps - c) # TODO is that true? Plot it
    
    def w(i, j):
        dij = shape_dist(i, j)  # Shape distance between points i and j
        d_mag = 0.5 * (np.matmul(gamma[i].transpose(), gamma[i]) + np.matmul(gamma[j].transpose(), gamma[j])) 
        d_smooth = np.linalg.norm(gamma[j] - gamma[i])  # Difference in gamma values
        if node_clearance is None:
            d_sdf = 0
        else:
            d_sdf = 0.5 * (clearance_penalty(node_clearance[i]) +
                    clearance_penalty(node_clearance[j]))
        return alpha * dij + beta * d_mag + delta * d_smooth + lamb * d_sdf
    return w


def make_edge_sweep_checker(r, scene_sdf, s_list=(0.25, 0.5, 0.75)):
    """
    Creates a function to check if an edge is collision-free.

    Parameters:
        r (array-like): A list or array of points.
        scene_sdf (function): A signed distance field function that returns the distance to the nearest obstacle.
        s_list (tuple): A list of interpolation factors to check along the edge.

    Returns:
        ok (function): A function that checks if an edge is collision-free.
    """
    def ok(i, j):
        # Check intermediate points along the edge
        for s in s_list:
            # Interpolate between points i and j
            interp = (1.0 - s) * r[i] + s * r[j]
            # Check if the interpolated point is in collision
            if scene_sdf(interp).min() < 0.0:
                return False
        return True
    return ok

def path_pairs(path_idx):
    return list(zip(path_idx[:-1], path_idx[1:]))
 
def path_metrics(path_idx, terms, params):
    alpha = params["alpha"]; beta = params["beta"]
    delta = params["delta"]; lamb = params["lamb"]
    r = params["r"]
    sdf_fn = params.get("sdf_fn", None)

    pairs = list(zip(path_idx[:-1], path_idx[1:]))

    # sum components using the same 'terms' used for the graph
    comp_sums = {"dij":0.0, "d_mag":0.0, "d_smooth":0.0, "d_sdf":0.0}
    max_jump = 0.0
    for i,j in pairs:
        t = terms(i, j)
        for k in comp_sums: comp_sums[k] += t[k]
        max_jump = max(max_jump, t["d_smooth"])

    C_shape, C_mag, C_smooth, C_sdf = (comp_sums[k] for k in ("dij","d_mag","d_smooth","d_sdf"))
    C_total = alpha*C_shape + beta*C_mag + delta*C_smooth + lamb*C_sdf

    # optional SDF validation along the whole path (independent of node_clearance)
    c_min = c_mean = collided = None
    if sdf_fn is not None:
        traj_pts = r[path_idx].reshape(-1, 3)
        sdf_vals = sdf_fn(traj_pts)
        c_min = float(sdf_vals.min()); c_mean = float(sdf_vals.mean())
        collided = c_min < 0.0

    steps = max(len(pairs), 1)
    return {
        "total_cost": C_total,
        "shape_cost": C_shape,
        "mag_cost": C_mag,
        "smooth_cost": C_smooth,
        "sdf_cost": C_sdf,
        "max_activation_jump": max_jump,
        "num_nodes": len(path_idx),
        "num_edges": len(pairs),
        "shape_per_step": C_shape/steps,
        "mag_per_step": C_mag/steps,
        "smooth_per_step": C_smooth/steps,
        "sdf_per_step": C_sdf/steps,
        "total_per_step": C_total/steps,
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


# === OLD STUFF BELOW ===
# def make_edge_terms(params):
#     r = params["r"]
#     gamma = params["gamma"]
#     eps = params["eps"]
#     node_clearance = params.get("node_clearance", None)  

#     shape_dist = make_shape_dist(r)  # Create the shape distance function
    
#     def clearance_penalty(c):
#         # Compute a penalty for low clearance values
#         return 0.0 if c >= eps else 1.0 / (eps - c) # TODO is that true? Plot it
    
#     def terms(i, j):
#         dij = shape_dist(i, j)  # Shape distance between points i and j
#         d_mag = 0.5 * (np.matmul(gamma[i].transpose(), gamma[i]) + np.matmul(gamma[j].transpose(), gamma[j])) 
#         d_smooth = np.linalg.norm(gamma[j] - gamma[i])  # Difference in gamma values
#         if node_clearance is None:
#             d_sdf = 0
#         else:
#             d_sdf = 0.5 * (clearance_penalty(node_clearance[i]) +
#                     clearance_penalty(node_clearance[j]))
#         return {"dij": dij, "d_mag": d_mag, "d_smooth": d_smooth, "d_sdf": d_sdf}
#     return terms


# def make_edge_weight(terms, params):
#     alpha = params["alpha"]; beta = params["beta"]; delta = params["delta"]; 
#     lamb = params["lamb"]; 

#     def w(i, j):
#         t = terms(i, j)
#         return alpha * t["dij"] + beta * t["d_mag"] + delta * t["d_smooth"] + lamb * t["d_sdf"]
#     return w

# def make_edge_weight(params):
#     r = params["r"]
#     gamma = params["gamma"]
#     alpha = params["alpha"]; beta = params["beta"]; delta = params["delta"]; 
#     lamb = params["lamb"]; eps = params["eps"]
#     node_clearance = params.get("node_clearance", None)  

#     shape_dist = make_shape_dist(r)  # Create the shape distance function
    
#     def clearance_penalty(c):
#         # Compute a penalty for low clearance values
#         return 0.0 if c >= eps else 1.0 / (eps - c) # TODO is that true? Plot it
    
#     def w(i, j):
#         dij = shape_dist(i, j)  # Shape distance between points i and j
#         d_mag = 0.5 * (np.matmul(gamma[i].transpose(), gamma[i]) + np.matmul(gamma[j].transpose(), gamma[j])) 
#         d_smooth = np.linalg.norm(gamma[j] - gamma[i])  # Difference in gamma values
#         if node_clearance is None:
#             d_sdf = 0
#         else:
#             d_sdf = 0.5 * (clearance_penalty(node_clearance[i]) +
#                     clearance_penalty(node_clearance[j]))
#         return alpha * dij + beta * d_mag + delta * d_smooth + lamb * d_sdf
#     return w
# def make_edge_weight_basic(r):
#     """
#     Creates a function to compute the weight of an edge between two points.

#     Parameters:
#         r (array-like): A list or array of points.
#         gamma (array-like): A list or array of activation values.

#     Returns:
#         w (function): A function that computes the weight of an edge between two points i and j.
#     """
#     shape_dist = make_shape_dist(r)  # Create the shape distance function
#     def w(i, j):
#         dij = shape_dist(i, j)  # Shape distance between points i and j
#         return dij
#     return w


# def make_edge_weight_energy(r, gamma, alpha=1.0, beta=1.0, delta=1.0):
#     """
#     Creates a function to compute the weight of an edge between two points.

#     Parameters:
#         r (array-like): A list or array of points.
#         gamma (array-like): A list or array of activation values.
#         alpha (float): Weight for the shape distance term.
#         beta (float): Weight for the magnitude of gamma.
#         lam (float): Weight for the difference in gamma values.

#     Returns:
#         w (function): A function that computes the weight of an edge between two points i and j.
#     """
#     shape_dist = make_shape_dist(r)  # Create the shape distance function
#     def w(i, j):
#         dij = shape_dist(i, j)  # Shape distance between points i and j
#         d_mag = 0.5 * (np.matmul(gamma[i].transpose(), gamma[i]) + np.matmul(gamma[j].transpose(), gamma[j])) 
#         d_smooth = np.linalg.norm(gamma[j] - gamma[i])  # Difference in gamma values
#         # Compute the edge weight as a weighted sum of the terms
#         return alpha * dij + beta * d_mag + delta * d_smooth
#     return w


# def make_edge_weight_sdf(r, node_clearance, alpha=1.0, mu=0.5, eps=0.01):
#     """
#     Creates a function to compute the weight of an edge, considering shape distance and clearance.

#     Parameters:
#         r (array-like): A list or array of points.
#         node_clearance (array-like): Clearance values for each node.
#         alpha (float): Weight for the shape distance term.
#         mu (float): Weight for the clearance penalty term.
#         eps (float): Threshold for clearance penalty.

#     Returns:
#         w (function): A function that computes the weight of an edge between two points i and j.
#     """
#     shape_dist = make_shape_dist(r)  # Create the shape distance function
#     def clearance_penalty(c):
#         # Compute a penalty for low clearance values
#         return 0.0 if c >= eps else 1.0 / (eps - c) # TODO is that true? Plot it
#     def w(i, j):
#         base = shape_dist(i, j)  # Shape distance between points i and j
#         # Compute the average clearance penalty for the two nodes
#         csoft = 0.5 * (clearance_penalty(node_clearance[i]) +
#                        clearance_penalty(node_clearance[j]))
#         # Compute the edge weight as a weighted sum of the shape distance and clearance penalty
#         return alpha * base + mu * csoft
#     return w

# def make_edge_weight_multi(r, gamma, node_clearance, alpha=1.0, beta=1.0, delta=1.0,  lamb=1.0, eps=0.01):
#     shape_dist = make_shape_dist(r)  # Create the shape distance function
#     def clearance_penalty(c):
#         # Compute a penalty for low clearance values
#         return 0.0 if c >= eps else 1.0 / (eps - c) # TODO is that true? Plot it
#     def w(i, j):
#         dij = shape_dist(i, j)  # Shape distance between points i and j
#         d_mag = 0.5 * (np.matmul(gamma[i].transpose(), gamma[i]) + np.matmul(gamma[j].transpose(), gamma[j])) 
#         d_smooth = np.linalg.norm(gamma[j] - gamma[i])  # Difference in gamma values
#         d_sdf = 0.5 * (clearance_penalty(node_clearance[i]) +
#                        clearance_penalty(node_clearance[j]))
#         # Compute the edge weight as a weighted sum of the terms
#         return alpha * dij + beta * d_mag + delta * d_smooth + lamb * d_sdf
#     return w
