import numpy as np
import torch

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


def make_edge_weight_basic(r):
    """
    Creates a function to compute the weight of an edge between two points.

    Parameters:
        r (array-like): A list or array of points.
        gamma (array-like): A list or array of activation values.

    Returns:
        w (function): A function that computes the weight of an edge between two points i and j.
    """
    shape_dist = make_shape_dist(r)  # Create the shape distance function
    def w(i, j):
        dij = shape_dist(i, j)  # Shape distance between points i and j
        return dij
    return w


def make_edge_weight_tbd(r, gamma, alpha=1.0, beta=0.0, lam=0.0):
    """
    Creates a function to compute the weight of an edge between two points.

    Parameters:
        r (array-like): A list or array of points.
        gamma (array-like): A list or array of activation values.
        alpha (float): Weight for the shape distance term.
        beta (float): Weight for the magnitude of gamma.
        lam (float): Weight for the difference in gamma values.

    Returns:
        w (function): A function that computes the weight of an edge between two points i and j.
    """
    shape_dist = make_shape_dist(r)  # Create the shape distance function
    def w(i, j):
        dij = shape_dist(i, j)  # Shape distance between points i and j
        c_mag = 0.5 * (np.linalg.norm(gamma[i]) + np.linalg.norm(gamma[j]))  # Average magnitude of gamma
        c_dg = np.linalg.norm(gamma[j] - gamma[i])  # Difference in gamma values
        # Compute the edge weight as a weighted sum of the terms
        return alpha * dij + beta * c_mag + lam * c_dg
    return w

def make_edge_weight_tbd_updated(r, gamma, alpha=1.0, beta=0.0, lam=0.0):
    """
    Creates a function to compute the weight of an edge between two points.

    Parameters:
        r (array-like): A list or array of points.
        gamma (array-like): A list or array of activation values.
        alpha (float): Weight for the shape distance term.
        beta (float): Weight for the magnitude of gamma.
        lam (float): Weight for the difference in gamma values.

    Returns:
        w (function): A function that computes the weight of an edge between two points i and j.
    """
    shape_dist = make_shape_dist(r)  # Create the shape distance function
    def w(i, j):
        dij = shape_dist(i, j)  # Shape distance between points i and j
        c_mag = 0.5 * (np.matmul(gamma[i].transpose(), gamma[i]) + np.matmul(gamma[j].transpose(), gamma[j])) 
        c_dg = np.linalg.norm(gamma[j] - gamma[i]) ** 2
        # Compute the edge weight as a weighted sum of the terms
        return alpha * dij + beta * c_mag + lam * c_dg
    return w

def make_edge_weight_sdf(r, node_clearance, alpha=1.0, mu=0.5, eps=0.01):
    """
    Creates a function to compute the weight of an edge, considering shape distance and clearance.

    Parameters:
        r (array-like): A list or array of points.
        node_clearance (array-like): Clearance values for each node.
        alpha (float): Weight for the shape distance term.
        mu (float): Weight for the clearance penalty term.
        eps (float): Threshold for clearance penalty.

    Returns:
        w (function): A function that computes the weight of an edge between two points i and j.
    """
    shape_dist = make_shape_dist(r)  # Create the shape distance function
    def clearance_penalty(c):
        # Compute a penalty for low clearance values
        return 0.0 if c >= eps else 1.0 / (eps - c)
    def w(i, j):
        base = shape_dist(i, j)  # Shape distance between points i and j
        # Compute the average clearance penalty for the two nodes
        csoft = 0.5 * (clearance_penalty(node_clearance[i]) +
                       clearance_penalty(node_clearance[j]))
        # Compute the edge weight as a weighted sum of the shape distance and clearance penalty
        return alpha * base + mu * csoft
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