"""
"""

__all__ = [
    "build_B", 
    "build_L_from_B",
    "gen_Z_random", 
    "gen_Z_distance", 
    "gen_Z_split", 
    "gen_Z_ring",
    ]

import numpy as np
import random

#######################################################################################
# Laplacian matrix


def build_B(list_edges: list[tuple[int, int]], N: int) -> np.ndarray:
    """
    Generate the incidence matrix for a graph.

    Parameters
    ----------
    list_edges : list of tuple[int, int]
        List of edges, where each edge is represented as a tuple (tail, head).
    N : int
        Number of nodes in the graph.

    Returns
    -------
    np.ndarray
        Incidence matrix of shape (N, E), where E is the number of edges.
    
    Note
    ----
    This definition of the incidence matrix is for computing z = sum(x_i - x_j).
    If you need z = sum(x_j - x_i), multiply B by -1.
    """
    B = np.zeros((N, len(list_edges)))
    for k, (tail, head) in enumerate(list_edges):
        B[tail, k] = 1
        B[head, k] = -1
    return B


def build_L_from_B(B: np.ndarray, W : np.ndarray = None) -> np.ndarray:
    """
    Compute the Laplacian matrix from the incidence matrix.

    The Laplacian matrix is defined as: L = B W B^T

    Parameters
    ----------
    B : np.ndarray
        Incidence matrix of shape (N, E), where N is the number of nodes 
        and E is the number of edges.
    W : np.ndarray, optional
        Diagonal weight matrix of shape (E, E), where each diagonal entry 
        corresponds to the weight of an edge. If None, assumes uniform weights.
        
    Returns
    -------
    np.ndarray
        Laplacian matrix of shape (N, N).

    Note
    ----
    If no weight matrix W is provided, it defaults to the unweighted case: L = B B^T.
    """
    if W is None:
        return B @ B.T
    else:
        return B @ W @ B.T


#######################################################################################
# Graph generators


def gen_Z_random(N: int, rounds: int = 1, seed: int = None) -> list[tuple[int, int]]:
    """
    Generate a random connected undirected graph using a heuristic.

    This function ensures that the generated graph is connected by iteratively 
    selecting edges between visited and non-visited nodes.

    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    rounds : int, optional
        Number of times to attempt adding additional edges (default is 1).
    seed : int, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    list of tuple[int, int]
        List of edges, represented as a tuple (tail, head), forming a connected graph.
    """
    if seed is not None:
        random.seed(seed)

    Z = []

    while rounds:
        non_visited_nd = set(range(N))
        non_visited_nd.remove(0)
        visited_nd = {0}

        while len(non_visited_nd) != 0:
            i = random.choice(list(visited_nd))
            j = random.choice(list(non_visited_nd))
            visited_nd.add(j)
            non_visited_nd.remove(j)

            if (i, j) not in Z and (j, i) not in Z:
                Z.append((i, j))

        rounds -= 1

    return Z


def gen_Z_distance(P: np.ndarray, dist_thr: float) -> list[tuple[int, int]]:
    """
    Generate a graph based on a distance threshold heuristic.

    For each pair of nodes (i, j), if the distance between them (d_ij) is less than 
    or equal to the given threshold (dist_thr), an edge (i, j) is added to the graph.

    Parameters
    ----------
    P : np.ndarray
        An array of shape (N, D), where N is the number of nodes and D is the number 
        of dimensions. Each row represents the coordinates of a node in the space.
    dist_thr : float
        The distance threshold. If the distance between two nodes is less than or 
        equal to this value, an edge will be created between them.

    Returns
    -------
    list of tuple[int, int]
        List of edges, each represented as a tuple (i, j), where i and j are node indices.
    """
    y2 = np.sum(P**2, axis=1)
    x2 = y2.reshape(-1, 1)
    dist = np.sqrt(x2 - 2 * P @ P.T + y2)

    mask = dist + 2 * np.eye(dist.shape[0]) * dist_thr <= dist_thr
    Z = [(i, j) for i, j in zip(*np.where(mask))]
    return Z


def gen_Z_split(N: int, order: int, n_breaks: int = 0) -> list[tuple[int, int]]:
    """
    Split a fully connected graph into `n_breaks` smaller fully connected graphs.

    The graph is initially divided into `order` subgraphs. Afterward, a number of edges 
    (based on `n_breaks`) are removed from the generated graph to split it into disconnected subgraphs.

    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    order : int
        Number of subgraphs to create.
    n_breaks : int, optional
        Number of connections to remove in each subgraph (default is 0, meaning no edges are removed).

    Returns
    -------
    list of tuple[int, int]
        List of edges, represented as a tuple (i, j), forming the generated graph with subgraphs.
    """
    X = np.ones((N, 2))
    for i in range(order):
        if i != order - 1:
            X[i * int(N / order) : (i + 1) * int(N / order), :] = [i, i]
        else:
            X[i * int(N / order) :, :] = [i, i]

    y2 = np.sum(X**2, axis=1)
    x2 = y2.reshape(-1, 1)
    dist = np.sqrt(x2 - 2 * X @ X.T + y2)

    dist_thr = 0.1
    mask = dist + 2 * np.eye(dist.shape[0]) * dist_thr <= dist_thr

    Z = [(i, j) for i, j in zip(*np.where(mask))]

    # Remove some conections
    N_subgraph = int(N / order)
    edges_subgraph = int(2 * N_subgraph * (N_subgraph - 1) / 2)
    idx_to_remove = []
    if n_breaks > 0:
        for i in range(order):
            for j in range(n_breaks):
                idx = edges_subgraph * i + int(N * N_subgraph / n_breaks) * j
                idx_to_remove.append(idx)
    Z = [edge for i, edge in enumerate(Z) if i not in idx_to_remove]
    return Z


def gen_Z_ring(N: int) -> list[tuple[int, int]]:
    """
    Generate a ring-shaped graph where each node is connected to its two neighbors.

    The graph consists of N nodes arranged in a ring structure, where node i is connected 
    to node i+1, and node N-1 is connected to node 0 to form the ring.

    Parameters
    ----------
    N : int
        Number of nodes in the graph.

    Returns
    -------
    list of tuple[int, int]
        List of edges, where each edge is represented as a tuple (i, j), 
        forming a closed ring graph.
    """
    Z = [(i, i + 1) for i in range(N - 1)]
    Z.extend([(i + 1, i) for i in range(N - 1)])
    Z.append((N - 1, 0))
    Z.append((0, N - 1))
    return Z


#######################################################################################
