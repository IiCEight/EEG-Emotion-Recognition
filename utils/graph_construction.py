import numpy as np
from utils.graphConstructionFromStandard import SEED_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT


def get_domain_general_adj(
    channel_list: list = SEED_CHANNEL_LIST,
    location_dict: dict = STANDARD_1005_CHANNEL_LOCATION_DICT,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Domain-general graph connectivity via arccos correlation on a shared sphere of radius r.

    All electrodes are placed on a sphere of radius r (the mean norm of all electrode
    positions). The pairwise connectivity follows equation (2) from the paper:

        A[i,j] = arccos( (x_i*x_j + y_i*y_j + z_i*z_j) / r^2 )

    i.e., arccos of the dot product of the r-normalized position vectors.
    Edges with value <= threshold are zeroed out (paper: threshold = 0.1).

    Args:
        channel_list: Ordered list of electrode names.
        location_dict: Dict mapping electrode name -> [x, y, z].
        threshold: Minimum connectivity value to retain an edge (default: 0.1).

    Returns:
        adj: (N, N) numpy array of connectivity values.
    """
    n = len(channel_list)

    # Collect valid positions and compute shared radius r
    positions = []
    valid = []
    for name in channel_list:
        if name in location_dict:
            positions.append(np.array(location_dict[name], dtype=np.float64))
            valid.append(True)
        else:
            positions.append(None)
            valid.append(False)

    valid_norms = [np.linalg.norm(p) for p, v in zip(positions, valid) if v]
    r = np.mean(valid_norms)

    # Scale all positions by shared radius r
    scaled = [p / r if v else None for p, v in zip(positions, valid)]

    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(n):
            if not valid[j]:
                continue
            # dot(pos_i/r, pos_j/r) = dot(pos_i, pos_j) / r^2
            dot = np.clip(np.dot(scaled[i], scaled[j]), -1.0, 1.0)
            connectivity = np.arccos(dot)
            adj[i, j] = connectivity if connectivity > threshold else 0.0

    return adj


def get_weighted_adj(weights_path: str, threshold: float = 0.0003) -> np.ndarray:
    """
    Apply pre-computed per-electrode weights to the domain-general adjacency matrix,
    then zero out weak edges below threshold.

    A'[i,j] = w[i] * w[j] * A[i,j],  set to 0 if A'[i,j] < threshold

    Args:
        weights_path: Path to .npy file containing weight vector of shape (62,).
        threshold: Minimum edge value to retain (default: 0.0003).

    Returns:
        adj: (62, 62) numpy array.
    """
    w = np.load(weights_path).flatten()  # shape (62,)
    A = get_domain_general_adj()         # shape (62, 62)
    weighted = np.outer(w, w) * A
    weighted[weighted < threshold] = 0.0
    return weighted


if __name__ == '__main__':
    adj = get_domain_general_adj()
    print("Shape:", adj.shape)
    print("Non-zero ratio:", np.sum(adj > 0) / adj.size)
    print("Min/Max non-zero:", adj[adj > 0].min(), adj[adj > 0].max())
    print("Sample row 0:", adj[0])
