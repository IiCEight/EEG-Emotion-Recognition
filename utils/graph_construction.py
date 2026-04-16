import numpy as np
from utils.graphConstructionFromStandard import SEED_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT


def get_domain_general_adj(
    channel_list: list = SEED_CHANNEL_LIST,
    location_dict: dict = STANDARD_1005_CHANNEL_LOCATION_DICT,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Domain-general graph connectivity based on unit-sphere projection of electrode positions.

    Each electrode's 3D (x,y,z) position is projected onto a unit sphere, then the
    pairwise connectivity is computed as:

        A[i,j] = arccos( dot(u_i, u_j) )   (angle between unit vectors, in radians)

    Edges with value <= threshold are zeroed out (considered non-significant).
    Ref: standard 10-20 electrode positions mapped to unit sphere, arccos correlation.

    Args:
        channel_list: Ordered list of electrode names.
        location_dict: Dict mapping electrode name -> [x, y, z].
        threshold: Minimum connectivity value to retain an edge (default: 0.1).

    Returns:
        adj: (N, N) numpy array of connectivity values.
    """
    n = len(channel_list)
    # Build unit vectors for each channel
    unit_vecs = []
    valid = []
    for name in channel_list:
        if name in location_dict:
            pos = np.array(location_dict[name], dtype=np.float64)
            norm = np.linalg.norm(pos)
            unit_vecs.append(pos / (norm + 1e-8))
            valid.append(True)
        else:
            unit_vecs.append(None)
            valid.append(False)

    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(n):
            if not valid[j]:
                continue
            # Clamp dot product to [-1, 1] to avoid arccos domain errors
            dot = np.clip(np.dot(unit_vecs[i], unit_vecs[j]), -1.0, 1.0)
            connectivity = np.arccos(dot)
            adj[i, j] = connectivity if connectivity > threshold else 0.0

    return adj


if __name__ == '__main__':
    adj = get_domain_general_adj()
    print("Shape:", adj.shape)
    print("Non-zero ratio:", np.sum(adj > 0) / adj.size)
    print("Min/Max non-zero:", adj[adj > 0].min(), adj[adj > 0].max())
    print("Sample row 0:", adj[0])
