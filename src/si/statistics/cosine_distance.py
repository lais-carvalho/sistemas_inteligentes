import numpy as np


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the cosine distance of a point (x) to a set of points y.
        for x_y1:
        dot_product = (x1 * y11) + (x2 * y12) + ...+ (xn * y1n)
        norm_A= sqrt((x1^2) + (x2^2) + ... + (xn^2))
        norm_B = sqrt((y11^2) + (y12^2) + ... + (y1n^2))
        distance = 1 - (dot_product/(norm_A * norm_B))
        ...

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Cosine distance for each point in y.
    """
    dot_product = np.dot(y, x)

    norm_x = np.linalg.norm(x)

    norm_y = np.linalg.norm(y, axis=1)

    distances = 1 - (dot_product / (norm_x * norm_y))

    return distances