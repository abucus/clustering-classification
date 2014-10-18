import numpy as np


def distance_point_to_hyperplane(p, coefs, intercept):
    return np.abs(np.dot(p, coefs) + intercept) / np.linalg.norm(coefs)
