import numpy as np


def estimate_incumbent(y):
    curr_inc = np.inf
    traj = []

    for yi in y:
        if yi < curr_inc:
            curr_inc = yi
        traj.append(curr_inc)
    return traj