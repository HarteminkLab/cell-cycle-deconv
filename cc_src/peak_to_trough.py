
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def combine_ptr_score(c, d, weight):
    score = np.power(c, weight)*np.power(d, 1-weight);
    return score


def compute_ptr(model, gene_f):

    cg1_indices = np.concatenate([model.config.phase_columns['CG1'], 
                                  model.config.phase_columns['postG1']])
    dg1_indices = np.concatenate([model.config.phase_columns['DG1'], 
                                  model.config.phase_columns['postG1']])

    cg1_f = gene_f[cg1_indices]
    dg1_f = gene_f[dg1_indices]

    c_timepoints = model.config.get_timepoints_for_branch('t')
    d_timepoints = model.config.get_timepoints_for_branch('b')

    _, scaled_cg1_f = rescale(c_timepoints, cg1_f)
    _, scaled_dg1_f = rescale(d_timepoints, dg1_f)

    # --------------

    cg1_20, cg1_80 = np.quantile(scaled_cg1_f, [0.2, 0.8])
    dg1_20, dg1_80 = np.quantile(scaled_dg1_f, [0.2, 0.8])

    cg1_20 = max(cg1_20, 1)
    dg1_20 = max(dg1_20, 1)

    # Need to normalize/timepoint scale.

    weight = 2./3.;
    cptr = cg1_80/cg1_20;
    dptr = dg1_80/dg1_20;

    combinedPtr = combine_ptr_score(cptr, dptr, weight)
    return combinedPtr


def rescale(x, y, interval=1):
    """
    Rescale x and y data points based on a specified interval.

    Parameters:
    x (array-like): Original x data points.
    y (array-like): Original y data points.
    interval (float, optional): Interval for rescaling. Default is 1.

    Returns:
    newx (numpy.ndarray): Rescaled x data points.
    newy (numpy.ndarray): Rescaled y data points.
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Expect at least 2 data points in both x and y")

    len_new = int(np.ceil((x[-1] - x[0]) / interval))
    newx = np.linspace(x[0], x[0] + len_new * interval, len_new + 1)
    newy = np.zeros(len_new + 1)

    for idx in range(len_new):
        cur_x = newx[idx]
        # Find the left and the right neighbors
        left_pos = np.where(x <= cur_x)[0][-1]
        right_pos = np.where(x >= cur_x)[0][0]

        x_left = x[left_pos]
        y_left = y[left_pos]
        x_right = x[right_pos]
        y_right = y[right_pos]

        if x_left == x_right:
            newy[idx] = y_left
        else:
            newy[idx] = y_left + (cur_x - x_left) / (x_right - x_left) * (y_right - y_left)

    # Extend the last point as in the original function
    newx[-1] = newx[-2] + interval
    newy[-1] = newy[-2]

    return newx, newy
