import numpy as np
from scipy.stats import norm
from math import comb

def calcH(model):
    mu0, lambda_val, delta, sigma0, sigmav, alpha, beta = model.parameters

    max_cellcycles = 10
    max_R = 10
    max_G = 10

    timepoints = model.timepoints
    num_timepoints = len(timepoints)

    initialTimepointsList = model.initial_timepoints
    bottomTimepointsList = model.bottom_timepoints
    topTimepointsList = model.top_timepoints

    initialBranchPartialH = [np.zeros((num_timepoints, len(lst)-1)) for lst in initialTimepointsList]
    topBranchPartialH = [np.zeros((num_timepoints, len(lst)-1)) for lst in topTimepointsList]
    bottomBranchPartialH = [np.zeros((num_timepoints, len(lst)-1)) for lst in bottomTimepointsList]

    for i in range(num_timepoints):
        t = timepoints[i]
        Q = 0
        for r in range(max_R + 1):
            Q += Qr(mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha)

        frac_init = Qr(mu0, sigma0, sigmav, delta, lambda_val, t, 0, alpha) / Q

        for idx, array in enumerate(initialTimepointsList):
            cdf = norm.cdf(array, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
            initialBranchPartialH[idx][i, :] = np.diff(cdf) * frac_init

        for runs in range(1, max_R + 1):
            for idx, array in enumerate(topTimepointsList):
                cdf = norm.cdf(array + runs * lambda_val, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                topBranchPartialH[idx][i, :] += np.diff(cdf) * frac_init

        frac_rest_all = 0
        for r in range(1, max_R + 1):
            for g in range(1, r + 1):
                frac_rest = Mgr(mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha) / Q
                frac_rest_all += frac_rest

                if frac_rest > 1e-10:
                    trun_cdf = norm.cdf(r * lambda_val + (g-1) * delta - alpha, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                    trun_denom = 1 - trun_cdf

                    for idx, array in enumerate(topTimepointsList):
                        for runs in range(r + 1, max_R + 1):
                            cdf = norm.cdf(array + runs * lambda_val + g * delta, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                            portion = np.zeros_like(cdf)
                            portion[1:] = np.diff(cdf)
                            portion = np.where(trun_denom == 0, 0, portion - trun_cdf) / trun_denom
                            topBranchPartialH[idx][i, :] += portion[1:]

                    for idx, array in enumerate(bottomTimepointsList):
                        cdf = norm.cdf(array + r * lambda_val + g * delta, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                        portion = np.zeros_like(cdf)
                        portion[1:] = np.diff(cdf)
                        portion = np.where(trun_denom == 0, 0, portion - trun_cdf) / trun_denom
                        portion[portion < 0] = 0
                        bottomBranchPartialH[idx][i, :] += portion[1:]

    Hsegments = {}
    relations = model.relations

    for i in range(len(relations)):
        relation = relations[i]
        for idx in range(1, len(relation) - 1, 2):
            label = relation[idx]
            num = int(relation[idx + 1])
            if label == 'i':
                matrix = initialBranchPartialH[num]
            elif label == 't':
                matrix = topBranchPartialH[num]
            elif label == 'b':
                matrix = bottomBranchPartialH[num]

            if idx == 1:
                Hsegments[i] = matrix
            else:
                Hsegments[i] += matrix

    H = np.hstack(list(Hsegments.values()))

    Hpos = {}
    cur_start = 0
    for i in range(len(Hsegments)):
        cur_len = Hsegments[i].shape[1]
        cur_end = cur_start + cur_len
        Hpos[i] = [cur_start, cur_end]
        cur_start = cur_end

    # Scale the final matrix such that each row has an equal sum
    for i in range(H.shape[0]):
        w = np.sum(H[i, :])
        H[i, :] = H[i, :] / w

    print(topBranchPartialH[0][4])

    return H, Hsegments, Hpos

def Qr(mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha):
    START = 1000
    if r == 0:
        return START
    else:
        N = 0
        for i in range(r):
            normval = 1 - norm.cdf(r * lambda_val + i * delta - alpha, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
            N += normval * START * comb(r-1, i)
        return N

def Mgr(mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha):
    START = 1000
    if g == 0:
        if r == 0:
            return START
        else:
            return 0
    elif g > 0:
        if r < g:
            return 0
        else:
            normval = 1 - norm.cdf(r * lambda_val + (g-1) * delta - alpha, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
            return normval * START * comb(r-1, g-1)
    else:
        print('Error: g < 0')
