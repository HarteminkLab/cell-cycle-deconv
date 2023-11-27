from math import comb
from matplotlib import pyplot as plt
from scipy.stats import norm
from helpers import calcH, createF, get_wavelet_kernel

import cvxpy as cp
import numpy as np

class Model:
    """A model class to deconvolve gene expression data from CLOCCS cell cycle.

    Attributes:
        gene_name (str): Name of gene to deconvolve.
        gamma (float): Regularization parameter.
        orf_name (str): Name of ORF mapped to the specified gene.
        orf_id (int): Index of the specified ORF.
        g (list of float): Measured time series population data.
    """
    
    def __init__(self, config, gene_name, gamma):
        self.config = config
        self.gene_name = gene_name
        self.gamma = gamma

        # initialize orf identifications
        try:
            self.orf_name = config.gene_orf_map[gene_name]
        except KeyError:
            print(f'Gene: {gene_name} is not found in gene to orf mapping data.')
        try:
            self.orf_id = config.orf_index_map[self.orf_name]
        except KeyError:
            print(f'ORF: {self.orf_name} is not found in list of genes data.')

        # initialize g from datasets corresponding to orf_id
        try:
            g1 = config.data_wt1[self.orf_id, :]
        except IndexError:
            print(f'ORF ID: {self.orf_id} is not a valid row in WT1 dataset.')
        try:
            g2 = config.data_wt2[self.orf_id, :]
        except IndexError:
            print(f'ORF ID: {self.orf_id} is not a valid row in WT2 dataset.')
        self.g = np.concatenate((g1, g2))

        self.initial_phase_map, self.top_phase_map, self.bottom_phase_map = config.intervals_wt1[-1]
        H1, self.Hpos = calcH(config.intervals_wt1, config.WT1_TIMEPOINTS)
        H2, _ = calcH(config.intervals_wt2, config.WT2_TIMEPOINTS)
        self.H = np.concatenate((H1, H2))

    def deconvolve(self):
        WAVETYPE, WAVEPAR = "Symmlet", 5

        f_initial, f_initial_list = createF(self.Hpos, self.initial_phase_map)
        f_top, f_top_list = createF(self.Hpos, self.top_phase_map)
        f_bottom, f_bottom_list = createF(self.Hpos, self.bottom_phase_map)

        f_it = []
        for phase in self.initial_phase_map.values():
            se = self.Hpos[phase[1]]
            f_it.extend([e for e in range(se[0], se[1])])
        for phase in self.top_phase_map.values():
            se = self.Hpos[phase[1]]
            f_it.extend([e for e in range(se[0], se[1])])
        f_it = np.array(f_it)

        f_b = []
        for phase in self.bottom_phase_map.values():
            se = self.Hpos[phase[1]]
            f_b.extend([e for e in range(se[0], se[1])])
        f_b = np.array(f_b)

        f_final = np.zeros(self.H.shape[1])

        # right mirroring
        f_b_mirror = np.concatenate((f_b, f_b))
        f_it_mirror = np.concatenate((f_it, np.flip(f_it)))
        factor_fb = 1.5

        W1 = get_wavelet_kernel(WAVETYPE, len(f_it_mirror), WAVEPAR)
        W2 = get_wavelet_kernel(WAVETYPE, len(f_b), WAVEPAR)
        W2pad = np.zeros((len(f_b), len(f_b)))
        W2 = np.concatenate((np.concatenate((W2, W2pad)), np.concatenate((W2pad, np.fliplr(W2)))), axis=1)

        # Convex optimization
        f_right = cp.Variable(self.H.shape[1])
        objective_right = cp.Minimize(cp.square(cp.pos(cp.norm(self.H@f_right/self.g - 1))) 
                                + self.gamma * (cp.norm(W1@f_right[f_it_mirror], 1) 
                                + factor_fb * cp.norm(W2@f_right[f_b_mirror], 1))/self.g.mean())
        constraints_right = [f_right >= 0]
        prob_right = cp.Problem(objective_right, constraints_right)
        result_right = prob_right.solve(solver=cp.CLARABEL)

        # f_final[f_it[:len(f_it)//2]] = f_right.value[f_it[:len(f_it)//2]]
        # f_b_1 = f_right.value[f_b]
        # f_it_1 = f_right.value[f_it]

        # left mirroring
        # f_it_mirror = np.concatenate((np.flip(f_it), f_it))

        # f_left = cp.Variable(self.H.shape[1])
        # objective_left = cp.Minimize(cp.square(cp.pos(cp.norm(self.H@f_left/self.g - 1))) 
        #                         + self.gamma * (cp.norm(W1@f_left[f_it_mirror], 1) 
        #                         + factor_fb * cp.norm(W2@f_left[f_b_mirror], 1)/self.g.mean()))
        # constraints_left = [f_left >= 0]
        # prob_left = cp.Problem(objective_left, constraints_left)
        # result_left = prob_left.solve(solver=cp.CLARABEL)
        # f_final[f_it[len(f_it)//2:]] = f_left.value[f_it[len(f_it)//2:]]
        # f_b_2 = f_left.value[f_b]
        # f_it_2 = f_left.value[f_it]

        # f_final[f_b] = (f_b_1 + f_b_2) / 2

        # f = f_final

        pred_g = np.matmul(self.H, f_right.value)
        
        W1 = get_wavelet_kernel(WAVETYPE, len(f_it), WAVEPAR)
        W2 = get_wavelet_kernel(WAVETYPE, len(f_b), WAVEPAR)
        sn = (np.linalg.norm(np.matmul(W1, f_right.value[f_it]), 1) + np.linalg.norm(np.matmul(W2, f_right.value[f_b]), 1)) / np.mean(self.g)
        rn = np.square(np.clip(np.linalg.norm(np.matmul(self.H, f_right.value) / self.g - 1), 0, None))

        print(sn)
        print(rn)
