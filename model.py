import constants
import cvxpy as cp
from matplotlib import pyplot as plt
import numpy as np

import temp


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

        (self.parameters, self.relations, self.initial_timepoints, 
        self.top_timepoints, self.bottom_timepoints, self.initial_phase_map, 
        self.top_phase_map, self.bottom_phase_map) = config.model_intervals_1
        self.timepoints = config.WT1_TIMEPOINTS
        
    def deconvolve(self):
        # Load datasets ported from MATLAB
        calculatedH, _, _ = temp.calcH(self)
        calcH = np.concatenate((calculatedH, calculatedH))
        H = constants.H

        return

        f_initial = constants.F_INITIAL
        f_top = constants.F_TOP
        f_bottom = constants.F_BOTTOM
        padding = constants.PADDING

        W1 = constants.W1
        W2 = constants.W2
        W3 = constants.W3

        # Convex optimization
        f = cp.Variable(H.shape[1] + padding)
        objective = cp.Minimize(cp.square(cp.pos(cp.norm(H@f[0:H.shape[1]]/self.g.T - 1)))
                          + self.gamma * (cp.norm(W1@f[f_initial], 1) 
                                          + cp.norm(W2@f[f_top], 1) 
                                          + cp.norm(W3@f[f_bottom], 1))/self.g.mean())
        constraints = [f >= 0]
        prob = cp.Problem(objective, constraints)
        print(cp.installed_solvers())
        result = prob.solve(solver=cp.CLARABEL, verbose=True)
        mse = np.square(f.value - constants.F_PADDED).mean()
        print(f'MSE: {mse}')
        print(f'Objective: {objective.value}')

        # Plot F ported from MATLAB versus solved through CVXPY
        plt.plot(f.value, label='CP f')
        plt.plot(constants.F_PADDED, label='MATLAB f')
        plt.legend()
        plt.show()

