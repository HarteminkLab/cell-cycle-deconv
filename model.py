from math import comb
from matplotlib import pyplot as plt
from scipy.stats import norm

import constants
import cvxpy as cp
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
        H1, Hpos1 = self.calcH()
        (self.parameters, self.relations, self.initial_timepoints, 
        self.top_timepoints, self.bottom_timepoints, self.initial_phase_map, 
        self.top_phase_map, self.bottom_phase_map) = config.model_intervals_2
        self.timepoints = config.WT2_TIMEPOINTS
        H2, Hpos2 = self.calcH()
        self.H = np.concatenate((H1, H2))
        self.Hpos = Hpos1
        
    def deconvolve(self):
        # Load datasets ported from MATLAB
        WAVETYPE = "Symmlet"
        WAVEPAR = 5

        f_initial, f_top, f_bottom = self.createFs()

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

        W1 = temp.getWaveletKernel(WAVETYPE, len(f_it_mirror), WAVEPAR)
        W2 = temp.getWaveletKernel(WAVETYPE, len(f_b), WAVEPAR)
        W2pad = np.zeros(len(f_b))
        W2 = np.concatenate((np.concatenate((W2, W2pad), axis=1), np.concatenate((W2pad, np.fliplr(W2)), axis=1)), axis=0)
        
        print(W2.shape)

        return


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

    def createFs(self):
        f_initial = self.createFBranch(self.initial_phase_map)
        f_top = self.createFBranch(self.top_phase_map)
        f_bottom = self.createFBranch(self.bottom_phase_map)
        return f_initial, f_top, f_bottom
    
    def createFBranch(self, phaseMapping):
        f_partial = []
        for array in phaseMapping.values():
            phaseName = array[0]
            subintervalStartEnd = self.Hpos[array[1]]
            f_partial.extend([e for e in range(subintervalStartEnd[0], subintervalStartEnd[1])])
        return np.array(f_partial)

    def calcH(self):
        mu0, lambda_val, delta, sigma0, sigmav, alpha, beta = self.parameters

        max_cellcycles = 10
        max_R = 10
        max_G = 10

        timepoints = self.timepoints
        num_timepoints = len(timepoints)

        initialTimepointsList = self.initial_timepoints
        bottomTimepointsList = self.bottom_timepoints
        topTimepointsList = self.top_timepoints

        initialBranchPartialH = [np.zeros((num_timepoints, len(lst)-1)) for lst in initialTimepointsList]
        topBranchPartialH = [np.zeros((num_timepoints, len(lst)-1)) for lst in topTimepointsList]
        bottomBranchPartialH = [np.zeros((num_timepoints, len(lst)-1)) for lst in bottomTimepointsList]

        for i in range(num_timepoints):
            t = timepoints[i]
            Q = 0
            for r in range(max_R + 1):
                Q += self.Qr(mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha)

            frac_init = self.Qr(mu0, sigma0, sigmav, delta, lambda_val, t, 0, alpha) / Q

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
                    frac_rest = self.Mgr(mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha) / Q
                    frac_rest_all += frac_rest

                    if frac_rest > 1e-10:
                        trun_cdf = norm.cdf(r * lambda_val + (g-1) * delta - alpha, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                        trun_denom = 1 - trun_cdf

                        for idx, array in enumerate(topTimepointsList):
                            for runs in range(r + 1, max_R + 1):
                                cdf = norm.cdf(array + runs * lambda_val + g * delta, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                                if trun_denom == 0:
                                    portion = cdf * 0
                                else:
                                    portion = (cdf - trun_cdf) / trun_denom
                                topBranchPartialH[idx][i, :] += np.diff(portion) * frac_rest

                        for idx, array in enumerate(bottomTimepointsList):
                            cdf = norm.cdf(array + r * lambda_val + g * delta, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                            if trun_denom == 0:
                                portion = cdf * 0
                            else:
                                portion = (cdf - trun_cdf) / trun_denom
                            bottomBranchPartialH[idx][i, :] += np.diff(portion) * frac_rest

        Hsegments = {}
        relations = self.relations

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

        return H, Hpos
    
    
    def Qr(self, mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha):
        START = 1000
        if r == 0:
            return START
        else:
            N = 0
            for i in range(r):
                normval = 1 - norm.cdf(r * lambda_val + i * delta - alpha, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
                N += normval * START * comb(r-1, i)
            return N

    def Mgr(self, mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha):
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