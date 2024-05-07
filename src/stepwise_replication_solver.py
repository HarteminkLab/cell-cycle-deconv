
import cvxpy as cp

import pandas as pd
import numpy as np

from src.helpers import calcH
from matplotlib import pyplot as plt
from src.utils import print_fl
from src.global_config import GlobalConstants
from src.config import load_yl_rg1_vst_config

class StepReplicationChromatinDeconvolveSolver:
	"""
	Compute an estimate for a single point in the deconvolution timing profiles in which
	replication occurs.
	"""

	def __init__(self, mnase_analysis_rep1, mnase_analysis_rep2):

		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.mnase_analysis_rep2 = mnase_analysis_rep2

		self.config = load_yl_rg1_vst_config(1)
		self.H, _ = calcH(self.config.intervals_wt1, 
			GlobalConstants.CHROM_WT1_TIMEPOINTS)

	def select_bin(self):

		from src.sgd import get_orfname

		early_bin = self.mnase_analysis_rep1.normalized_bin_curves\
			.loc[get_orfname('VPS8')]
		late_bin = self.mnase_analysis_rep1.normalized_bin_curves\
			.loc[get_orfname('SSK22')]

		# use copy number dataset to determine range of values
		copy_num_file = f'datasets/computed_mnase/dna_copy_scaling_rep1.csv'
		copy_num_rep1 = pd.read_csv(copy_num_file).set_index("Unnamed: 0")

		# the values of g are normalized to be between 1 and 2.
		# So normalize them to be within the range of the copy number
		# values. 
		# todo: Should rethink how the bins should be normalized.
		#       As they should reflect the actual copy number of the sample
		#       including the halted cells proportion, meaning
		#       the max will never actually get to 2.0 in the experiment.
		copy_min, copy_max = copy_num_rep1.min().scale, copy_num_rep1.max().scale
		scale_g = (copy_max-copy_min)

		g_early = late_bin.values.reshape((-1, 1))*scale_g+copy_max
		g_late = early_bin.values.reshape((-1, 1))*scale_g+copy_min
		
		self.g = np.hstack([g_early, g_late])


	def solve(self):

		g = self.g

		n, m = self.H.shape
		n, u = self.g.shape

		config = self.config
		transition_point = cp.Variable(integer=True)

		f_dg1_i = config.get_Hpositions_for_phase('DG1')
		f_rg1_i = config.get_Hpositions_for_phase('RG1')
		f_cg1_i = config.get_Hpositions_for_phase('CG1')
		f_pg1_i = config.get_Hpositions_for_phase('postG1')

		f_rg1 = np.zeros((len(f_dg1_i), u)).astype(bool)
		f_cg1 = np.zeros((len(f_dg1_i), u)).astype(bool)
		f_dg1 = np.zeros((len(f_dg1_i), u)).astype(bool)
		f_pg1 = cp.Variable((len(f_pg1_i), u), boolean=True)
		f_halted = np.zeros((1, u)).astype(bool)

		# F is vertical stack of 0s for all of the G1s, the
		# Post G1 boolean vector we are searching for, and a 0 for halted
		# F will be converted to 1+ values in the objective
		# and the final solution.
		f = cp.vstack([f_rg1, f_cg1, f_dg1, f_pg1, f_halted])+1

		elementwise_result = self.H@f - self.g

		objective = cp.Minimize(
			cp.sum(cp.norm(elementwise_result, 'fro')**2)
		)

		constraints = []
		                        
		# Post G1
		for i in range(1, len(f_pg1_i)):
		    index = f_pg1_i[i]
		    prev_index = f_pg1_i[i-1]
		    constraints.append(f[index] >= f[prev_index])

		problem = cp.Problem(objective, constraints)
		problem.solve(verbose=False)

		print("CVXPY problem finished with status: ", problem.status)
		# print("Transition point", (f.value > 1.5).argmax())

		self.f = f.value

	def plot_result(self):
		plt.figure(figsize=(13, 2))

		plt.subplot(1, 3, 1)
		plt.plot(self.f)

		plt.subplot(1, 3, 2)
		plt.plot(self.H@self.f)

		plt.subplot(1, 3, 3)
		plt.plot(self.g)
