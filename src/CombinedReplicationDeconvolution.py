

 
import numpy as np
import cvxpy as cp
import pandas as pd

from src.timer import Timer
from src.utils import print_fl
from matplotlib import pyplot as plt
from src.replication_deconvolution_solver import deconvolve_replication_brute_force
from src.RealDataReplication import RealDataReplicationDeconvolution


class CombinedReplicationDeconvolution():
	"""This model deconvolve the replication timing using the data from both replicates.

	Single F, Separate Ns, Bs, Hs, and Gs
	"""
	def __init__(self, config1, config2, chr):

		self.chr = chr

		# Load the MNase data for each replicate for the relevant chromosome
		self.real_deconv1 = RealDataReplicationDeconvolution(config1, chr=chr, replicate=1)
		self.real_deconv2 = RealDataReplicationDeconvolution(config2, chr=chr, replicate=2)

		# Disable H optimization for the combined model
		self.disable_H_optimization = True

		# Handle thresholding, each will threshold different regions, 
		# so union the regions that are thresholded...
		# todo: is union the most appropriate?
		# Override each threshold by the union
		union_threshold = self.real_deconv1.selected_threshold_region & self.real_deconv2.selected_threshold_region
		self.real_deconv1.selected_threshold_region = union_threshold
		self.real_deconv2.selected_threshold_region = union_threshold


	def setup_deconvolution(self):
		# Setup deconvolution for each to initialize H, N, B, and G

		self.real_deconv1.setup_deconvolution()
		self.real_deconv2.setup_deconvolution()
		self.combine_replicate_data_structures()


	def combine_replicate_data_structures(self):

		self.H = np.concatenate([self.real_deconv1.H, self.real_deconv2.H], axis=0)
		self.G = np.concatenate([self.real_deconv1.G, self.real_deconv2.G], axis=0)

		# Construct N as average DNA from replicate 1 and 2 concatenated
		self.average_DNA = np.concatenate([self.real_deconv1.average_DNA, self.real_deconv2.average_DNA])
		self.N = np.linalg.inv(np.diag(self.average_DNA))

		# Construct B as the average of replicate 1 and 2
		self.B = (self.real_deconv1.initial_B + self.real_deconv2.initial_B)/2.


	def deconvolve(self):
		"""Deconvolve the replication by combining the Ns, Gs, Hs, and Bs"""

		# Config of the first replicate (using the indices for deconvolution, so either config
		# will work)
		config = self.real_deconv1.config

		self.F, self.rn = deconvolve_replication_brute_force(config, 
			self.H, self.G, self.N, self.B)


	def iterative_deconvolution_updates(self, total_iterations, verbose=True, timer=None):

		from src.RealDataReplication import iterative_deconvolution_updates

		# Config of the first replicate (using the indices for deconvolution, so either config
		# will work)
		config = self.real_deconv1.config

		result = iterative_deconvolution_updates(
		    config=config, H=self.H, G=self.G, initial_N=self.N, 
		    initial_B=self.B, total_iterations=total_iterations, 
		    timer=timer, verbose=verbose)

		# Store the final result items
		self.F = result.Fs[-1]
		self.B = result.Bs[-1]
		self.N = result.Ns[-1]
		self.rn = result.iterative_update_rns[-1]

		return result


	def run_epochs(self, num_epochs, function_update=None):

		from src.optimize_H import ParameterOptimizer

		timer = Timer()

		num_iterations_N_B = 20
		update_params_df = pd.DataFrame()

		N = self.N
		B = self.B

		from src.optimize_H import create_bounds_params_from_config
		params1_df = create_bounds_params_from_config(self.real_deconv1.config)
		params2_df = create_bounds_params_from_config(self.real_deconv2.config)

		Hs = np.zeros((num_epochs, *self.H.shape))
		Ns = np.zeros((num_epochs, *N.shape))
		Bs = np.zeros((num_epochs, *B.shape))
		Fs = np.zeros((num_epochs, *self.F.shape))

		def split_combined_N(N):
			"""Split N1 and N2 using the number of timepoints in replicate 1.
			The optimizers run on each of the replicates separately and require
			N to be split, while the deconvolution algorithm uses and updates
			the combined N term."""
			len_N1 = len(self.real_deconv1.config.timepoints)
			N1 = np.diag(np.diag(N)[:len_N1])
			N2 = np.diag(np.diag(N)[len_N1:])
			return N1, N2

		# Initialize with the last value of N, split for each replicate
		N = self.N
		N1, N2 = split_combined_N(self.N)
		B = self.B

		optimizer1 = ParameterOptimizer(
			init_params_df=params1_df,
			config=self.real_deconv1.config,
			N=N1,
			F=self.F,
			B=self.B,
			G=self.real_deconv1.G
		)

		optimizer2 = ParameterOptimizer(
			init_params_df=params2_df,
			config=self.real_deconv2.config,
			N=N2,
			F=self.F,
			B=self.B,
			G=self.real_deconv2.G
		)

		for epoch in range(num_epochs):

			print_fl(f"Epoch: {epoch}")
			

			if not self.disable_H_optimization:
				# Run the optimizer on each of the replicate Hs
				print_fl(f"[{epoch}]: Running H optimizer for replicate 1")
				optimizer1.optimize(maxiter=1000, verbose=True)
				print_fl(f"[{epoch}]: Running H optimizer for replicate 2")
				optimizer2.optimize(maxiter=1000, verbose=True)
				print_fl(f"[{epoch}]: Done.")

			# Combine the Hs
			self.H = np.concatenate([optimizer1.current_H, optimizer2.current_H], axis=0)
				
			# Updates as self.F, self.N, and self.B
			print_fl(f"[{epoch}]: Running iterative updates and combined deconvolution for F, N, and B")
			self.iterative_deconvolution_updates(
				total_iterations=num_iterations_N_B, timer=timer, verbose=False)
			timer.print_time()
			
			# Create parameter row
			params_row1 = pd.DataFrame([optimizer1.params_df['value']], index=[epoch])
			params_row2 = pd.DataFrame([optimizer2.params_df['value']], index=[epoch])
			params_row = params_row1.join(params_row2, lsuffix='_1', rsuffix='_2')

			params_row['opt_H_rn_1'] = optimizer1.rn
			params_row['opt_H_rn_2'] = optimizer2.rn
			params_row['F_rn'] = self.rn

			update_params_df = pd.concat([update_params_df, params_row])

			# Split the newly learned N for each optimizer
			N1, N2 = split_combined_N(self.N)
			optimizer1.N = N1
			optimizer2.N = N2

			# Updates to B and F
			optimizer1.B = self.B
			optimizer1.F = self.F
			optimizer2.B = self.B
			optimizer2.F = self.F

			Hs[epoch] = self.H
			Fs[epoch] = self.F
			Ns[epoch] = self.N
			Bs[epoch] = self.B

			print_fl(update_params_df.iloc[-1])

			# For progressive saving or other tracking logic
			if function_update is not None:
				function_update(epoch, update_params_df, Hs, Fs, Ns, Bs)

		return update_params_df, Hs, Fs, Ns, Bs

	def plot_heatmaps(self):
		from src.RealDataReplication import plot_heatmaps

		masked_columns = self.real_deconv1.masked_G_df.columns
		full_columns = self.real_deconv1.G_df.columns

		fig = plot_heatmaps(self.N, self.F, self.H, self.B, self.G, column_names=masked_columns,
			full_column_names=full_columns)
		return fig
