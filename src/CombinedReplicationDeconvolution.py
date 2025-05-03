

 
import numpy as np
import cvxpy as cp
import pandas as pd

from src.timer import Timer
from src.utils import print_fl
from matplotlib import pyplot as plt
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


	def setup_deconvolution(self, warm_start_output_directory=None, warm_start_chrom=None):
		# Setup deconvolution for each to initialize H, N, B, and G

		from src.RealDataReplication import load_N_F_B_for_replication_deconv_from_save

		# Load the F, N, and B from disk
		if warm_start_output_directory is not None:	
			# Load the F, N, and B from disk
			print_fl(f"Warm start load F N and B from disk {warm_start_output_directory}")
			F1, N1, B1 = load_N_F_B_for_replication_deconv_from_save(warm_start_output_directory, 1, warm_start_chrom)
			F2, N2, B2 = load_N_F_B_for_replication_deconv_from_save(warm_start_output_directory, 2, warm_start_chrom)
		else:
			N1 = None
			N2 = None

		# Load just N, more important than B. And we can deconvolve other chromosomes easily
		# First set of iterations will provide a consistent replication profile
		self.real_deconv1.setup_deconvolution(initial_B=None, initial_N=N1)
		self.real_deconv2.setup_deconvolution(initial_B=None, initial_N=N2)
		self.combine_replicate_data_structures()


	def combine_replicate_data_structures(self):

		self.H, self.G = concatenate_H_G(self.real_deconv1.config.H, self.real_deconv2.config.H,
										 self.real_deconv1.G, self.real_deconv2.G)

		# Construct N as average DNA from replicate 1 and 2 concatenated
		# self.average_DNA = np.concatenate([self.real_deconv1.average_DNA, self.real_deconv2.average_DNA])
		#self.N = np.linalg.inv(np.diag(self.average_DNA))
		n1 = np.diag(self.real_deconv1.initial_N)
		n2 = np.diag(self.real_deconv2.initial_N)

		combined_n = np.concatenate([n1, n2])
		self.N = np.diag(combined_n)

		# Construct B as the average of replicate 1 and 2
		b1 = np.diag(self.real_deconv1.initial_B)
		b2 = np.diag(self.real_deconv2.initial_B)
		combined_b = (b1+b2)/2.
		self.B = np.diag(combined_b)

		print("H, G, N, B shapes:", self.H.shape, self.G.shape, self.N.shape, self.B.shape)


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

		# Use the G df's full column set, may need to handle the thresholding and
		# masking of the individual replicates...

		self.F_df = pd.DataFrame(self.F, columns=self.real_deconv1.masked_start_indices,
			index=range(self.F.shape[0]))

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


def load_config_from_replication_runs(output_directory, chrom, mode='chromatin'):
	"""Load configs from the output directory of previously run replicates.

	This should be used only for the parameter learning/replication profile
	optimization steps. Once completed, any models for chromatin and gene
	expression deconvolution will belong in the models folder.
	"""

	from src.config import load_cloccs_configs
	from src.RealDataReplication import modify_config_from_run

	print("Loading configs from replication runs: ", output_directory)

	# This function will assume we are loading from CLOCCS
	# and updating to the latest parameter run results in the output directory

	config1, config2 = load_cloccs_configs(mode=mode)
	config1 = modify_config_from_run(config1, output_directory, 1, chrom)
	config2 = modify_config_from_run(config2, output_directory, 2, chrom)

	return config1, config2


def concatenate_H_G(H1, H2, G1, G2):
	H = np.concatenate([H1, H2], axis=0)

	if len(G1.shape) == 2:
		G = np.concatenate([G1, G2], axis=0)
	else:
		G = np.concatenate([G1, G2])

	return H, G

