
import numpy as np
import cvxpy as cp
from src.timer import Timer
import pandas as pd
from matplotlib import pyplot as plt
from src.replication_deconvolution_solver import deconvolve_replication, deconvolve_replication_brute_force
from src.utils import print_fl

early_color = plt.get_cmap('Oranges')(0.75)
late_color = plt.get_cmap('Purples')(0.75)


class RealDataReplicationDeconvolution():
	"""This model deconvolve the replication timing.
	"""
	def __init__(self, config=None, chr=10, replicate=None, configs=None,
		deconvolve_combined=False):

		self.chrom = chr

		if deconvolve_combined:

			print_fl("Setting up combined deconvolution model")

			self.load_replicate_data(chr, 1)
			self.setup_deconvolution(configs[0])
			self.H1 = self.H
			self.average_DNA_1 = self.average_DNA
			self.N1 = self.N
			self.G1 = self.G
			self.B1 = self.B
			self.config1 = configs[0]
			self.thresh1 = self.selected_threshold_region

			self.load_replicate_data(chr, 2)
			self.setup_deconvolution(configs[1])
			self.average_DNA_2 = self.average_DNA
			self.H2 = self.H
			self.B2 = self.B
			self.G2 = self.G
			self.config2 = configs[0]
			self.thresh2 = self.selected_threshold_region

			self.H = np.concatenate([self.H1, self.H2], axis=0)
			self.G = np.concatenate([self.G1, self.G2], 
				axis=0)

			# Construct N as average DNA from replicate 1 and 2 concatenated
			self.average_DNA = np.concatenate([self.average_DNA_1, self.average_DNA_2])
			self.N = np.linalg.inv(np.diag(self.average_DNA))

			# Construct B as the average of replicate 1 and 2
			self.B = (self.B1 + self.B2)/2.

			self.deconvolve_combined = True
			self.selected_threshold_region = self.thresh1 | self.thresh2
			self.replicate = 'combined'

		else:
			self.replicate = replicate
			self.config = config
			self.load_replicate_data(chr, replicate)
			self.setup_deconvolution(config)


	def load_replicate_data(self, chr, replicate):

		from src.mnase_10kb_loader import MNase10kbLoader

		mnase_loader = MNase10kbLoader()
		mnase_loader.load_mnase_data(replicate=replicate, chromosome=chr)
		mnase_loader.compute_sliding_window_counts_all_times()

		self.mnase_loader = mnase_loader
		self.unnormalized_total_occupancy = mnase_loader.all_counts_unnormalized_df
		self.normalized_occupancy = mnase_loader.normalized_total_occupancy_df
	
		# Setup regions to threshold, 
		# regions with low occupancy will be omitted when needed
		self.selected_threshold_region = self.normalized_occupancy.T.mean(axis=0) > 0.75


	def setup_deconvolution(self, config, initial_N=None, initial_B=None):
		"""Setup the deconvolution:

		1. The config and H
		"""

		self.config = config
		self.config.calculate_H()
		self.H = self.config.H

		print_fl("Initializing N using config H.")
		self.G = self.normalized_occupancy.T.values

		if initial_N is None:
			self.average_DNA, self.initial_N = compute_N(config)
		else:
			self.initial_N = initial_N

		print_fl("Initializing B using timepoint 0")
		if initial_B is None:
			self.initial_B = np.diag(self.G[0])
		else:
			self.initial_B = initial_B


	def deconvolve(self):
		self.setup_deconvolution(self.config)
		self.F, self.rn = deconvolve_replication_brute_force(self.config, 
			self.H, self.G, self.N, self.B)


	def iterative_deconvolution_updates(self, total_iterations, timer=None,
		initial_N=None, initial_B=None):
		"""Iteratively deconvolve for the replication curve F.

		Then update N and B. Keep track of the residual norm to identify
		when the solution converges.
		"""

		if timer is None:
			timer = Timer()

		if initial_N is None:
			initial_N = self.initial_N

		if initial_B is None:
			initial_B = self.initial_B

		# The Ns and Bs to start each iteration. 
		# the first entry will be the initial conditions
		self.Ns = np.zeros((total_iterations, *initial_N.shape))
		self.Bs = np.zeros((total_iterations, *initial_B.shape))
		self.Ns[0] = initial_N
		self.Bs[0] = initial_B

		# The residual following the end of the each iteration
		self.iterative_update_rns = np.zeros(total_iterations)

		# The iteratively updated Fs following each run
		n, m = self.H.shape
		num_sites = initial_B.shape[0]
		self.Fs = np.zeros((total_iterations, m, num_sites))

		for iteration in range(total_iterations):

			print("Iteration", iteration)

			N = self.Ns[iteration]
			B = self.Bs[iteration]

			result = deconvolve_replication_brute_force(self.config, 
				self.H, self.G, N, B, timer=timer)

			F, rn = result

			# Store the solutions in the F and rn datum
			self.Fs[iteration] = F
			self.iterative_update_rns[iteration] = rn

			# Update N and B
			updated_N, updated_B = self.update_N_B(N, self.H, self.G, B, F)

			# Update N and B for the next run
			# skip the last entry
			if iteration < total_iterations-1:
				self.Ns[iteration+1] = updated_N
				self.Bs[iteration+1] = updated_B

			print(f"Iteration completed {timer.get_time()}, rn={rn}")

			self.F = F
			self.rn = rn
			self.N = N
			self.B = B

	def compute_rn(self, H, F, N, B, G):

		NHFB = N @ H @ F @ B
		loss = np.mean((NHFB - G)**2)

		return loss


	def update_N_B(self, N, H, G, B, F):
		"""Using the solution from the last run, update N and B"""

		# Where is G? in this formulation
		GBinv_HF_div = np.divide((G@np.linalg.inv(B)), (H@F))
		updated_N = np.diag(GBinv_HF_div.mean(axis=1))

		num_rows = G.shape[0]
		G_sums = G.T @ np.ones((num_rows, 1))
		NHF_sums = (updated_N@H@F).T @ np.ones((num_rows, 1))
		updated_b_diag = (G_sums / NHF_sums).flatten()
		updated_B = np.diag(updated_b_diag)
		# updated_B = self.initial_B

		return updated_N, updated_B


	def plot_heatmaps(self):
		N, F, G, B, H = self.N, self.F, self.G, self.B, self.H

		plot_heatmaps(N, F, H, B, G)
		plt.suptitle(f"Replication {self.replicate}"
			f" deconvolution,\nChromosome {self.chrom}")


	def save_replication_deconvolution(self):
		from src.utils import mkdirs_safe

		copy_correction_save_dir = 'data/copy_correction/'
		mkdirs_safe([copy_correction_save_dir])

		N_save_path = f'{copy_correction_save_dir}/N_combined.npy'
		B_save_path = f'{copy_correction_save_dir}/B_chr{self.chrom}_combined.csv'
		F_save_path = f'{copy_correction_save_dir}/F_chr{self.chrom}_combined.csv'

		np.save(N_save_path, self.N)
		np.save(B_save_path, self.B)

		B_df = pd.DataFrame(self.B,
					index=self.normalized_occupancy.index,
					columns=self.normalized_occupancy.index)
		B_df.index.name = 'start'
		B_df.to_csv(B_save_path)

		F_df = pd.DataFrame(self.F, 
			columns=self.normalized_occupancy.index)
		F_df.to_csv(F_save_path, index=False)

		print(f"Saved to: {N_save_path}")
		print(f"Saved to: {B_save_path}")
		print(f"Saved to: {F_save_path}")


	def plot_B(self):

		from src.sgd import get_chromosome_length

		chrom = self.chrom
		B = self.B
		chrom_len = get_chromosome_length(chrom)

		ys = np.diag(B)
		xs = np.linspace(0, chrom_len, len(ys))

		plt.figure(figsize=(9, 3))
		plt.plot(xs, ys, c=plt.get_cmap('Spectral')(0.3), lw=3)
		plt.title(f"Average 10 kb occupancy, chr{chrom}", pad=11)
		plt.xlim(0, chrom_len)
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Average occupancy")
		plt.subplots_adjust(bottom=0.3)


	def plot_G(self):

		tps = self.config.timepoints
		start_indices = self.unnormalized_total_occupancy.index.values
		extent = [0, start_indices[-1], 0, tps[-1]]

		plt.figure(figsize=(13, 3))
		plt.imshow(self.G, cmap='RdBu_r', vmin=0, vmax=2, 
			interpolation='none', aspect='auto',
			extent=extent)
		plt.colorbar()
		plt.title(f"Experiment 10 kb MNase-seq reads, replicate 1, chr{self.chrom}")
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Experimental time")

		plt.subplots_adjust(bottom=0.2)


	def plot_predicted_G(self):

		predicted_G = self.N@self.H@self.F@self.B

		tps = self.config.timepoints
		start_indices = self.unnormalized_total_occupancy.index.values
		extent = [0, start_indices[-1], 0, tps[-1]]

		plt.figure(figsize=(13, 3))
		plt.imshow(predicted_G, cmap='RdBu_r', vmin=0, vmax=2, 
			interpolation='none', aspect='auto',
			extent=extent)
		plt.colorbar()
		plt.title(f"Predicted $G$, replicate 1, chr{self.chrom}")
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Experimental time")

		plt.subplots_adjust(bottom=0.2)

	def plot_residual(self):

		predicted_G = self.N@self.H@self.F@self.B
		residual = predicted_G - self.G

		tps = self.config.timepoints
		start_indices = self.unnormalized_total_occupancy.index.values
		extent = [0, start_indices[-1], 0, tps[-1]]

		plt.figure(figsize=(13, 3))
		plt.imshow(residual, cmap='RdBu_r', vmin=-1, vmax=1, 
			interpolation='none', aspect='auto',
			extent=extent)
		plt.colorbar()
		plt.title(f"Residual, replicate 1, chr{self.chrom}")
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Experimental time")

		plt.subplots_adjust(bottom=0.2)

	def plot_F(self):
		from src.sgd import get_chromosome_length
		from src.chromatin_model import draw_phase_label_annotations

		chrom = self.chrom
		chrom_len = get_chromosome_length(chrom)

		F = self.top_F_df

		fig = plt.figure(figsize=(13, 3))

		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')
		t_tps = config.get_timepoints_for_branch('t')

		extent = [0, F.index[-1],
			F.columns[0], F.columns[-1]]

		plt.imshow(F.T, cmap='RdBu_r', vmin=0, vmax=2, 
			interpolation='none', aspect='auto',
				  extent=extent, origin='lower')

		cbar = plt.colorbar()
		cbar.ax.set_ylim(1, 2)
		cbar.ax.set_yticks([1, 2])
		cbar.ax.set_ylabel("Copy number")

		plt.title(f"Chr{chrom} replication profile, $F_r$")

		ax = plt.gca()

		draw_phase_label_annotations(ax, config, 
			flip=False, annotations_x=-11000)

		plt.xlim(extent[0]-25000, extent[1])
		plt.ylim(extent[3]-1, extent[2])
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Average single\ncell time, min")
		plt.subplots_adjust(bottom=0.2)

	def compute_replication_profile(self):

		from src.chromatin_model import draw_phase_label_annotations

		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')
		t_tps = config.get_timepoints_for_branch('t')

		t_index_tp_mapping = pd.DataFrame({'tp': config.get_timepoints_for_branch('t')},
			index=config.get_Hpositions_for_branch('t'))

		timing_dict = t_index_tp_mapping.iloc[:, 0].to_dict()

		start_indices = self.unnormalized_total_occupancy.index.values

		F = self.F
		replication_profile = pd.DataFrame(np.argmax(F, axis=0), index=start_indices, 
			columns=['replication_index'])
		replication_profile['replication_timing'] = replication_profile['replication_index'].map(timing_dict)

		self.replication_profile = replication_profile

		# Create a dataframe of the top branch replication profile
		self.top_F_df = pd.DataFrame(F.T[:, t_indices], 
			index=start_indices, columns=t_tps)

	def plot_example_f_curves(self):

		from src.chromatin_model import draw_phase_label_annotations

		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')
		t_tps = config.get_timepoints_for_branch('t')

		start_indices = self.unnormalized_total_occupancy.index.values
		sorted_replication_profile = self.replication_profile.sort_values('replication_index')

		early_row = sorted_replication_profile.iloc[10]
		late_row = sorted_replication_profile.loc[570000]
		self.early_row = early_row
		self.late_row = late_row

		F = self.F

		# Create a dataframe of the top branch replication profile
		top_F_df = pd.DataFrame(F.T[:, t_indices], 
			index=start_indices, columns=t_tps)

		plt.figure(figsize=(8, 4))

		plt.plot(top_F_df.loc[early_row.name]+0.005, 
				 label=f"Early, {early_row.name}",
				c=early_color, lw=4)
		plt.plot(top_F_df.loc[late_row.name]-0.005, 
				 label=f"Late, {late_row.name}",
				 c=late_color, lw=4)
		plt.legend(loc='upper left')
		plt.ylabel("Copy number")
		plt.xlabel("Average single cell time, min")
		draw_phase_label_annotations(plt.gca(), config, flip=True, annotations_x=0.9)
		plt.xlim(t_tps[0], t_tps[-1])
		plt.yticks([1, 2])
		plt.title("Example replication curves", pad=11)


def threshold_selection(dat, threshold_selection, fill=1.,
	renormalize=False):
	new_dat = dat.copy()
	new_dat[:, ~threshold_selection] = fill

	# If thresholded to fill with nans, we can renormalize such
	# that the non-thresholded out regions mean to 1
	if renormalize:

		# Get the current mean of the good rows
		row_means = np.nanmean(new_dat, axis=1)

		# Divide the data by these mean
		new_dat = new_dat / row_means.reshape((-1, 1))

	return new_dat


def plot_comparison_occupancies_G():
	# Def plot of chromosome 4 values
	# # Comparison of chrIV t=0 and t=30
	# plt.figure(figsize=(24, 6))
	# plt.subplot(3, 1, 1)
	# plt.plot(real_deconv_chr4.normalized_occupancy.T.loc[0])
	# plt.xticks([])
	# plt.axhline(1, c='black', lw=1)
	# plt.title("T=0")

	# plt.subplot(3, 1, 2)
	# plt.plot(real_deconv_chr4.normalized_occupancy.T.loc[30])
	# plt.axhline(1, c='black', lw=1)
	# plt.xticks([])
	# plt.title("T=30")

	# plt.subplot(3, 1, 3)
	# plt.plot(np.log2(real_deconv_chr4.normalized_occupancy.T.loc[30] / 
	#                  real_deconv_chr4.normalized_occupancy.T.loc[0])
	#         )
	# plt.axhline(0, c='black', lw=1)
	# plt.ylim(-1, 1)
	# plt.title("log2 ratio difference")
	# plt.suptitle("ChrIV t=0 vs t=30 occupancy")
	pass

def plot_histogram_occupancies_G(config, G):
	fig, axs = plt.subplots(3, 6, figsize=(13, 6))

	tps = config.timepoints
	axs = np.array(axs).T.flatten()
	plot_G = G.T

	for ax in axs:
		ax.set_xticks([])
		ax.set_yticks([])
	for i in range(plot_G.shape[1]):
		ax = axs[i]
		
		ax.hist(plot_G[:, i], bins=20, facecolor=plt.get_cmap('Spectral')(i/plot_G.shape[1]),
			   edgecolor='gray', lw=1)
		ax.set_xlim(0, 2)
		ax.set_ylim([0, 120])
		ax.axvline(1, c='black', lw=1, ls='dotted')
		ax.set_title(f"{tps[i]} min")
		if i % 3 == 2:
			ax.set_xticks([0, 1, 2])
		
	plt.suptitle("Distribution of normalized G data per timepoint")
	plt.subplots_adjust(hspace=0.5)

	# This shoes the data for G is normal-ish and centered around 1.

def plot_heatmaps(N, F, H, B, G):

	inv_B = np.linalg.inv(B)
	transformed_G = (np.linalg.inv(N)@G@inv_B)

	plt.figure(figsize=(13, 11))

	plt.subplot(6, 1, 1)
	plt.imshow(F, cmap='RdBu_r', vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.xticks([])
	plt.colorbar()
	plt.title("$F$")

	plt.subplot(6, 1, 2)
	plt.imshow(H@F, cmap='RdBu_r', vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.xticks([])
	plt.colorbar()
	plt.title("$HF$")

	plt.subplot(6, 1, 3)
	plt.imshow(transformed_G, cmap='RdBu_r', vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.xticks([])
	plt.colorbar()
	plt.title("$(N^{-1})G(B^{-1})$")

	plt.subplot(6, 1, 4)
	predicted_G = N@H@F@B
	plt.imshow(predicted_G, cmap='RdBu_r', vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.xticks([])
	plt.colorbar()
	plt.title("Predicted G: $NHFB$")

	plt.subplot(6, 1, 5)
	plt.imshow(G, cmap='RdBu_r', vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.colorbar()
	plt.xticks([])
	plt.title("$G$")

	plt.subplot(6, 1, 6)
	plt.imshow((N@H@F@B) - (G), vmin=-1, vmax=1, cmap='RdBu_r', interpolation='none', aspect='auto')
	plt.colorbar()
	plt.title("$NHFB - G$")
	plt.subplots_adjust(hspace=0.5, top=0.9)


def compute_N(config, plot=False):
	from src.model import color_for_key

	config.calculate_H()
	H = config.H
	tps = config.timepoints

	cg1_mass = H[:, config.get_Hpositions_for_phase('RG1')].sum(axis=1)
	rg1_mass = H[:, config.get_Hpositions_for_phase('CG1')].sum(axis=1)
	s_mass = H[:, config.get_Hpositions_for_phase('S')].sum(axis=1)
	g2m_mass = H[:, config.get_Hpositions_for_phase('G2M')].sum(axis=1)
	h_mass = H[:, config.get_Hpositions_for_phase('H')].sum(axis=1)

	# Assume linear transition of S-phase
	s_indices = config.get_Hpositions_for_phase('S')
	s_masses = np.linspace(1, 2, len(s_indices))

	g1_mass = cg1_mass+rg1_mass
	s_dna_content = H[:, config.get_Hpositions_for_phase('S')] @ np.diag(s_masses).sum(axis=1)
	replicating_mass = s_dna_content+g2m_mass*2

	average_DNA = h_mass+g1_mass+replicating_mass
	N = np.linalg.inv(np.diag(average_DNA))

	if plot:
		plt.figure(figsize=(6, 4))

		plt.fill_between(tps, h_mass, 0, label="H mass", color=color_for_key('H'))

		plt.fill_between(tps, g1_mass+h_mass, h_mass, label="G1 mass", color=color_for_key('CG1'))

		# ---------

		inv_n = g1_mass+s_dna_content+g2m_mass*2+h_mass

		plt.fill_between(tps, g1_mass+s_dna_content+h_mass, 
							  g1_mass+h_mass, label="S mass", color=color_for_key('S'))

		plt.fill_between(tps, inv_n, 
							 g1_mass+s_dna_content+h_mass, 
							 label="G2M mass", color=color_for_key('G2M'))

		plt.plot(tps, inv_n, c='red', lw=4, 
			label="Avg. DNA content")
		plt.ylim(0, 2.2)

		plt.legend(ncol=3, loc='upper right')

		plt.title("Estimation of average DNA content, CLOCCS")
		plt.xlabel("Clock time")
		plt.ylabel("Average DNA content")
		plt.xlim(0, tps[-1])

	return average_DNA, N


def read_n_fr_b(chrom, deconv_span):
	from src.mnase_10kb_loader import get_bin_for_position

	N = np.load('data/copy_correction/N_combined.npy')
	B_df = pd.read_csv(f'data/copy_correction/B_chr{chrom}_combined.csv').set_index('start')
	Fr_df = pd.read_csv(f'data/copy_correction/F_chr{chrom}_combined.csv')
	B_df.columns = B_df.columns.astype(int)
	Fr_df.columns = Fr_df.columns.astype(int)

	start_indices = B_df.index

	mid_span = (deconv_span[0]+deconv_span[1])/2
	bin_idx, start = get_bin_for_position(mid_span, start_indices)

	# Load the b and f_replication for the span to be deconvolved.
	b = B_df.loc[start].loc[start]
	fr = Fr_df[start]

	return N, fr, b


def read_no_copy_correction_n_fr_b(H):

	n, m = H.shape
	N = np.eye(n)
	b = 1
	fr = np.ones(m)

	return N, fr, b

