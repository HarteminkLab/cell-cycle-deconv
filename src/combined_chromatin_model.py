
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.chromatin_model import ChromatinModel
from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver
from src.utils import print_fl
from src.global_config import GlobalConstants
from src.figure_configs import FiguresConfig


class CombinedChromatinModel:
	"""
	In this class we will construct the combined chromatin deconvolution model
	"""

	def __init__(self, config1, config2):

		# Rename the config such that when plotting with the title
		# the combined model name is used.
		# Config2 will not be used for plotting

		if config1.config_type == 'delta':
			config1.name = f"Combined Prepend Model, $\\alpha$={config1.alpha},{config2.alpha}"
			config2.name = f"Combined Prepend Model,, $\\alpha$={config1.alpha},{config2.alpha}"
		elif config1.config_type == 'distinct':
			config1.name = f"Combined Distinct, $\\alpha$={config1.alpha},{config2.alpha}"
			config2.name = f"Combined Distinct, $\\alpha$={config1.alpha},{config2.alpha}"
		elif config1.config_type == 'shared':
			config1.name = f"Combined Shared, $\\alpha$={config1.alpha},{config2.alpha}"
			config2.name = f"Combined Shared, $\\alpha$={config1.alpha},{config2.alpha}"

		self.chrom1_model = ChromatinModel(config1)
		self.chrom2_model = ChromatinModel(config2)
		self.G_deconvolution_offset = 1


	def load_mnase_span(self, chrom, mnase_span, verbose=True):
		self.chrom1_model.load_mnase_span(chrom, mnase_span, verbose=verbose)
		self.chrom2_model.load_mnase_span(chrom, mnase_span, verbose=verbose)


	def load_combined_mnase_gene(self, gene_name):
		"""This takes the place of load_mnase_gene, as we don't need the
		replicate parameter anymore"""
		self.chrom1_model.load_mnase_gene(gene_name)
		self.chrom2_model.load_mnase_gene(gene_name)

	def load_combined_mnase_orc(self, orc_or_ars):
		"""This takes the place of load_mnase_gene, as we don't need the
		replicate parameter anymore"""
		self.chrom1_model.load_mnase_orc(orc_or_ars)
		self.chrom2_model.load_mnase_orc(orc_or_ars)


	def load_copy_correction_data(self):
		from src.RealDataReplication import read_n_fr_b
		chrom = self.chrom1_model.chr
		mnase_span = self.chrom1_model.mnase_span

		# Loads the combined N and f replication
		# todo: refactor for single vs combined replicate model
		print(f"todo: Loading copy correction N, Fr, B, testing with prototype replication data")
		combined_dir = 'output/prototype_pipeline_subset/combined_replication'
		combined_N = np.load(f'{combined_dir}/N.npy')
		_, _, self.f_replication, self.b = read_n_fr_b(chrom, mnase_span, 1)

		self.N = combined_N


	def	setup_deconv_model(self, gamma=0.007, G=None, G1=None, G2=None, wavelet="Symmlet",
			padding_type='both'):

		chrom1_model = self.chrom1_model
		chrom2_model = self.chrom2_model

		# Load N, freplication and b replication data for correction
		self.load_copy_correction_data()
		N = self.N
		f_replication = self.f_replication
		b = self.b

		self.gamma = gamma

		# Next we will need to setup the deconvolution model to combine the H
		# and the deconvolution G data

		if G1 is None:
			self.G1 = chrom1_model.G
		else:
			self.G1 = G1

		if G2 is None:
			self.G2 = chrom2_model.G
		else:
			self.G2 = G2

		# Combine the two G datasets row-wise
		if G is None:
			self.G = np.concatenate([self.G1, 
				self.G2])
		else:
			self.G = G

		if self.G_deconvolution_offset > 0:
			print_fl(f"Adding a deconvolution offset to G: {self.G_deconvolution_offset}")
			self.G = self.G+self.G_deconvolution_offset

		# Create the first replicates model and H
		self.H1 = chrom1_model.config.calculate_H()

		# And the second
		self.H2 = chrom2_model.config.calculate_H()

		# Combine the H matrices
		self.H = np.concatenate([self.H1, self.H2])

		self.solver = ChromatinDeconvolveSolver(self.chrom1_model.config, self.G, self.N,
			wavelet=wavelet, padding_type=padding_type, f_replication=f_replication,
			b=b)
		self.solver.H = self.H

		# For plotting results
		self.chrom1_model.solver = self.solver
		self.chrom2_model.solver = self.solver

		self.found_optimal_success = None
		self.deconvolved_f_value = None


	def deconvolve(self, gamma, kappa, verbose=True):
		self.F = self.solver.deconvolve_G_iteratively(gamma=gamma, kappa=kappa, verbose=verbose)


	def find_origin_p1_and_m1_nucleosome_position(self):
		self.chrom1_model.find_origin_p1_and_m1_nucleosome_position()
		self.chrom2_model.find_origin_p1_and_m1_nucleosome_position()

	def deconvolve_find_optimal_gamma(self):
		"""
		Find the optimal gamma value
		"""
		from src.timer import Timer

		from src.find_gamma_chromatin import FindOptimalGammaChromatin

		timer = Timer()

		self.setup_deconv_model()
		self.find_gamma_chromatin = FindOptimalGammaChromatin(self.solver)
		self.found_optimal_success = self.find_gamma_chromatin.find_optimal(silence=False)
		self.gamma = self.find_gamma_chromatin.gamma
		self.rn = self.find_gamma_chromatin.rn
		self.sn = self.find_gamma_chromatin.sn
		self.deconvolved_f_value = self.find_gamma_chromatin.f

		self.set_results(self.deconvolved_f_value, 
						  self.rn, self.sn,
						  self.gamma)

		print_fl(f"Found optimal gamma in: {timer.get_time()}")
		print_fl(f"Find optimal success: {self.found_optimal_success}")
		print_fl(f"The fitting norm is {self.rn:.2f}, "
			  f"the smoothing norm is: {self.sn:.2f}")

	def set_results(self, f, rn, sn, gamma):
		"""Following completion of deconvolution or find gamma deconvolution, we
		will need to set the results to the appropriate fields.

		TODO: This may need to be cleaned up but for now each of the chrom models
		have plotting code individually, so we set the results in each of them
		"""

		self.rn = rn
		self.sn = sn
		self.gamma = gamma

		self.chrom1_model.deconvolved_f_value = f
		self.chrom1_model.rn = rn
		self.chrom1_model.sn = sn
		self.chrom1_model.gamma = gamma

		self.chrom2_model.deconvolved_f_value = f
		self.chrom2_model.rn = rn
		self.chrom2_model.sn = sn
		self.chrom2_model.gamma = gamma

		# Set the f, rn, and sn from the results to the model
		self.pred_G = np.matmul(self.H, f) 
		tp1 = self.chrom1_model.timepoints
		self.pred_G1 = self.pred_G[0:len(tp1)]
		self.pred_G2 = self.pred_G[len(tp1):]

		self.chrom1_model.compute_ptr()
		self.chrom2_model.compute_ptr()


	def plot_branches(self, F=None, figsize=(20, 7)):
		if F is None: F = self.solver.F

		F = F/F.mean()

		from src.chromatin_deconvolution_solver import plot_branches
		plot_branches(self.chrom1_model.config, self.chrom1_model.chr,
			self.chrom1_model.mnase_span, F, figsize=figsize, vmax=30)

	# def plot_raw_prediction(self, replicate, vmax=20):
	# 	"""Plot the resulting comparison between the raw and predicted data"""

	# 	if replicate == 1:
	# 		title = self.chrom1_model.define_title().replace("Combined", "Combined-Rep.1")
	# 		fig = self.chrom1_model.plot_prediction_comparison(self.pred_G1, title, vmax)
	# 	else:
	# 		title = self.chrom1_model.define_title().replace("Combined", "Combined-Rep.2")
	# 		fig = self.chrom2_model.plot_prediction_comparison(self.pred_G2, title, vmax)

	# 	return fig


	def plot_normalization_sanity_check(self):

		fig, axs = plt.subplots(2, 3, figsize=(13, 6))
		plt.subplots_adjust(top=0.8)
		plt.suptitle("Normalization verification")

		axs = np.array(axs)

		first_row = axs[0]
		second_row = axs[1]

		self.chrom1_model.plot_normalization_sanity_check(axs=first_row)
		self.chrom2_model.plot_normalization_sanity_check(axs=second_row)

		for ax in second_row:
			ax.set_title("")

		return fig



	def plot_raw_origin_data(self, replicate, vmax=100):
		if replicate == 1:
			return self.chrom1_model.plot_raw_orc_data(vmax=vmax)
		else:
			return self.chrom2_model.plot_raw_orc_data(vmax=vmax)


	def plot_nfr_origin_occ(self):
		# todo: get the timepoitns squared away so we can report the timing of these events...

		# average the t_timepoints
		t_tps_1 = np.array(self.chrom1_model.config.get_timepoints_for_branch('t'))
		t_tps_2 = np.array(self.chrom2_model.config.get_timepoints_for_branch('t'))
		t_tps_mean = (t_tps_1+t_tps_2)/2

		self.chrom1_model.plot_nfr_origin_occ_comparision(t_tps=t_tps_mean)

		# Replication time is average of the replicate 1 and replicate 2 times
		# offset by g1's length
		_, g1_1, _ = self.chrom1_model.config.get_g1_lens()
		_, g1_2, _ = self.chrom2_model.config.get_g1_lens()
		g1 = (g1_1 + g1_2)/2
		replication_time = self.chrom1_model.origin.replication_time-g1

		plt.axvline(replication_time, c='black', lw=1, ls='dotted', zorder=0)
		plt.suptitle(f"{self.chrom1_model.origin.ars_name}")
		print("todo: resolve the timepoints plotted to be the average of rep1 and rep2")


	def plot_sm_nuc_hm(self):

		# Summarize the small fragment changes into a matrix, collapse reads of small factor length
		# down to a single vector in which we can create a time course matrix on.
		center = self.chrom1_model.computed_plus_one
		xlims = center-500, center+500

		from src.chromatin_metrics import fragment_lengths_definitions
		from src.global_config import GlobalConstants
		config1 = self.chrom1_model.config
		t_indices = config1.get_Hpositions_for_branch('t')

		# Starting with lengths less than 100
		f_imgs = self.chrom1_model.get_f_images()

		select_nuc_frag_bins = ((GlobalConstants.Y_LEN_DEFINITIONS > 120) & \
								(GlobalConstants.Y_LEN_DEFINITIONS < 170))[:-1]
		select_small_frag_bins = (GlobalConstants.Y_LEN_DEFINITIONS <= 100)[:-1]

		def select_bins_sum(f_imgs, selected_bins_indices):
			selected_bins = f_imgs[:, selected_bins_indices, :]
			selected_bins_sum = selected_bins.mean(axis=1)[t_indices]
			return selected_bins_sum

		def plot_frag_bins_hm(selected_bins_sum, vmin=0, vmax=10, cmap='magma_r'):
			plt.imshow(selected_bins_sum, vmax=vmax, vmin=vmin, 
				extent=[self.chrom1_model.bin_extents[0],
						self.chrom1_model.bin_extents[1],
					   0, selected_bins_sum.shape[0]], aspect='auto', interpolation='none',
					   origin='lower', cmap=cmap)
			plt.xlim(*xlims)
			plt.ylim(selected_bins_sum.shape[0], 0)
			
		fig = plt.figure(figsize=(16, 4))
		small_frags_sum = select_bins_sum(f_imgs, select_small_frag_bins)

		#plt.subplot(2, 1, 5)
		nuc_frags_sum = select_bins_sum(f_imgs, select_nuc_frag_bins)

		gene = self.chrom1_model.gene
		from src.sgd import get_gene_title_name

		xticks, xtick_labels = self.chrom1_model.generate_xticks()
		
		plt.subplot(2, 3, 4)
		plot_frag_bins_hm(-small_frags_sum, vmin=-5, vmax=5, cmap='RdBu_r')
		plt.title("Small Fragments")
		plt.xticks(xticks, xtick_labels)
		plt.xlim(*xlims)

		plt.subplot(2, 3, 5)
		plot_frag_bins_hm(nuc_frags_sum, vmin=-5, vmax=5, cmap='RdBu_r')
		plt.title("Nucleosome fragments")
		plt.xticks(xticks, xtick_labels)
		plt.xlim(*xlims)

		plt.subplot(2, 3, 6)
		plot_frag_bins_hm(nuc_frags_sum-small_frags_sum, vmin=-5, vmax=5, cmap='RdBu_r')
		plt.title("Nucleosome+Small Fragments")
		plt.suptitle(get_gene_title_name(gene.name), fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.8, hspace=0.5)

		plt.xticks(xticks, xtick_labels)
		plt.xlim(*xlims)

		plt.subplot(2, 3, 1)
		ax = plt.gca()
		self.chrom1_model.plot_orf_annotation(ax)
		ax.set_xticks(xticks)
		ax.set_xticklabels(xtick_labels)
		ax.set_xlim(xlims)

		plt.subplot(2, 3, 2)
		ax = plt.gca()
		self.chrom1_model.plot_orf_annotation(ax)
		ax.set_xticks(xticks)
		ax.set_xticklabels(xtick_labels)
		ax.set_xlim(xlims)

		plt.subplot(2, 3, 3)
		ax = plt.gca()
		self.chrom1_model.plot_orf_annotation(ax)
		ax.set_xticks(xticks)
		ax.set_xticklabels(xtick_labels)
		ax.set_xlim(xlims)

		return fig


	def save_origin_plots(self, plot_dir, origin_index):

		from src.figure_configs import save_figure_for_analysis
		origin = self.chrom1_model.origin

		origin_run_name = f"{origin_index}_{origin['ars_name']}_{origin.name}"

		fig = self.plot_raw_origin_data(1)
		save_path = f"{plot_dir}/{origin_run_name}_raw_rep1.png"
		save_figure_for_analysis(save_path)
		plt.close(fig)

		fig = self.plot_raw_origin_data(2)
		save_path = f"{plot_dir}/{origin_run_name}_raw_rep2.png"
		save_figure_for_analysis(save_path)
		plt.close(fig)

		fig = self.create_deconvolution_plots_abbreviated_flipped(vmax=75)
		save_path = f"{plot_dir}/{origin_run_name}_deconvolution.png"
		save_figure_for_analysis(save_path)
		plt.close(fig)

		fig = self.create_deconvolution_plots_abbreviated_flipped(zoom=800, vmax=75, num_rows=4)
		save_path = f"{plot_dir}/{origin_run_name}_deconvolution_zoomed.png"
		save_figure_for_analysis(save_path)
		plt.close(fig)

		# fig = self.chrom1_model.plot_nucleosome_shift()
		# save_path = f"{plot_dir}/{origin_run_name}_shift.png"
		# save_figure_for_analysis(save_path)
		# plt.close(fig)

		# fig = self.chrom1_model.plot_origin_trackers()
		# save_path = f"{plot_dir}/{origin_run_name}_tracking.png"
		# save_figure_for_analysis(save_path)
		# plt.close(fig)

		# fig = self.plot_nfr_origin_occ()
		# save_path = f"{plot_dir}/{origin_run_name}_nfr_origin_occ.png"
		# save_figure_for_analysis(save_path)
		# plt.close(fig)

	# def save_deconvolved_origin_outputs(self, out_dir, index):

	# 	origin = self.chrom1_model.origin
	# 	f = self.deconvolved_f_value
	# 	tracking_df = self.chrom1_model.get_origin_tracking_df()

	# 	origin_save_name = f"{origin.name}_{origin.ars_name}"

	# 	f_save_path = f'{out_dir}/{index}_f_{origin_save_name}.npy'
	# 	tracking_save_path = f'{out_dir}/{index}_tracking_{origin_save_name}.csv'
	# 	meta_save_path = f'{out_dir}/{index}_meta_{origin_save_name}.csv'

	# 	#---------- Save to disk -------------

	# 	# Save the f to disk
	# 	np.save(f_save_path, f)

	# 	# Save the ptr to disk
	# 	tracking_df.to_csv(tracking_save_path)

	# 	# Save meta information
	# 	from datetime import datetime
	# 	run_date = datetime.now().strftime("%D")

	# 	df = pd.DataFrame({
	# 		'rn': self.solver.rn, 'sn': self.solver.sn, 
	# 			'gm': self.solver.gamma,
	# 		'config1': self.chrom1_model.config.name,
	# 		'config2': self.chrom2_model.config.name,
	# 		'model1_path': self.chrom1_model.config.model_wt1_file,
	# 		'model2_path': self.chrom2_model.config.model_wt1_file,
	# 		'run_date': run_date,
	# 		'replicate': "combined",
	# 		'image_shape': str(GlobalConstants.IMAGE_SHAPE),
	# 		},
	# 		index=[self.chrom1_model.origin.name])
	# 	df.to_csv(meta_save_path, float_format="%.4f")

	# 	print_fl(f"Saved to {f_save_path}...")
	# 	print_fl(f"Saved to {tracking_save_path}...")
	# 	print_fl(f"Saved to {meta_save_path}...")

	# def save_deconvolved_outputs(self, out_dir, index):

	# 	orf_name = self.chrom1_model.gene.name
	# 	gene_name = self.chrom1_model.gene['gene']
	# 	f = self.deconvolved_f_value
	# 	G1 = self.G1
	# 	G2 = self.G2

	# 	f_save_path = f'{out_dir}/{index}_f_{orf_name}_{gene_name}.npy'
	# 	g1_save_path = f'{out_dir}/{index}_g1_{orf_name}_{gene_name}.npy'
	# 	g2_save_path = f'{out_dir}/{index}_g2_{orf_name}_{gene_name}.npy'

	# 	g1_correction_save_path = f'{out_dir}/{index}_g1_correction_{orf_name}_{gene_name}.csv'
	# 	g2_correction_save_path = f'{out_dir}/{index}_g2_correction_{orf_name}_{gene_name}.csv'

	# 	ptr_save_path = f'{out_dir}/{index}_ptr_{orf_name}_{gene_name}.npy'
	# 	meta_save_path = f'{out_dir}/{index}_meta_{orf_name}_{gene_name}.csv'

	# 	#---------- Save to disk -------------

	# 	def create_df(chrom_model):
	# 		correction2_df = pd.DataFrame({
	# 			'uncorrected': chrom_model.uncorrected_G.sum(axis=1),
	# 			'corrected': chrom_model.G.sum(axis=1),
	# 		}, index=chrom_model.config.WT1_TIMEPOINTS)
	# 		return correction2_df

	# 	g1_cor_df = create_df(self.chrom1_model)
	# 	g2_cor_df = create_df(self.chrom2_model)
	# 	g1_cor_df.to_csv(g1_correction_save_path)
	# 	g2_cor_df.to_csv(g2_correction_save_path)

	# 	# Save the f to disk
	# 	np.save(g1_save_path, G1)
	# 	np.save(g2_save_path, G2)

	# 	# Save the f to disk
	# 	np.save(f_save_path, f)

	# 	# Save the ptr to disk
	# 	np.save(ptr_save_path, self.chrom1_model.f_ptrs)

	# 	# Save meta information
	# 	from datetime import datetime
	# 	run_date = datetime.now().strftime("%D")

	# 	df = pd.DataFrame({
	# 		'rn': self.solver.rn, 'sn': self.solver.sn, 
	# 			'gm': self.solver.gamma,
	# 		'config1': self.chrom1_model.config.name,
	# 		'config2': self.chrom2_model.config.name,
	# 		'model1_path': self.chrom1_model.config.model_wt1_file,
	# 		'model2_path': self.chrom2_model.config.model_wt1_file,
	# 		'run_date': run_date,
	# 		'replicate': "combined",
	# 		'image_shape': str(GlobalConstants.IMAGE_SHAPE),
	# 		'rep1_+1': self.chrom1_model.computed_plus_one,
	# 		'rep1_+2': self.chrom2_model.computed_plus_one
	# 		},
	# 		index=[orf_name])
	# 	df.to_csv(meta_save_path, float_format="%.4f")

	# 	print_fl(f"Saved to {g1_save_path}...")
	# 	print_fl(f"Saved to {g2_save_path}...")
	# 	print_fl(f"Saved to {g1_correction_save_path}...")
	# 	print_fl(f"Saved to {g2_correction_save_path}...")
	# 	print_fl(f"Saved to {f_save_path}...")
	# 	print_fl(f"Saved to {ptr_save_path}...")
	# 	print_fl(f"Saved to {meta_save_path}...")

def load_chromatin_model_from_disk(gene_name, chromatin_dir, f_only=False):

	from src.config import load_yl_rg1_vst_config

	import os
	import glob
	
	f_pattern = os.path.join(chromatin_dir, f'*_f_*{gene_name}*')
	ptr_pattern = os.path.join(chromatin_dir, f'*_ptr_*{gene_name}*')
	meta_pattern = os.path.join(chromatin_dir, f'*_meta_*{gene_name}*')

	#g_filepath = glob.glob(g_pattern)[0]
	f_filepath = glob.glob(f_pattern)[0]
	ptr_filepath = glob.glob(ptr_pattern)[0]
	meta_filepath = glob.glob(meta_pattern)[0]

	config = load_yl_rg1_vst_config(1)

	f = np.load(f_filepath)

	if f_only:
		return f

	ptr = np.load(ptr_filepath)
	meta_data = pd.read_csv(meta_filepath)
	meta_data = meta_data.iloc[0]

	config = load_yl_rg1_vst_config(1)
	chromatin_model = ChromatinModel(config)

	chromatin_model.load_mnase_gene(gene_name)
	chromatin_model.create_deconvolution_bins()
	chromatin_model.setup_deconv_model()
	chromatin_model.setup_solver()

	chromatin_model.deconvolved_f_value = f
	chromatin_model.f_ptrs = ptr.flatten()

	chromatin_model.solver.rn = meta_data.rn
	chromatin_model.solver.sn = meta_data.sn
	chromatin_model.solver.gamma = meta_data.gm

	return chromatin_model


def load_combined_model(config_type='shared'):
	from src.config import load_configs_by_config_type
	chrom_config1, chrom_config2 = load_configs_by_config_type(config_type,
		with_copy_correction=True)
	from src.combined_chromatin_model import CombinedChromatinModel
	combined_model = CombinedChromatinModel(chrom_config1, chrom_config2)
	return combined_model
