
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.utils import print_fl
from src.geneset import get_deconvolved_geneset
from src.config import load_yl_rg1_vst_config


class GeneChromatinReplicationAnalysis:
	"""
	Analysis to examine chromatin pre/post replication fork passing through.

	Goal: Make it easy to examine different subsets of genes at different time deltas
	before and after the replication.

	To identify potential effects of the replication fork in the chromatin at genes 
	(expressed/unexpressed), transcribing
	with and towards the replication fork, etc.
	"""

	def __init__(self, promoter_analysis, ge_analysis, repl_timing):

		repl_timing = repl_timing.dropna()

		# Using the timepoint values from config 1
		self.config = load_yl_rg1_vst_config(1)

		geneset = get_deconvolved_geneset()

		# Now we will need to index the geneset so we can subset the f images properly
		# Replication timing profiles were computed for a subset ~5200 genes
		# So we will need to keep track of the indices of the genes and F properly

		geneset_repl = geneset[['gene', 'TSS', 'chr', 'strand']].copy()
		geneset_repl['f_gene_index'] = np.arange(len(geneset_repl))

		# Subset by the genes for which we have replication timing
		geneset_repl = geneset_repl.loc[repl_timing.index]
		geneset_repl['replication_timing'] = repl_timing.timing
		geneset_repl['replication_H_index'] = repl_timing.H_index

		# There may be some genes for which we don't have gene expression data as well
		gene_index = geneset_repl.join(ge_analysis.gene_expression_f[[]], how='inner').f_gene_index
		gene_orf_name_index = geneset_repl.join(ge_analysis.gene_expression_f[[]], 
			how='inner').index.values
		geneset_repl = geneset_repl.loc[gene_orf_name_index]

		# Subset the set f images (np array) so use the integer index
		sc_f_w_repl_images = promoter_analysis.strand_corrected_f_images[gene_index]

		print(len(geneset_repl), "genes with replication timing")
		print("Shape of the f images: ", sc_f_w_repl_images.shape)

		# Now that we have subset the f images, reset the f gene indexing
		geneset_repl['f_gene_index'] = np.arange(len(geneset_repl))

		self.gene_expression_f = ge_analysis.gene_expression_f.loc[gene_orf_name_index]
		self.geneset = geneset
		self.geneset_repl = geneset_repl
		self.sc_f_w_repl_images = sc_f_w_repl_images

		# Next compute the expression value at replication time
		repl_h_index = self.geneset_repl.replication_H_index
		expression_at_replication = self.gene_expression_f.apply(lambda 
			row: row.iloc[repl_h_index[row.name]], axis=1)
		self.geneset_repl['expression_at_replication'] = expression_at_replication


		# Add replication direction
		repl_direction = self.compute_replication_direction(geneset_repl)
		self.geneset_repl['replication_direction'] = repl_direction.repl_direction

	def compute_replication_direction(self, repl_genes):
		def compute_left_right_fork(prev_gene, gene, next_gene):

			direction = None
			# If gene to the left is earlier
			# and gene to the right is later,
			# gene transcribes left to right from the watson strand pov
			if (prev_gene.replication_timing <= gene.replication_timing and
				gene.replication_timing <= next_gene.replication_timing):
				direction = 'right'

			# If gene to the right is earlier
			# and gene to the late is later,
			# gene transcribes right to left from the watson strand pov
			elif (prev_gene.replication_timing >= gene.replication_timing and
				gene.replication_timing >= next_gene.replication_timing):
				direction = 'left'

			# Gene is earlier than neighbors, early peak
			elif (prev_gene.replication_timing >= gene.replication_timing and
				gene.replication_timing <= next_gene.replication_timing): 
				direction = 'early_peak'
				
			# Gene is later than neighbors, late trough
			elif (prev_gene.replication_timing <= gene.replication_timing and
				gene.replication_timing >= next_gene.replication_timing): 
				direction = 'late_trough'

			return direction

		# Iterate and compute the direction of transcription using neighbors
		repl_dir_genes = self.geneset_repl.copy()
		repl_dir_genes['repl_direction'] = None

		for chrom in range(1, 17):
			
			chr_repl_genes = repl_genes[repl_genes.chr == chrom]
			n_chr_genes = len(chr_repl_genes)
			
			for i in range(0, n_chr_genes):
				
				prev_i = i-1
				next_i = i+1
				
				if i == 0:
					prev_i = i
				elif i == n_chr_genes-1:
					next_i = i
					
				prev_gene = chr_repl_genes.iloc[prev_i]
				gene = chr_repl_genes.iloc[i]
				next_gene = chr_repl_genes.iloc[next_i]
				direction = compute_left_right_fork(prev_gene, gene, next_gene)
				
				repl_dir_genes.loc[gene.name, 'repl_direction'] = direction
		return repl_dir_genes

	def plot_replication_direction(self, chrom):
		repl_dir_genes = self.geneset_repl
		chr_genes = repl_dir_genes[repl_dir_genes.chr == chrom]

		color_map = {
			'left': 'blue',
			'right': 'red',
			'early_peak': 'gray',
			'late_trough': 'black'
		}

		marker_map = {
			'left': '<',
			'right': '>',
			'early_peak': '^',
			'late_trough': 'v'
		}

		plt.figure(figsize=(13, 2))

		for direction in marker_map.keys():
			dir_genes = chr_genes[chr_genes.replication_direction == direction]
			plt.scatter(dir_genes.TSS, dir_genes.replication_timing, c=color_map[direction],
			 marker=marker_map[direction], s=20)

		plt.ylim(65, 25)
		plt.title(f"Replication direction, chr{chrom}")

	def select_subset(self, subset_orfs=None, delta_index=25):

		geneset_repl = self.geneset_repl.copy()

		if subset_orfs is not None:
			self.subset_orfs = subset_orfs
			subset_genes = geneset_repl.loc[subset_orfs]
			sc_subset_f_imgs = self.sc_f_w_repl_images[subset_genes.f_gene_index]
		else:
			subset_genes = geneset_repl
			sc_subset_f_imgs = self.sc_f_w_repl_images

		n = len(sc_subset_f_imgs)

		# Set nans to 0
		sc_subset_f_imgs[np.isnan(sc_subset_f_imgs)] = 0

		# Get the prior and post replication indices
		# trying delta index of 10 columns (not sure how many minutes this is)
		self.delta_index = delta_index

		prior_index = subset_genes.replication_H_index.values-self.delta_index
		at_index = subset_genes.replication_H_index.values
		post_index = subset_genes.replication_H_index.values+self.delta_index

		# If post index is greater than the available indices, wrap around
		tp_indices = self.config.get_Hpositions_for_branch('t')
		last_tp_index = tp_indices[-1]
		sel_wrap_around = post_index > last_tp_index

		# wrap around the index
		# todo: note we need to wrap around the timepoints as well
		post_index[sel_wrap_around] = post_index[sel_wrap_around]-(last_tp_index+1)

		rg1_len, cg1_len, dg1_len = self.config.get_g1_lens()

		subset_genes['prior_i'] = prior_index
		subset_genes['post_i'] = post_index

		subset_genes['prior_tp'] = [self.config.get_timepoint_for_index(i)+cg1_len
		 for i in prior_index]
		subset_genes['post_tp'] = [self.config.get_timepoint_for_index(i)+cg1_len 
		for i in post_index]

		def compute_delta_timing(subset_genes):
			from scipy import stats
			subset_genes = subset_genes.copy().dropna()
			min_delta = subset_genes.post_tp - subset_genes.replication_timing
			# Use mode, as most genes will not require the wrap-around timepoint
			# calculation
			# todo: This shouldn't be necessary if we are able to calculate the 
			# wrap-around
			# for the timepoints properly.
			return stats.mode(min_delta.values, keepdims=True)[0][0]

		# Compute the delta pre-replication and post-replication in minutes
		self.delta_min = compute_delta_timing(subset_genes)

		prior_f_images = sc_subset_f_imgs[np.arange(n), prior_index]
		at_f_images = sc_subset_f_imgs[np.arange(n), at_index]
		post_f_images = sc_subset_f_imgs[np.arange(n), post_index]

		self.prior_f_images = prior_f_images
		self.at_f_images = at_f_images
		self.post_f_images = post_f_images
		self.subset_genes = subset_genes

	def plot_pre_post_images(self, title=None):
		plt.figure(figsize=(8, 5))
		plt.subplots_adjust(hspace=0.5, wspace=0.2)

		def plot_f_img(img):
			plt.imshow(img, cmap='magma_r', 
					   origin='lower', aspect='auto', vmin=0, vmax=30)
			plt.xticks([])
			plt.yticks([])
			plt.axvline(9, c='black', lw=0.5, ls='solid', alpha=0.5)

		prior = self.prior_f_images.mean(axis=0)
		post = self.post_f_images.mean(axis=0)
		at = self.at_f_images.mean(axis=0)
		
		plt.subplot(4, 3, 1)
		plot_f_img(prior)
		plt.title(f"Prior to replication -{self.delta_min:.0f} min", fontsize=9)

		plt.subplot(4, 3, 4)
		plot_f_img(at)
		plt.title("At replication", fontsize=9)

		plt.subplot(4, 3, 7)
		plot_f_img(post)
		plt.title(f"Post-replication +{self.delta_min:.0f} min", fontsize=9)

		# -------- Difference trace plots plots ---------------

		from src.chromatin_metrics import yl_rep2_len_spans
		from src.global_config import GlobalConstants

		small_lens, med_lens, nuc_lens = yl_rep2_len_spans()
		
		small_bins = small_lens[0]//GlobalConstants.BIN_HEIGHT,  \
			small_lens[1]//GlobalConstants.BIN_HEIGHT
		med_lens = med_lens[0]//GlobalConstants.BIN_HEIGHT, \
			med_lens[1]//GlobalConstants.BIN_HEIGHT
		nuc_bins = nuc_lens[0]//GlobalConstants.BIN_HEIGHT, \
			nuc_lens[1]//GlobalConstants.BIN_HEIGHT

		def plot_frag_traces(trace_img):

			# -------- subpanel definitions -----------
			sub_panel_height = 1.0
			sub_panel_height_2 = sub_panel_height/2

			num_panels = 3
			total_height = sub_panel_height*3
			total_height_2 = total_height/2

			# scale the traceplots so they fit in the subpanels
			scalar = 0.75

			# -----------------------------------------

			small_trace = trace_img[small_bins[0]:small_bins[1]].sum(axis=0)*scalar
			intermediate_trace = trace_img[med_lens[0]:med_lens[1]].sum(axis=0)*scalar
			nuc_trace = trace_img[nuc_bins[0]:nuc_bins[1]].sum(axis=0)*scalar

			# Fill between plot
			def plot_fill_pos_neg(trace_data, offset):
				xs = np.arange(0, len(trace_data))
				positive_color = plt.get_cmap('RdBu_r')(0.7)
				negative_color = plt.get_cmap('RdBu_r')(0.3)

				positive_trace = trace_data.copy()
				positive_trace[positive_trace < 0] = 0
				# truncate within bounds
				positive_trace[positive_trace > sub_panel_height_2] = sub_panel_height_2 


				negative_trace = trace_data.copy()
				negative_trace[negative_trace > 0] = 0
				# truncate within bounds
				negative_trace[negative_trace < -sub_panel_height_2] = -sub_panel_height_2

				plt.fill_between(xs, positive_trace+offset, offset, 
					color=positive_color, lw=0)
				plt.fill_between(xs, negative_trace+offset, offset, 
					color=negative_color, lw=0)

			plot_fill_pos_neg(nuc_trace, sub_panel_height)
			plot_fill_pos_neg(intermediate_trace, 0)
			plot_fill_pos_neg(small_trace, -sub_panel_height)

			# tss
			plt.axvline(9, c='black', lw=0.5, ls='solid', alpha=0.5)

			# Dividers between subpanels
			plt.axhline(-sub_panel_height_2, c='black', lw=1)
			plt.axhline(sub_panel_height_2, c='black', lw=1)


			plt.ylim(-total_height_2, total_height_2)
			plt.xlim(0, trace_img.shape[1]-1)
			plt.xticks([])
			plt.yticks([-sub_panel_height, 0, sub_panel_height], ['Sm', "Int", "Nuc"], 
				rotation=0, ha='right',
				fontsize=7)
			plt.gca().tick_params(axis='y', which='major', length=0, pad=2)

		at_minus_prior = at - prior
		post_minus_at = post - at
		post_minus_prior = post - prior

		# plt.subplot(4, 3, 2)
		# plot_frag_traces(at_minus_prior)
		# plt.title("Frag. Difference: At - Prior", fontsize=9)

		# plt.subplot(4, 3, 5)
		# plot_frag_traces(post_minus_at)
		# plt.title("Frag. Difference: Post - At", fontsize=9)

		plt.subplot(4, 3, 5)
		plot_frag_traces(post_minus_prior)
		plt.title("Frag. Difference: Post - Prior", fontsize=9)

		# -------- Difference plots ---------------

		def plot_f_img_diff(img):
			plt.imshow(img, origin='lower', aspect='auto', 
					   cmap='RdBu_r', vmin=-1, vmax=1)
			plt.xticks([])
			plt.yticks([])
			plt.gca().yaxis.set_label_position("right")
			plt.axvline(9, c='black', lw=0.5, ls='solid', alpha=0.5)

		plt.subplot(4, 3, 6)
		plot_f_img_diff(post_minus_prior)
		plt.title("Difference: Post - Prior", fontsize=9)

		if title is not None:
			title = f"{title}, Prior/Post Replication, n={len(self.subset_genes)}"
		else:
			title = f"Prior/Post Replication, n={len(self.subset_genes)}"

		plt.suptitle(title)


	def compute_gene_nuc_entropy(self):

		from src.chromatin_metrics import yl_rep2_len_spans
		from src.global_config import GlobalConstants
		from src.helpers import calc_entropy

		gene_f_images = self.sc_f_w_repl_images.astype(float)

		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')

		bin_width, bin_height = GlobalConstants.BIN_WIDTH, GlobalConstants.BIN_HEIGHT

		small_lens, med_lens, nuc_lens = yl_rep2_len_spans()
		nuc_bins = nuc_lens[0]//bin_height, \
			nuc_lens[1]//bin_height
		med_bins = med_lens[0]//bin_height, \
			med_lens[1]//bin_height

		# Starting with the the end of the promoter bins
		# through the end of the total number of bins
		# include the +1, so subtract 1
		gene_body_bins = GlobalConstants.PROM_LEN//bin_width-1, GlobalConstants.NUM_BINS_X

		# Get the nucleosomes for all genes and collapse the fragment length dimension    
		nucs_over_time = gene_f_images[:, t_indices, nuc_bins[0]:nuc_bins[1], 
			gene_body_bins[0]:gene_body_bins[1]].sum(axis=2)

		print("Shape of the f images: ", gene_f_images.shape)
		print("Shape of the bins to compute entropy over: ", nucs_over_time.shape)

		nucs_over_time = nucs_over_time.astype(float)
		nucs_over_time[np.isnan(nucs_over_time)] = 0.
		gene_nuc_entropy = np.apply_along_axis(calc_entropy, 2, nucs_over_time)

		# Z-score normalization
		normalized_gene_nuc_entropy = (gene_nuc_entropy - gene_nuc_entropy.mean(axis=1)\
									.reshape((-1, 1))) / \
									gene_nuc_entropy.std(axis=1).reshape((-1, 1))
		self.gene_nuc_entropy, self.normalized_gene_nuc_entropy = \
			gene_nuc_entropy, normalized_gene_nuc_entropy


	def plot_heatmap(self, plot_data, replication_df, vmin, vmax, fig, title):

		if fig is None:
			fig = plt.figure(figsize=(6, 6))

		n = len(replication_df)

		# Plot the G1 and postG1 heatmaps separately
		postG1_indices = self.config.get_Hpositions_for_phase('postG1')
		cg1_indices = self.config.get_Hpositions_for_phase('CG1')

		postG1_ts = self.config.get_phase_timepoints_for_phase('postG1')
		cg1_ts = self.config.get_phase_timepoints_for_phase('CG1')

		len_pg1_inds = len(postG1_indices)
		len_cg1_inds = len(cg1_indices)
		cg1_indices_in_t = np.arange(len_cg1_inds)
		postG1_indices_in_t = np.arange(len_cg1_inds, len_cg1_inds+len_pg1_inds)

		plt.imshow(plot_data[:, cg1_indices_in_t], aspect='auto', cmap='RdBu_r',
			vmin=vmin,vmax=vmax, extent=[cg1_ts[0], postG1_ts[0], 1, n], origin='lower')
		plt.imshow(plot_data[:, postG1_indices_in_t], aspect='auto', cmap='RdBu_r',
			vmin=vmin,vmax=vmax, extent=[postG1_ts[0], postG1_ts[-1], 1, n], origin='lower')

		plt.xlabel("Deconvolved time along mother branch, min")
		plt.ylabel("Gene rank sorted by replication time")

		_, cg1_len, _ = self.config.get_g1_lens()
		ys = np.arange(n)

		# Offset the replication timing by the cg1 length
		plt.scatter(replication_df.replication_timing-cg1_len, ys, s=0.5, 
			c='black', marker='D')
		plt.title(f"{title}")

		# Earliest at the top
		plt.ylim(n, 1)

	def plot_early_late_entropy_hm_comparision(self, k=500):
		sorted_geneset = self.geneset_repl.sort_values('replication_timing')
		nuc_entropy = self.normalized_gene_nuc_entropy[sorted_geneset.f_gene_index]
		self.plot_early_late_hm_comparision(nuc_entropy, k=k, 
			title="Nucleosome entropy")

	def plot_early_late_sm_hm_comparision(self, promoter_analysis, k=500):

		sorted_geneset = self.geneset_repl.sort_values('replication_timing')
		t_indices = self.config.get_Hpositions_for_branch('t')

		# Sort the data by the replication timing
		sm_t_dat = promoter_analysis.sm_prom_occ_df.loc[sorted_geneset.index][t_indices]
		mean = sm_t_dat.values.mean(axis=1).reshape((-1, 1))
		normalized_sm_t_dat = (sm_t_dat.values - mean)

		self.plot_early_late_hm_comparision(normalized_sm_t_dat, k=k, 
			title="Promoter small fragments")

	def plot_early_late_gene_expression_comparison(self, ge_analysis, k=500):

		sorted_geneset = self.geneset_repl.sort_values('replication_timing')
		t_indices = self.config.get_Hpositions_for_branch('t')
		gene_expression_f = ge_analysis.gene_expression_f.join(sorted_geneset[[]], 
			how='inner')

		# Sort the data by the replication timing
		ge_sorted_by_rep = gene_expression_f.loc[sorted_geneset.index]
		dat = ge_sorted_by_rep.values.astype(float)

		dat = dat[:, t_indices]
		dat = (dat - dat.mean(axis=1).reshape((-1, 1))) / (dat.std(axis=1).reshape((-1, 1)))

		self.plot_early_late_hm_comparision(dat, 
			title="Deconvolved gene expression")

	def plot_early_late_plus_one_comparison(self, k=500):

		plus_fp = 'output/deconvolved_plus_one_tracking/computed_plus_one_movement_meannorm.csv'
		gene_plus_one_position_z = pd.read_csv(plus_fp)
		gene_plus_one_position_z = gene_plus_one_position_z.set_index('orf_name')

		# Sort the data by the replication timing
		sorted_geneset = self.geneset_repl.sort_values('replication_timing')

		p1_meta_data = pd.read_csv('output/deconvolved_plus_one_tracking/p1_meta_data.csv').set_index('orf_name')
		sorted_geneset = p1_meta_data[['bin_max']].join(sorted_geneset, how='inner')

		# Filter out low nuc coverage genes
		sorted_geneset = sorted_geneset[sorted_geneset.bin_max > 10]

		print(len(sorted_geneset))

		dat = gene_plus_one_position_z.loc[sorted_geneset.index].values

		self.plot_early_late_hm_comparision(dat, 
			title="+1 nucleosome shift", vmin=-3, vmax=3, k=k)

	def plot_early_late_hm_comparision(self, data_to_plot, k=500, title=None,
		vmin=-3, vmax=3):

		sorted_geneset = self.geneset_repl.sort_values('replication_timing')

		fig = plt.figure(figsize=(13, 6))
		plt.suptitle(title, fontsize=23)
		plt.subplots_adjust(top=0.85)

		plt.subplot(1, 3, 1)
		n = len(data_to_plot)
		self.plot_heatmap(data_to_plot, sorted_geneset,
									 vmin=vmin, vmax=vmax, title=f"All genes, n={n}", fig=fig)

		plt.subplot(1, 3, 2)
		self.plot_heatmap(data_to_plot[:k], sorted_geneset.head(k), 
			vmin=vmin, vmax=vmax, title=f"Earliest k={k}", fig=fig)

		plt.subplot(1, 3, 3)
		self.plot_heatmap(data_to_plot[-k:], sorted_geneset.tail(k), 
			vmin=vmin, vmax=vmax, title=f"Latest k={k}",
			fig=fig)


	
def get_repl_positions_in_t(gene_chrom_repl_analysis, indices_in_t_of_replication):
	"""Get replication indices in terms of the top branches indexing.
	Subset the top branch indices, then convert the replication indices (that
	were in H indexing) into the top branches indexing.
	
	H = [0, 1, 2, 3]
	top = [2, 3] # subset of H
	repl = [2, 3] # indices in H
	
	return [0, 1] # updated indices in the top vector
	"""
	from src.helpers import indices_of_mapping_array

	config = gene_chrom_repl_analysis.config
	t_timepoints = config.get_timepoints_for_branch('t')
	h_positions = config.get_Hpositions_for_branch('t')
	ret_indices = indices_of_mapping_array(h_positions, 
		indices_in_t_of_replication)
	return ret_indices


