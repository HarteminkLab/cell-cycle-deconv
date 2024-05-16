
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.utils import print_fl
from src.geneset import get_deconvolved_geneset
from src.config import load_yl_rg1_vst_config
from src.global_config import GlobalConstants

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

		entropy_df = pd.DataFrame(gene_nuc_entropy, 
			index=self.geneset_repl.index)

		# 0 will be a clear outlier with the range of entropy values we compute
		entropy_df.fillna(0.) 

		normalized_entropy_df = pd.DataFrame(normalized_gene_nuc_entropy, 
			index=self.geneset_repl.index)

		self.entropy_df = entropy_df
		self.normalized_entropy_df = normalized_entropy_df
		self.mean_entropy_per_gene = entropy_df.mean(axis=1)

		# Compute the entropy at time of replication
		repl_indices_t = self.get_repl_positions_in_t()
		entropy_at_repl = self.entropy_df.values[\
			np.arange(len(repl_indices_t)), repl_indices_t]
		self.entropy_at_repl = pd.Series(entropy_at_repl, 
			index=self.mean_entropy_per_gene.index) # Convert the nd array to a pd.series


	def plot_heatmap(self, plot_data, replication_df, vmin, vmax, fig, title, 
		cmap='RdBu_r', show_colorbar=False):

		if fig is None:
			fig = plt.figure(figsize=(6, 6))

		n = len(replication_df)

		plt.xlabel("Time, min")

		_, cg1_len, _ = self.config.get_g1_lens()
		ys = np.arange(n)

		# Offset the replication timing by the cg1 length
		replication_time = replication_df.replication_timing-cg1_len

		if self.entropy_tx_analysis_mode == "at_replication":
			plt.plot(replication_time, ys, lw=0.5, c='black')
		elif self.entropy_tx_analysis_mode == "avg_delta_5_replication":

			delta_replication = self.replication_delta_5.loc[replication_df.index]
			plt.plot(delta_replication.repl_tp_minus_delta, ys, lw=0.5, c='black')
			plt.plot(replication_time, ys, lw=0.5, c='black', alpha=0.5)
			plt.plot(delta_replication.repl_tp_plus_delta, ys, lw=0.5, c='black')

		plt.title(f"{title}")

		# Earliest at the top
		plt.ylim(n, 1)
		plt.yticks([])

		self.plot_gene_hm_dat_t_branch(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
		if show_colorbar:
			plt.colorbar()


	def plot_early_late_entropy_hm_comparision(self, k=200, normalize=False,
		subset_orfs=None, title=None):

		geneset = self.geneset_repl

		if subset_orfs is not None:
			geneset = geneset.loc[subset_orfs]

		sorted_geneset = geneset.sort_values('replication_timing')

		if normalize:

			if title is None: title = "Nucleosome entropy"
			nuc_entropy = self.entropy_df.loc[sorted_geneset.index].values
			self.plot_early_late_hm_comparision(nuc_entropy, sorted_geneset, k=k, 
				title=title, vmin=3, vmax=4., cmap='viridis')
		else:
			if title is None: title = "Normalized nucleosome entropy"
			nuc_entropy = self.normalized_entropy_df.loc[sorted_geneset.index].values
			self.plot_early_late_hm_comparision(nuc_entropy, sorted_geneset, k=k, 
				title=title)

	def plot_early_late_sm_hm_comparision(self, promoter_analysis, k=500,
		subset_orfs=None, title=None, normalize=False):

		geneset = self.geneset_repl

		if subset_orfs is not None:
			geneset = geneset.loc[subset_orfs]

		sorted_geneset = geneset.sort_values('replication_timing')
		t_indices = self.config.get_Hpositions_for_branch('t')

		# Sort the data by the replication timing
		sm_t_dat = promoter_analysis.sm_prom_occ_df.loc[sorted_geneset.index][t_indices]
		mean = sm_t_dat.values.mean(axis=1).reshape((-1, 1))
		normalized_sm_t_dat = (sm_t_dat.values - mean)

		if title is None:
			title = title

		if normalize:
			self.plot_early_late_hm_comparision(normalized_sm_t_dat, sorted_geneset, k=k, 
				title=title)
		else:
			self.plot_early_late_hm_comparision(sm_t_dat.values, sorted_geneset, k=k, 
				title=title, vmin=0, vmax=50, cmap="viridis")

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

		self.plot_early_late_hm_comparision(dat, sorted_geneset,  
			title="Deconvolved gene expression")

	def plot_early_late_plus_one_comparison(self, k=500):

		gene_plus_one_position_z = pd.read_csv(GlobalConstants.PLUS_ONE_FILEPATH)
		gene_plus_one_position_z = gene_plus_one_position_z.set_index('orf_name')

		# Sort the data by the replication timing
		sorted_geneset = self.geneset_repl.sort_values('replication_timing')

		p1_meta_data = pd.read_csv(GlobalConstants.PLUS_ONE_METADATA_FILEPATH).set_index('orf_name')
		sorted_geneset = p1_meta_data[['bin_max']].join(sorted_geneset, how='inner')

		# Filter out low nuc coverage genes
		sorted_geneset = sorted_geneset[sorted_geneset.bin_max > 10]

		dat = gene_plus_one_position_z.loc[sorted_geneset.index].values

		self.plot_early_late_hm_comparision(dat, sorted_geneset, 
			title="+1 nucleosome shift", vmin=-3, vmax=3, k=k)

	def plot_early_late_hm_comparision(self, data_to_plot, sorted_geneset, k=500, title=None,
		vmin=-3, vmax=3, cmap='RdBu_r'):

		fig = plt.figure(figsize=(13, 6))
		plt.suptitle(title, fontsize=23)
		plt.subplots_adjust(top=0.85)

		plt.subplot(1, 3, 1)
		n = len(data_to_plot)
		self.plot_heatmap(data_to_plot, sorted_geneset,
									 vmin=vmin, vmax=vmax, cmap=cmap,
									 title=f"All genes, n={n}", fig=fig)

		plt.subplot(1, 3, 2)
		self.plot_heatmap(data_to_plot[:k], sorted_geneset.head(k), 
			vmin=vmin, vmax=vmax, title=f"Earliest k={k}", fig=fig, 
			 cmap=cmap)

		plt.subplot(1, 3, 3)
		self.plot_heatmap(data_to_plot[-k:], sorted_geneset.tail(k), 
			vmin=vmin, vmax=vmax, title=f"Latest k={k}",
			fig=fig, cmap=cmap, show_colorbar=True)


	def plot_gene_hm_dat_t_branch(self, plot_data, cmap, vmin, vmax):

		n = len(plot_data)
		
		# Plot the G1 and postG1 heatmaps separately
		postG1_indices = self.config.get_Hpositions_for_phase('postG1')
		cg1_indices = self.config.get_Hpositions_for_phase('CG1')

		postG1_ts = self.config.get_phase_timepoints_for_phase('postG1')
		cg1_ts = self.config.get_phase_timepoints_for_phase('CG1')

		len_pg1_inds = len(postG1_indices)
		len_cg1_inds = len(cg1_indices)
		cg1_indices_in_t = np.arange(len_cg1_inds)
		postG1_indices_in_t = np.arange(len_cg1_inds, len_cg1_inds+len_pg1_inds)

		plt.imshow(plot_data[:, cg1_indices_in_t], aspect='auto', cmap=cmap,
			vmin=vmin,vmax=vmax, extent=[cg1_ts[0], postG1_ts[0], 1, n], origin='lower')
		plt.imshow(plot_data[:, postG1_indices_in_t], aspect='auto', cmap=cmap,
			vmin=vmin,vmax=vmax, extent=[postG1_ts[0], postG1_ts[-1], 1, n], origin='lower')

		plt.xlim(cg1_ts[0], postG1_ts[-1])
	

	def plot_hi_lo_entropy_hm_analysis(self):

		# Let's partition out the different entropy levels
		# discern whether there is a pattern amongst low and high entropic nucleosomes

		from src.helpers import get_quantile_values

		geneset = self.geneset_repl
		sorted_geneset = geneset.sort_values('replication_timing')
		nuc_entropy = self.entropy_df.loc[sorted_geneset.index]

		mean_entropy_per_gene = self.mean_entropy_per_gene.loc[sorted_geneset.index]
		plt.figure(figsize=(4, 2))
		plt.hist(mean_entropy_per_gene, bins=20)
		plt.title("Distribution of mean entropy per gene")
		plt.xlabel("Entropy, log2")

		segments, qvals, lens = get_quantile_values(mean_entropy_per_gene.dropna(), q=[0.25, 0.75])
		low_e_genes, med_e_genes, high_e_genes = segments
		qlow, qhigh = qvals

		plt.axvline(qlow, c='red')
		plt.axvline(qhigh, c='red')

		fig = plt.figure(figsize=(13, 4))
		plt.subplots_adjust(wspace=0.25)
		plt.subplot(1, 3, 1)
		repl_timing = sorted_geneset.loc[low_e_genes.index]
		dat = nuc_entropy.loc[low_e_genes.index]

		self.plot_heatmap(dat.values, repl_timing, 
			vmin=2.8, vmax=3.2, cmap='viridis', fig=fig, 
			title=f"Low entropy genes\nn={len(dat)}", 
			show_colorbar=True)

		plt.subplot(1, 3, 2)
		repl_timing = sorted_geneset.loc[med_e_genes.index]
		dat = nuc_entropy.loc[med_e_genes.index]

		self.plot_heatmap(dat.values, repl_timing, 
			vmin=3.2, vmax=3.5, cmap='viridis', fig=fig, 
			title=f"Medium entropy genes\nn={len(dat)}", 
			show_colorbar=True)

		plt.subplot(1, 3, 3)

		repl_timing = sorted_geneset.loc[high_e_genes.index]
		dat = nuc_entropy.loc[high_e_genes.index]

		self.plot_heatmap(dat.values, repl_timing, 
			vmin=3.5, vmax=4., cmap='viridis', fig=fig, 
			title=f"High entropy genes\nn={len(dat)}", 
			show_colorbar=True)

	def set_entropy_vs_tx_df(self, mode='at_replication'):

		# Set the mode of analysis, to compare at/during replication
		# at the exact time of replication vs [-delta, +delta] range around replication
		self.entropy_tx_analysis_mode = mode

		sorted_geneset_repl_df = self.geneset_repl.sort_values('replication_timing')

		if mode == 'at_replication':
			entropy_values = self.mean_entropy_per_gene
			expression_values = sorted_geneset_repl_df['expression_at_replication']
			mode_name = "at replication"

		elif mode == 'avg_delta_5_replication':
			entropy_values = self.delta_entropy_5['mean_delta_entropy']
			expression_values = self.delta_expression_5['mean_delta_expression']
			mode_name = "$replication \\pm$ 5 min "

		elif mode == 'avg_delta_10_replication':
			entropy_values = self.delta_entropy_10['mean_delta_entropy']
			expression_values = self.delta_expression_10['mean_delta_expression']
			mode_name = "$replication \\pm$ 10 min "

		else:
			raise ValueError(f"Unhandled mode: {mode}")

		# Entropy and expression data frame
		self.mode_name = mode_name
		entropy_vs_exp_df = sorted_geneset_repl_df.loc[entropy_values.index][['replication_timing']]
		entropy_vs_exp_df['expression'] = expression_values
		entropy_vs_exp_df['entropy_value'] = entropy_values

		self.entropy_vs_exp_df = entropy_vs_exp_df

	def get_entropy_vs_tx_df(self):
		return self.entropy_vs_exp_df
	
	def plot_entropy_vs_expression_analysis(self, x_cutoffs = []):
		def plot_entropy_vs_expression_scatter(entropy_vs_exp_df, title):

			from src.DensityScatterPlotter import DensityScatterPlotter

			entropy_vs_exp_df = entropy_vs_exp_df.dropna()

			x, y = entropy_vs_exp_df.expression, \
				entropy_vs_exp_df.entropy_value
			
			ax = plt.gca()
			n = len(x)
   
			ax.scatter(x, y, s=13, edgecolor='#ddd', facecolors='none', 
				zorder=1)

			# A custom color map with less extreme ends
			from src.plot_helpers import create_sub_colormap
			cmap = create_sub_colormap('YlGnBu_r', 0.3, 0.9, 'lighter_YlBuGn')

			density_scatter_pltr = DensityScatterPlotter()
			density_scatter_pltr.bw = [0.1, 0.025]
			density_scatter_pltr.cmap = cmap
			density_scatter_pltr.alpha = 1.
			density_scatter_pltr.set_data(x, y)
			density_scatter_pltr.plot_ax(ax, plot_colorbar=False)
			ax.set_title(f"{title},\nn={n}")
			ax.set_xlim(2.5, 20)
			ax.set_ylim(2, 4.2)

			for x in x_cutoffs:
				ax.axvline(x, c='black', lw=1, ls='dotted')

		entropy_vs_exp_df = self.get_entropy_vs_tx_df()

		from src.helpers import get_quantile_values

		# ---------- Compute segments

		segments, qvals, lens = get_quantile_values(entropy_vs_exp_df.replication_timing, 
			q=[0.25, 0.5, 0.75])
		early_genes, early_mid_genes, mid_late_genes, late_genes = segments
		qlow, qmid, qhigh = qvals

		# ---------- Plot

		fig = plt.figure(figsize=(20, 4))
		plt.suptitle(f"Gene expression vs Entropy, {self.mode_name}", fontsize=16)
		plt.subplots_adjust(top=0.8)

		plt.subplot(1, 5, 1)
		plot_entropy_vs_expression_scatter(entropy_vs_exp_df, title="All genes")
		plt.xlabel("Gene expression, VST")
		plt.ylabel("Entropy, log2")

		plt.subplot(1, 5, 2)		
		plot_entropy_vs_expression_scatter(entropy_vs_exp_df.loc[early_genes.index], 
			title="Early replicating genes")

		plt.subplot(1, 5, 3)
		plot_entropy_vs_expression_scatter(entropy_vs_exp_df.loc[early_mid_genes.index],
			title="Mid replicating  genes")

		plt.subplot(1, 5, 4)
		plot_entropy_vs_expression_scatter(entropy_vs_exp_df.loc[mid_late_genes.index],
			title="Mid replicating  genes")

		plt.subplot(1, 5, 5)
		plot_entropy_vs_expression_scatter(entropy_vs_exp_df.loc[late_genes.index],
			title="Late replicating genes")


	def ge_get_cutoffs_at_repl(self, x_cutoffs):
		"""Get the gene expression cutoffs, assign each gene to the 
		expression level cutoff group during replication time."""

		def assign_group_cutoffs(data_df, key, group_cutoffs, group_names):
			dat_df = data_df.copy()

			# assign groups by these ranges:
			for i in range(1, len(group_cutoffs)):
				
				group_name = group_names[i-1]
					
				left = group_cutoffs[i-1]
				right = group_cutoffs[i]
				sel = (dat_df[key] >= left) & (dat_df[key] < right)
				dat_df.loc[sel, 'group_id'] = i
				dat_df.loc[sel, 'group_name'] = group_name

			dat_df['group_id'] = dat_df['group_id'].astype(int)

			return dat_df

		grouped_data_df = assign_group_cutoffs(
			self.get_entropy_vs_tx_df(),
			'expression',  [0, x_cutoffs[0], x_cutoffs[1], 100], 
				[f'<{x_cutoffs[0]}', 
				 f'{x_cutoffs[0]}-{x_cutoffs[1]}', 
				 f'>{x_cutoffs[1]}'])

		return grouped_data_df


	def plot_box_plot_entropy_ge(self, x_cutoffs):

		# Let's try to segment by equal spaces of the expression values
		# Visually recreate the above plot, may not be equal number of groups.

		grouped_data_df = self.ge_get_cutoffs_at_repl(x_cutoffs)

		# ------- Box plotter ----------

		from src.boxplot import BoxPlotPlotter

		# Segments of early, mid, late genes
		entropy_vs_exp_df = self.get_entropy_vs_tx_df()
		from src.helpers import get_quantile_values
		segments, qvals, lens = get_quantile_values(entropy_vs_exp_df.replication_timing, 
			q=[0.25, 0.5, 0.75])
		early_genes, early_mid_genes, mid_late_genes, late_genes = segments

		box_plotter = BoxPlotPlotter()
		box_plotter.set_data([grouped_data_df.loc[early_genes.index],
							  grouped_data_df.loc[early_mid_genes.index],
							  grouped_data_df.loc[mid_late_genes.index],
							  grouped_data_df.loc[late_genes.index],
							 ], data_key='entropy_value', 
							 group_key='group_id',
							 group_name_key='group_name',
							category_names=[
								'Early',
								'Early-mid',
								'Mid-late',
								'Late'])

		# Plot the box plot
		plt.figure(figsize=(6, 4))
		ax = plt.gca()

		box_plotter.plot_box_plot(ax, title=f"Nucleosome Entropy vs expression, {self.mode_name}")


	def get_repl_positions_in_t(self):
		"""Get replication indices in terms of the top branches indexing.
		Subset the top branch indices, then convert the replication indices (that
		were in H indexing) into the top branches indexing.
		
		H = [0, 1, 2, 3]
		top = [2, 3] # subset of H
		repl = [2, 3] # indices in H
		
		return [0, 1] # updated indices in the top vector
		"""
		from src.helpers import indices_of_mapping_array

		indices_in_t_of_replication = self.geneset_repl.replication_H_index
		return self.convert_h_indices_to_t(indices_in_t_of_replication)

	def convert_h_indices_to_t(self, indices_in_t):
		"""Get replication indices in terms of the top branches indexing.
		Subset the top branch indices, then convert the replication indices (that
		were in H indexing) into the top branches indexing.
		
		H = [0, 1, 2, 3]
		top = [2, 3] # subset of H
		indices = [2, 3] # indices in H
		
		return [0, 1] # updated indices in the top vector
		"""

		from src.helpers import indices_of_mapping_array
		config = self.config
		t_timepoints = config.get_timepoints_for_branch('t')
		h_positions = config.get_Hpositions_for_branch('t')
		ret_indices = indices_of_mapping_array(h_positions, 
			indices_in_t)
		return ret_indices


	def compute_replication_window_deltas(self):
		"""Compute the indices and timepoints for replication span for +/- 5 and 10 minutes"""

		# Compute for 5 minute and 10 minutes
		self.replication_delta_5 = self.compute_replication_window_delta(5)
		self.replication_delta_10 = self.compute_replication_window_delta(10)


	def compute_replication_window_delta(self, delta_in_min=5):
		"""Compute the span of indices and timepoints around replication time to provide
		a range of when replication approximately occurred. 
		
		Handle edge cases:
		1. G1 and postG1 are not evenly spaced
		2. Handle wrap around if near the start or end of the top branches indices.
		"""
		
		from src.delta_replication_indexing import get_delta_rep_index
		
		config = self.config

		gene_repls = self.geneset_repl
		g1_indices = config.get_Hpositions_for_phase('CG1')
		postg1_indices = config.get_Hpositions_for_phase('postG1')

		g1_tps = config.get_phase_timepoints_for_phase('CG1')
		postg1_tps = config.get_phase_timepoints_for_phase('postG1')

		repl_deltas = gene_repls[['replication_timing', 'replication_H_index']].copy()

		repl_deltas['repl_H_index_minus_delta'] = 0
		repl_deltas['repl_H_index_plus_delta'] = 0
		repl_deltas['repl_tp_minus_delta'] = 0
		repl_deltas['repl_tp_plus_delta'] = 0
		repl_deltas['delta_min'] = delta_in_min

		for orf_name, gene in gene_repls.iterrows():

			index_of_rep_H = gene.replication_H_index
			rep_timing = gene.replication_timing

			idx_minus_delta_in_H, idx_plus_delta_in_H, \
			tp_minus_delta, tp_plus_delta \
				= get_delta_rep_index(g1_indices, postg1_indices, g1_tps, postg1_tps,
				index_of_rep_H, delta_in_min)

			repl_deltas.loc[orf_name, 'repl_H_index_minus_delta'] = idx_minus_delta_in_H
			repl_deltas.loc[orf_name, 'repl_H_index_plus_delta'] = idx_plus_delta_in_H

			repl_deltas.loc[orf_name, 'repl_tp_minus_delta'] = tp_minus_delta
			repl_deltas.loc[orf_name, 'repl_tp_plus_delta'] = tp_plus_delta

		return repl_deltas


	def compute_delta_entropies(self):
		"""Compute the delta entropy values for +/- 5 and 10 minutes"""

		self.delta_entropy_5 = self.compute_delta_entropy(self.replication_delta_5)
		self.delta_entropy_10 = self.compute_delta_entropy(self.replication_delta_10)

	def compute_delta_entropy(self, replication_delta_df):
		"""
		Compute the entropy at repl_time-delta  and repl_time+delta. Represents change in entropy through replication

		And the average entropy between [-delta, +delta]. Represents entropy through replication
		"""

		minus_delta_idx_H = replication_delta_df.repl_H_index_minus_delta
		plus_delta_idx_H = replication_delta_df.repl_H_index_plus_delta

		# Convert to indices in t
		# And keep as series object
		index = minus_delta_idx_H.index
		minus_delta_idx_t = self.convert_h_indices_to_t(minus_delta_idx_H)
		plus_delta_idx_t = self.convert_h_indices_to_t(plus_delta_idx_H)
		minus_delta_idx_t = pd.Series(minus_delta_idx_t, index=index)
		plus_delta_idx_t = pd.Series(plus_delta_idx_t, index=index)

		# ------------ Compute entropy ---------------

		n = len(minus_delta_idx_t)
		entropy_df = self.entropy_df

		# Entropy at -delta and +delta
		minus_delta_entropy = entropy_df.values[np.arange(n), minus_delta_idx_t]
		plus_delta_entropy = entropy_df.values[np.arange(n), plus_delta_idx_t]

		# Avreage entropy through replication [-delta, +delta]
		from src.helpers import get_mean_between_indices
		mean_delta_entropy = get_mean_between_indices(entropy_df, 
			minus_delta_idx_t, plus_delta_idx_t)
		mean_delta_entropy = pd.Series(mean_delta_entropy, index=index)

		# Create a dataframe for the computed entropy values
		delta_entropy_replication_df = replication_delta_df.copy()
		delta_entropy_replication_df['minus_delta_entropy'] = minus_delta_entropy
		delta_entropy_replication_df['plus_delta_entropy'] = plus_delta_entropy
		delta_entropy_replication_df['mean_delta_entropy'] = mean_delta_entropy

		return delta_entropy_replication_df


	def compute_delta_gene_expressions(self):
		"""Compute the delta entropy values for +/- 5 and 10 minutes"""
		self.delta_expression_5 = self.compute_delta_gene_expression(self.replication_delta_5)
		self.delta_expression_10 = self.compute_delta_gene_expression(self.replication_delta_10)

	def compute_delta_gene_expression(self, replication_delta_df):
		# Do the same for gene expression
		# gene_chrom_repl_analysis.compute_delta_gene_expression()

		gene_expression_df = self.gene_expression_f

		# gene expression is in terms of H, so the indices can be used directly
		minus_delta_idx_H = replication_delta_df.repl_H_index_minus_delta
		plus_delta_idx_H = replication_delta_df.repl_H_index_plus_delta

		index = minus_delta_idx_H.index
		gene_expression_df = gene_expression_df.loc[index].copy()
		# ------------ Compute entropy ---------------

		n = len(minus_delta_idx_H)

		# Expression at -delta and +delta
		minus_delta_expression = gene_expression_df.values[np.arange(n), minus_delta_idx_H]
		plus_delta_expression = gene_expression_df.values[np.arange(n), plus_delta_idx_H]

		# Average expresison through replication [-delta, +delta]
		from src.helpers import get_mean_between_indices
		mean_delta_expression = get_mean_between_indices(gene_expression_df, 
			minus_delta_idx_H, plus_delta_idx_H)
		mean_delta_expression = pd.Series(mean_delta_expression, index=index)

		# Create a dataframe for the computed entropy values
		delta_expression_replication_df = replication_delta_df.copy()
		delta_expression_replication_df['minus_delta_expression'] = minus_delta_expression
		delta_expression_replication_df['plus_delta_expression'] = plus_delta_expression
		delta_expression_replication_df['mean_delta_expression'] = mean_delta_expression

		return delta_expression_replication_df
		
	def plot_replication_delta_windows(self):

		fig = plt.figure(figsize=(8, 3))
		_, cg1_len, _ = self.config.get_g1_lens()

		plt.subplot(1, 2, 1)
		repl_deltas = self.replication_delta_5.sort_values('replication_timing')
		ys = np.arange(len(repl_deltas))
		plt.plot(repl_deltas.replication_timing-cg1_len, ys, lw=1, c='black')
		plt.plot(repl_deltas.repl_tp_minus_delta, ys, lw=1, c='red')
		plt.plot(repl_deltas.repl_tp_plus_delta, ys, lw=1, c='red')
		plt.xlim(-20, 65)
		plt.ylim(len(ys), 0)
		plt.title("$\\pm$5 min")
		plt.xlabel("Time, minutes")
		plt.ylabel("Gene index")
			
		plt.subplot(1, 2, 2)
		repl_deltas = self.replication_delta_10.sort_values('replication_timing')
		plt.plot(repl_deltas.replication_timing-cg1_len, ys, lw=1, c='black')
		plt.plot(repl_deltas.repl_tp_minus_delta, ys, lw=1, c='red')
		plt.plot(repl_deltas.repl_tp_plus_delta, ys, lw=1, c='red')
		plt.xlim(-20, 65)
		plt.ylim(len(ys), 0)
		plt.title("$\\pm$10 min")
		plt.xlabel("Time, minutes")
		plt.yticks([])

		plt.suptitle("Replication timing deltas")
		plt.subplots_adjust(top=0.8)
