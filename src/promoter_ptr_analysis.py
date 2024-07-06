
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.reference_data import load_spellman_orfs, load_analysis_genes
from src.global_config import GlobalConstants


class PromoterPTRAnalysis:
	"""
	Analyse the promoter occupancy of deconvolved gene profiles.
	"""

	def __init__(self, chromatin_dir):
		self.geneset = load_analysis_genes()
		self.chromatin_dir = chromatin_dir

		# -------- Define the promoter and small fragments ---------------

		prom_x = -6*GlobalConstants.BIN_WIDTH, -2*GlobalConstants.BIN_WIDTH
		prom_len = prom_x[1]-prom_x[0]

		sm_lens = GlobalConstants.BIN_HEIGHT, GlobalConstants.BIN_HEIGHT*3

		self.prom_x = prom_x
		self.prom_len = prom_len
		self.sm_lens = sm_lens

		# Convert the ranges into indices
		self.promoter_bin_indices = (prom_x[0] + GlobalConstants.PROM_LEN) // GlobalConstants.BIN_WIDTH, \
			(prom_x[1] + GlobalConstants.PROM_LEN) // GlobalConstants.BIN_WIDTH
		self.frag_len_bin_indices = sm_lens[0] // GlobalConstants.BIN_HEIGHT, \
			sm_lens[1] // GlobalConstants.BIN_HEIGHT

	def load_f_files(self):
		"""Load all of the gene F results into a dataframe, flatten the F images for the dataframe."""

		chromatin_dir = self.chromatin_dir

		from src.deconv_data import load_f_files

		self.all_gene_fs_df = load_f_files(chromatin_dir, self.genes)
		self.geneset['gene_idx'] = np.arange(len(all_gene_fs_df))


	def correct_f_images_by_strand(self):
		f_values = self.all_gene_fs_df.values
		self.f_imgs = f_values.reshape((f_values.shape[0], -1, *GlobalConstants.IMAGE_SHAPE)).astype(float)
		crick_mask = self.geneset.strand == '-'
		self.strand_corrected_f_images = self.strand_correct_f_images(self.f_imgs, crick_mask).astype(float)


	def compute_small_fragments_promoter_occupancy(self):

		chrom_f_imgs = self.strand_corrected_f_images

		promoter_bin_indices, frag_len_bin_indices = self.promoter_bin_indices, \
			self.frag_len_bin_indices

		sm_prom_bins = chrom_f_imgs[:, :, frag_len_bin_indices[0]:frag_len_bin_indices[1], 
			promoter_bin_indices[0]:promoter_bin_indices[1]]
		sm_prom_bin_occ = sm_prom_bins.sum(axis=2).sum(axis=2)

		# Set nans to 0
		sm_prom_bin_occ = sm_prom_bin_occ.astype(float)
		sm_prom_bin_occ[np.isnan(sm_prom_bin_occ)] = 0

		self.sm_prom_bin_occ = sm_prom_bin_occ
		self.sm_prom_occ_df = pd.DataFrame(sm_prom_bin_occ, index=self.all_gene_fs_df.index)

	def compute_ptr_min_maxes(self):

		from src.config import load_yl_rg1_vst_config
		from src.peak_to_trough import compute_ptr, compute_max_min_locations

		sm_prom_bin_occ = self.sm_prom_bin_occ

		index = self.all_gene_fs_df.index

		# Use config of replicate 1, this won't matter between the two replicates
		# because both will map to the same columns in the final F vector
		self.config = load_yl_rg1_vst_config(1)

		promoter_ptr_arr = np.apply_along_axis(lambda row: compute_ptr(self.config, row,
			return_indices=False), 1, sm_prom_bin_occ)
		promoters_ptr_df = pd.DataFrame(promoter_ptr_arr, index=index,
					 columns=['mother_ptr', 'daughter_ptr', 'ptr'])

		promoter_min_max_arr = np.apply_along_axis(lambda row: compute_max_min_locations(
		self.config, row), 1, sm_prom_bin_occ)
		promoter_min_max_arr = np.apply_along_axis(lambda row: compute_max_min_locations(
		self.config, row), 1, sm_prom_bin_occ)

		min_rets_df = pd.DataFrame(promoter_min_max_arr[:, 0, :-1].astype(float), index=index,
			 columns=['min_value', 'min_f_idx', 'min_tp'])
		min_rets_df['min_phase'] = promoter_min_max_arr[:, 0, -1]
		max_rets_df = pd.DataFrame(promoter_min_max_arr[:, 1, :-1].astype(float), index=index,
			 columns=['max_value', 'max_f_idx', 'max_tp'])
		max_rets_df['max_phase'] = promoter_min_max_arr[:, 1, -1]

		self.promoter_min_maxs = min_rets_df.join(max_rets_df).join(promoters_ptr_df)


	def strand_correct_f_images(self, f_imgs, crick_mask):
	
		# Flip the crick genes horizontally
		crick_imgs = f_imgs[crick_mask]
		flipped_crick_imgs = np.flip(crick_imgs, axis=3)

		strand_corrected_f_imgs = f_imgs.copy()	
		strand_corrected_f_imgs[crick_mask] = flipped_crick_imgs

		return strand_corrected_f_imgs


	def plot_orf_name_imgs(self, time_delta_analysis, orf_name):

		f_index = self.geneset[['strand']].copy()
		f_index['gene_index'] = np.arange(len(f_index))


		gene = f_index.loc[orf_name]
		gene_imgs = self.f_imgs[gene.gene_index]

		gene_imgs = self.f_imgs[f_index.loc[orf_name].gene_index]

		fig, (peak_axs, trough_axs) = plt.subplots(2, 3, figsize=(8, 3))
		plt.subplots_adjust(top=0.8, hspace=0.35)
		
		# add box around promoter occupancy
		# add identified motifs from chip-exo and fimo
		x1, x2 = self.prom_x
		y1, y2 = self.sm_lens
		
		def show_img(ax, img):
			if gene.strand == '-':
				img = np.fliplr(img)

			from src.global_config import GlobalConstants

			ax.imshow(img, origin='lower', cmap='magma_r', aspect='auto',
				extent=[-GlobalConstants.PROM_LEN, GlobalConstants.GB_LEN,
				0, GlobalConstants.MAX_Y_LEN])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.axvline(-GlobalConstants.BIN_WIDTH, c='#555', lw=1, ls='solid', alpha=0.5)
			from src.plot_helpers import plot_rect2
			plot_rect2(ax, x1, y1, x2, y2, edgecolor='blue', fill=None, lw=0.75)
		
		def plot_branch_peak_trough(prom_indices, ax_peak, ax_trough, name):
			show_img(ax_peak, gene_imgs[int(prom_indices.index_max)])
			ax_peak.set_ylabel("Peak")
			show_img(ax_trough, gene_imgs[int(prom_indices.index_min)])
			ax_trough.set_ylabel("Trough")
			ax_peak.set_title(name)
		
		mother_prom_indices = time_delta_analysis.prom_mother_max_mins.loc[orf_name]
		daughter_prom_indices = time_delta_analysis.prom_daughter_max_mins.loc[orf_name]
		
		plot_branch_peak_trough(mother_prom_indices, peak_axs[0], trough_axs[0], 'Mother')
		plot_branch_peak_trough(daughter_prom_indices, peak_axs[1], trough_axs[1], 'Daughter')
		
		from src.sgd import get_gene_title_name
		
		gene_title = get_gene_title_name(orf_name)
		plt.suptitle(gene_title, fontsize=16)

		from src.plot_helpers import hide_spines
		hide_spines(peak_axs[2])
		hide_spines(trough_axs[2])

		# Load the tf sites bound to the promoter and
		# correct the location based on the plus one
		# (the plot is centered on the +1) and correct for strand
		tf_sites = time_delta_analysis.get_rossi_for_orf(orf_name)
		row = time_delta_analysis.plus_ones.loc[orf_name]
		plus_1 = row['+1_mean']
		tf_sites['loc_corrected'] = tf_sites['start'] - plus_1
		if gene.strand == '-':
			tf_sites['loc_corrected'] = -tf_sites['loc_corrected']

		def plot_tf_sites(ax, tf_sites):
			for idx, tf_site in tf_sites.iterrows():
				ax.scatter(tf_site.loc_corrected, 15, marker='^', s=10, color=tf_site.tf_color,
					label=tf_site.tf)

		for ax in [peak_axs[0], peak_axs[1], trough_axs[0], trough_axs[1]]:
			plot_tf_sites(ax, tf_sites)

		plot_tf_sites(peak_axs[2], tf_sites)
		peak_axs[2].legend(loc='upper left', ncol=3, frameon=False)
		peak_axs[2].set_ylim(0, 1)

		print(tf_sites[['tf', 'loc_corrected']])