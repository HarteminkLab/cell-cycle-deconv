
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
		sm_range = sm_lens[1]-sm_lens[0]

		self.prom_x = prom_x
		self.prom_len = prom_len
		self.sm_lens = sm_lens
		self.sm_range = sm_range

		# Convert the ranges into indices
		self.promoter_bin_indices = (prom_x[0] + GlobalConstants.PROM_LEN) // GlobalConstants.BIN_WIDTH, \
		    (prom_x[1] + GlobalConstants.PROM_LEN) // GlobalConstants.BIN_WIDTH
		self.frag_len_bin_indices = sm_lens[0] // GlobalConstants.BIN_HEIGHT, \
		    sm_lens[1] // GlobalConstants.BIN_HEIGHT

	def load_f_files(self):
		"""Load all of the gene F results into a dataframe, flatten the F images for the dataframe."""

		chromatin_dir = self.chromatin_dir

		f_filepaths = glob.glob(f'{chromatin_dir}/*_f_*.npy')

		# Load the F images for each deconvolved gene
		current_f = np.load(f_filepaths[0])
		print("Shape of the loaded F:", current_f.shape)

		m_times, u_vals = current_f.shape
		all_gene_fs_df = pd.DataFrame(index=self.geneset.index, 
		   columns=np.arange(m_times*u_vals))

		from src.timer import Timer

		timer = Timer()
		i = 0

		# For each deconvolved gene, load the ptr values and place them into the PTRs dataframe
		for path in f_filepaths:
			filename = path.split('/')[-1]
			orf_name = filename.split('_')[2]

			# Skip genes not in our analysis set
			# for runs in which we haven't filtered for low coverage genes yet
			if not orf_name in self.geneset.index.values: continue

			current_f = np.load(path)
			all_gene_fs_df.loc[orf_name] = current_f.flatten()
			
			if i % 1000 == 0:
				timer.print_time(f"{i+1}/{len(f_filepaths)}")
			i += 1
		self.all_gene_fs_df = all_gene_fs_df
		self.all_f_values_flattened = self.all_gene_fs_df.values


	def compute_small_fragments_promoter_occupancy(self):

		f_df = self.all_gene_fs_df
		f_values = f_df.values

		chrom_f_imgs = f_values.reshape(
		   (f_values.shape[0], -1, *GlobalConstants.IMAGE_SHAPE))

		promoter_bin_indices, frag_len_bin_indices = self.promoter_bin_indices, \
		    self.frag_len_bin_indices

		sm_prom_bins = chrom_f_imgs[:, :, frag_len_bin_indices[0]:frag_len_bin_indices[1], 
		    promoter_bin_indices[0]:promoter_bin_indices[1]]
		sm_prom_bin_occ = sm_prom_bins.sum(axis=2).sum(axis=2)

		# Set nans to 0
		sm_prom_bin_occ = sm_prom_bin_occ.astype(float)
		sm_prom_bin_occ[np.isnan(sm_prom_bin_occ)] = 0

		self.sm_prom_bin_occ = sm_prom_bin_occ

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
