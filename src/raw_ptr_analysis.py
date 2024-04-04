
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.reference_data import load_spellman_orfs, load_analysis_genes


class RawPTRAnalysis:
	"""Raw PTR Analysis
	"""

	def __init__(self, chromatin_dir, replicate):
		self.replicate = replicate
		self.geneset = load_analysis_genes()
		self.chromatin_dir = chromatin_dir


	def load_g_files(self):
		"""Load all of the gene G results into a dataframe, 
		flatten the F images for the dataframe."""

		geneset = load_analysis_genes()
			
		# Check the F images for each gene, determine if we 
		# need to modify the ptr calculation
		# In case +1 is too high of a psuedo count, we will reduce the pseudo count

		g_filepaths = glob.glob(f'{self.chromatin_dir}/*_g_*.npy')
		
		# Load the F images for each deconvolved gene
		current_g = np.load(g_filepaths[0])
		n_times, y_bins, x_bins = current_g.shape

		print("G shapes: ", n_times, y_bins, x_bins)

		all_gene_gs_df = pd.DataFrame(index=geneset.index, 
		   columns=np.arange(n_times*y_bins*x_bins))

		from src.timer import Timer

		timer = Timer()
		i = 0

		# For each deconvolved gene, load the ptr values and 
		# place them into the PTRs dataframe
		for path in g_filepaths:
			filename = path.split('/')[-1]
			orf_name = filename.split('_')[2]

			# Skip genes not in our analysis set
			# for runs in which we haven't filtered for low coverage genes yet
			if not orf_name in geneset.index.values: continue

			current_g = np.load(path)
			
			try:
				all_gene_gs_df.loc[orf_name] = current_g.flatten()
			except ValueError:
				print(f"Error with orf: {orf_name}. Skipping")
				i += 1
				continue

			if i % 2000 == 0:
				timer.print_time(f"{i+1}/{len(g_filepaths)}")
			i += 1

		# Drop genes that are not deconvolved
		all_gene_gs_df = all_gene_gs_df.dropna()

		self.all_gene_gs_df = all_gene_gs_df
		self.all_genes_g_imgs = all_gene_gs_df.values.reshape((-1, n_times, y_bins, x_bins))


	def compute_raw_ptr_values(self):
		"""Compute the raw ptr values for all genes"""
		from src.config import load_yl_rg1_vst_config, get_yl2019_chromatin_timepoints
		from src.timer import Timer

		print("Computing raw PTR values")
		timer = Timer()
		replicate = self.replicate
		config = load_yl_rg1_vst_config(replicate)
		timepoints = get_yl2019_chromatin_timepoints(replicate)

		# Compute the raw PTR for the gene for both replicates
		from src.peak_to_trough import compute_quantile_ptr

		mu0 = config.intervals_wt1[0][0]

		# Compute the peak to trough ratio for non recovery timepoints
		non_RG1_G = self.all_genes_g_imgs[:, timepoints > mu0, :]
		gene_g_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat), 1, non_RG1_G)
		mean_g_ptrs = gene_g_ptrs.mean(axis=1).mean(axis=1)

		mean_g_ptrs_df = self.all_gene_gs_df[[]].copy()
		mean_g_ptrs_df['mean_ptr'] = mean_g_ptrs

		self.mean_g_ptrs_df = mean_g_ptrs_df
		self.mean_g_ptrs = mean_g_ptrs
		timer.print_time("Done.")
