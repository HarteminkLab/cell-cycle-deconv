
import matplotlib.pyplot as plt
import numpy as np
from src.global_config import GlobalConstants
import pandas as pd


class GenomeDeconvolutionAnalysis(object):
	"""Class to perform analysis on genome-wide deconvolution
	results. 


	Common tasks:
	- Retrieve the MNase data surrounding all genes
		- Flip crick stranded genes
		- And various subsets of genes

	- Retrieve the MNase data surrounding all origins
	- Retrieve nucleosome reads

	Notes:
	- Note that stacking genes by +1 nucleosome will not align bins perfectly, so the stacked histogram 
	will need to be in base pairs (or rounded to the nearest bin.)
	- 


	"""

	def __init__(self, outdir):
		self.outdir = outdir
		

	def load_mnase_span(self, chrom, mnase_span):
		"""Load the MNase data for a given span"""

		load_spans = self.get_load_spans_10k(mnase_span)
		loaded_f_dat = None

		for load_span in load_spans:
			loaded_f = np.load(f'{self.outdir}/data/chr{chrom}/chr{chrom}_{load_span[0]}_{load_span[1]}.npy')
			
			if loaded_f_dat is None:
				loaded_f_dat = loaded_f
			else:
				loaded_f_dat = np.concatenate([loaded_f_dat, loaded_f], axis=2)

		def subset_10k_to_desired_span(gene_10k_data, load_span, desired_span):
			# Subset the loaded 10k window to the desired span, update
			# the span if the span is not a multiple of the bin width
			bin_width = GlobalConstants.BIN_WIDTH
			first_bp = load_span[0]
			load_indices = (desired_span[0]-first_bp)//bin_width, (desired_span[1]-first_bp)//bin_width
			load_bps = load_indices[0]*bin_width+first_bp, load_indices[1]*bin_width+first_bp
			selected_loaded_data = gene_10k_data[:, :, load_indices[0]:load_indices[1]]
			selected_loaded_span = load_bps

			return selected_loaded_data, selected_loaded_span

		loaded_span = (load_spans[0][0], load_spans[-1][1])
		loaded_subset_data, loaded_subset_span = subset_10k_to_desired_span(loaded_f_dat, 
			loaded_span, mnase_span)

		return loaded_subset_data, loaded_subset_span


	def get_load_spans_10k(self, span):
		"""Get the 10k load spans that span the given span. Assumes the desired span is not larger than 10k."""
		import math

		start, end = span
		
		scale = 10000
		
		start_int, end_int = int(math.floor(start / scale)), int(math.ceil(end / scale))    
		
		# If the given span is between two 10k windows, return two spans
		if end_int-start_int > 1:
			spans = [
				(start_int*scale, (start_int+1)*scale),
				((start_int+1)*scale, end_int*scale),
			]
		else:
			spans = [(start_int*scale, end_int*scale)]
		
		return spans


	def plot_deconvolved_result(self, f_imgs, mnase_span, smooth=False, 
		normalize=False, vmax=1, figsize=(13, 11), xlims=None):
		from src.global_config import GlobalConstants
		from src.figure_configs import FiguresConfig
		from src.config import load_configs_by_config_type
		from src.orf_plotter import ORFAnnotationPlotter
		from src.geneset import get_deconvolved_geneset

		config, _  = load_configs_by_config_type('shared')
		indices, label_names = config.get_full_phase_indices()

		fig, axs = plt.subplots(len(indices)+1, 1, figsize=figsize)

		ax = axs[0]
		ax.set_xticks([])
		ax.set_yticks([])

		from src.helpers import smooth_data
		extents = [mnase_span[0], mnase_span[-1], 0, GlobalConstants.MAX_Y_LEN]

		for i in range(len(indices)):
			ax = axs[i+1]
			label_name = label_names[i]
			
			current_f_img = f_imgs[indices[i]]

			if smooth:
				current_f_img = smooth_data(current_f_img, size=5, sigma=0.5)

			if normalize:
				current_f_img = current_f_img / current_f_img.sum() * 200.

			ax.imshow(current_f_img, origin='lower', cmap='magma_r', vmax=vmax, aspect='auto',
					 extent=extents)
			ax.set_ylabel(label_name, rotation=0, ha='right', labelpad=9)
			ax.set_yticks([])
			
			if i == (len(indices)-1):
				ax.set_xticks(np.arange(mnase_span[0], mnase_span[1], 2000), minor=False)
				ax.set_xticks(np.arange(mnase_span[0], mnase_span[1], 200), minor=True)
			else:
				ax.set_xticks([])

			ax.axvline(0, c='black', lw=1, alpha=0.5, ls='dotted')

			if xlims is not None:
				ax.set_xlim(*xlims)

		plt.subplots_adjust(top=0.923)

		return fig


	def load_stacked_mnase_data_for_genes(self, genes, chroms=range(1, 17), 
		center_mode='+1', padding=2000, normalize=False, compute_gb_occ=False):
		"""Load set of genes and flip crick strand genes"""
		from src.geneset import get_deconvolved_geneset
		
		if 'chr' not in genes.columns:
			genes_reference = get_deconvolved_geneset()
			genes = genes.join(genes_reference[['chr', 'strand']], how='inner')
		
		# Combine gene data with +1
		if center_mode == "+1":
			rep1_plus1s = pd.read_csv('datasets/computed_mnase/rep1_plus_ones.csv')\
				.set_index('orf_name')
			rep2_plus1s = pd.read_csv('datasets/computed_mnase/rep2_plus_ones.csv')\
				.set_index('orf_name')
			combined_plus1s = (rep1_plus1s + rep2_plus1s)/2
			genes = combined_plus1s.join(genes, how='inner')
			center_key = '+1'
		elif center_mode == 'PAS_nuc':
			PAS_nuc = pd.read_csv('datasets/computed_mnase/computed_PAS_nucs.csv')\
				.set_index('orf_name')
			PAS_nuc['mean_PAS_nuc'] = PAS_nuc.mean(axis=1)
			genes = PAS_nuc.join(genes, how='inner')
			center_key = 'mean_PAS_nuc'
		else:
			raise ValueError(f"Invalid center_mode: {center_mode}")

		loaded_gene_dat = None
		i = 0
		failed_file_not_found_count = 0

		# todo: Parameterize selecting for gene body nucleosomes for occupancy:

		if center_mode == '+1':
			gb_nuc_indices = np.arange(90, 161)
		elif center_mode == 'PAS_nuc':
			 # Select all but the last nucleosome, image is 200 bins wide (halfway is 100)
			gb_nuc_indices = np.arange(40, 90)
		else:
			raise ValueError(f"Invalid center_mode: {center_mode}")

		gb_nuc_occ = pd.DataFrame(columns=range(178), index=genes.index)

		for chrom in chroms:
			chr_genes = genes[(genes.chr == chrom)]
			for orf_name, gene in chr_genes.iterrows():

				try:
					span = int(gene[center_key]-padding), int(gene[center_key]+padding)
				except ValueError:
					print("Failed to find center key for: ", gene.name)
					failed_file_not_found_count += 1
					continue
				
				try:
					gene_data, loaded_span = self.load_mnase_span(chrom, span)
				except FileNotFoundError:
					failed_file_not_found_count += 1
					continue

				if gene.strand == '-':
					gene_data = np.flip(gene_data, axis=2)

				if normalize:
					TOTAL = 1000
					gene_data_sum = gene_data.sum(axis=1).sum(axis=1).reshape((-1, 1, 1))
					gene_data = gene_data / gene_data_sum * TOTAL

				if loaded_gene_dat is None:
					loaded_gene_dat = np.zeros(gene_data.shape)

				if compute_gb_occ:
					gb_nuc_occ.loc[orf_name, :] = gene_data[:, :, gb_nuc_indices].sum(axis=1).sum(axis=1)

				loaded_gene_dat += gene_data
				i += 1
				
		if failed_file_not_found_count > 0:
			print(f"Failed to load {failed_file_not_found_count} files")

		mean_gene_dat = loaded_gene_dat / len(genes)

		if compute_gb_occ:
			return mean_gene_dat, gb_nuc_occ

		return mean_gene_dat
