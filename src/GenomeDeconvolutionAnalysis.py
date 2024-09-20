
import numpy as np
from src.global_config import GlobalConstants


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