
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PeakToTroughAnalysis:
	"""This class allows us to analyze the peak-to-trough deconvolution results of the chromatin.

	The objective for this analysis is to identify cell cycling genes, what threshold a gene can be considered
	cell cycling, how we can categorize, dilineate between cell cycling genes, and identify novel cell cycling regulatory
	chromatin that is improved by the deconvolution algorithm.
	"""

	def __init__(self, chromatin_dir):

		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')
		self.chromatin_dir = chromatin_dir

	def load_ptr_files(self):

		# Load ptr values
		file_paths = glob.glob(f'{self.chromatin_dir}/*_ptr_*.npy')
		ptrs_df = self.geneset[[]].copy()

		# Load the size of a flattened image
		loaded_ptrs = np.load(file_paths[0])
		m = loaded_ptrs.flatten().shape[0]

		for c in range(m):
			ptrs_df[c] = np.nan

		for path in file_paths:
			filename = path.split('/')[-1]
			orf_name = filename.split('_')[2]
			loaded_ptrs = np.load(path)
			ptrs_df.loc[orf_name] = loaded_ptrs.flatten()

		self.ptrs_df = ptrs_df#.dropna()

