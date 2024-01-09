
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class PeakToTroughAnalysis:
	"""This class will be used to process the chromatin peak to trough values, analyse,
	and plot. 

	We will also eventually analyze with respect to gene expression.
	"""

	def __init__(self):
		genes = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv')\
    		.set_index('orf_name')

		import glob
		import os
		import numpy as np

		directory_path = 'output/deconvolve_chromatin_2023_12_15/'
		pattern = os.path.join(directory_path, '*ptr*')
		files_with_ptr = glob.glob(pattern)

		# Let's store them in a dictionary indexed by the orf name to start, then we can reorder 
		# them into a numpy array
		orf_name_ptrs_dic = {}
		for filename in files_with_ptr:

		    orf_name = filename.split('/')[-1].split('_')[-1].replace('.npy', '')
		    gene_ptr = np.load(filename)

		    orf_name_ptrs_dic[orf_name] = gene_ptr

		print(f"Created a dictionary of PTR arrays of size: {len(orf_name_ptrs_dic)}")
		print(f"So, there were { len(genes) - len(orf_name_ptrs_dic)} genes that did not deconvolve")

		# Next, we let's make a numpy array ordered by the original genes list
		all_ptr_arr = np.zeros((len(genes), gene_ptr.shape[0], gene_ptr.shape[1]))

		for index in range(len(genes)):
		    
		    orf_name = genes.index.values[index]
		    
		    # Skip if we don't have a deconvolution for the gene
		    if orf_name in orf_name_ptrs_dic.keys():
		        ptr_img = orf_name_ptrs_dic[orf_name]
		        all_ptr_arr[index] = ptr_img

		print(f"Created a numpy array of all deconvolved gene PTRs of shape: {all_ptr_arr.shape}")

		self.all_ptr_arr = all_ptr_arr
		self.genes = genes


	def plot_histograms(self):
		"""
		Plot the histograms of every PTR value for every gene, as well as the maximum values.
		"""
		self.threshold = 100

		all_ptr_values = self.all_ptr_arr.flatten()

		# Let's get an idea of the range of PTR values we have for each gene, this will be
		# agnostic to the bin location in the gene
		plt.subplot(2, 2, 1)
		plt.hist(all_ptr_values, bins=100)
		plt.yscale('log')
		plt.title("Histogram of PTR values\nfor every bin for every gene")

		print(np.quantile(all_ptr_values, [0.8, 0.9, 0.95]))

		# We can try selecting the max PTR for each gene and plotting that
		plt.subplot(2, 2, 2)
		# Let's get an idea of the range of PTR values we have for each gene, this will be
		# agnostic to the bin location in the gene
		max_ptr_per_gene = self.all_ptr_arr.max(axis=1).max(axis=1)

		plt.hist(max_ptr_per_gene, bins=100)
		plt.yscale('log')
		plt.title("Histogram of the\nmaximum PTR for each gene")


	def examine_threshold_values(self):
		"""This script tries a few quantiles/threshold values against
		the flattened ptr values (all ptr values for a gene window for all genes)

		And examines how many genes exceed or are stricly below this value.

		TODO: 
		Currently thinking through if thresholding is how we want to examine this data.
		Possibly interesting when thinking of proportion of the local window around a gene TSS
		as cycling rather than one cell of the window.

		Will require further thought...

		"""

		# Drop nan values
		all_ptr_values = self.all_ptr_arr.flatten()
		all_ptr_values = all_ptr_values[~np.isnan(all_ptr_values)]

		# Let's try some thresholds, then count how many genes are above and below these thresholds
		prop_thresholds = [0.1, 0.15, 0.3, 0.6, 0.75, 0.9, 0.95, 0.975, 0.999]
		threshold_values = []
		higher_genes = []
		lower_genes = []
		n = len(max_ptr_values)

		print(f"For Peak-to-trough (PTR) thresholds of:")
		print("\n--------------------------------------------------------------------------\n")
		for i in range(len(prop_thresholds)):
		    prop = prop_thresholds[i]
		    threshold = np.quantile(all_ptr_values, prop)

		    genes_higher_than_thresh = len(max_ptr_values[max_ptr_values > threshold])
		    genes_all_lower_than_thresh = len(max_ptr_values[max_ptr_values < threshold])
		    
		    threshold_values.append(threshold)
		    higher_genes.append(genes_higher_than_thresh)
		    lower_genes.append(genes_all_lower_than_thresh)
		    
		    print(f"{threshold:.1f} - percentile: {prop*100:.1f}%")
		    print(f"    {(genes_higher_than_thresh)}/{n} "
		          f"({(genes_higher_than_thresh)/n*100.:.1f}%) "
		          "genes exceed this value for any cell")
		    
		    print(f"    {(genes_all_lower_than_thresh)}/{n} "
		          f"({(genes_all_lower_than_thresh)/n*100.:.1f}%) "
		          "genes are below this value for all cells")
		    
		    print("\n--------------------------------------------------------------------------\n")

		threshold_counts_df = pd.DataFrame({'threshold': threshold_values, 
		                                    'proportion': prop_thresholds,
		                                    'num_higher': higher_genes,
		                                    'num_less': lower_genes})
