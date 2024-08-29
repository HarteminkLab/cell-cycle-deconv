import numpy as np
from scipy.signal import correlate
from sklearn.cluster import DBSCAN
from src.timer import Timer
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import pandas as pd


class GeneClustering:
	"""Compute the gene clusters of expression"""

	def __init__(self, outdir):

		from src.Figure3_Chrom_Gene_Expression import Figure3_Chrom_GeneExpression
		figures = Figure3_Chrom_GeneExpression(outdir)
		gene_expression = figures.gene_expression_a.gene_expression_f

		from src.genset import get_deconvolved_geneset
		geneset = get_deconvolved_geneset()

		self.gene_expression = gene_expression
		self.geneset = geneset


	def set_threshold_ptr(self, q_ptr):

		from src.config import load_configs_by_config_type

		config1, config2 = load_configs_by_config_type('shared')
		t_indices = config1.get_Hpositions_for_branch('t')
		gene_expression_top_branch = self.gene_expression[t_indices]

		from src.peak_to_trough import compute_quantile_ptr_2d

		top_gene_expression_ptr = compute_quantile_ptr_2d(gene_expression_top_branch, hi=0.9, lo=0.1)
		top_gene_expression_ptr_df = gene_expression_top_branch[[]].copy()
		top_gene_expression_ptr_df['ptr'] = top_gene_expression_ptr

		# Select a set of genes in the upper 99% percentile
		q_val = np.quantile(top_gene_expression_ptr_df, q_ptr)
		quantile_ptr_genes = top_gene_expression_ptr_df[top_gene_expression_ptr_df.ptr > q_val]
		print(f"Selected {len(quantile_ptr_genes)} genes in the top 80% with quantile value greater"
			  f" than {q_val:.2f}")

		high_ptr_gene_expression = gene_expression_top_branch.loc[quantile_ptr_genes.index]

		plt.hist(top_gene_expression_ptr_df, bins=100)
		plt.yscale('log')
		plt.xlim(0.9, 4)
		plt.axvline(q_val, c='red')
		plt.title("Gene expression, PTR")

		self.config = config1
		self.q_val = q_val
		self.q_ptr = q_ptr
		self.top_gene_expression_ptr_df = top_gene_expression_ptr_df
		self.high_ptr_gene_expression = high_ptr_gene_expression


	def compute_distance_from_gene_expression(self):
		timer = Timer()

		high_ptr_gene_expression = self.high_ptr_gene_expression
		normalized_high_ptr_expression = (high_ptr_gene_expression - \
			high_ptr_gene_expression.mean(axis=1).values.reshape((-1, 1))) /\
			high_ptr_gene_expression.std(axis=1).values.reshape((-1, 1))

		time_series_data = normalized_high_ptr_expression.values

		# Compute the pairwise distance matrix using circular correlation
		n_samples = len(time_series_data)
		distance_matrix = np.zeros((n_samples, n_samples))

		for i in range(n_samples):
			for j in range(i + 1, n_samples):
				distance = circular_distance(time_series_data[i], time_series_data[j])
				distance_matrix[i, j] = distance
				distance_matrix[j, i] = distance
				
			if i % 100 == 0:
				timer.print_time(f"{i}/{n_samples}")

		timer.print_time()

		self.distance_matrix = distance_matrix
		self.normalized_high_ptr_expression = normalized_high_ptr_expression


	def cluster_expression(self, num_clusters):
		self.kmedoids, labels, medoid_indices = cluster_timeseries_kmedoids(self.distance_matrix, num_clusters)
		clustered_expression = self.high_ptr_gene_expression.copy()
		clustered_expression['cluster'] = labels
		self.clustered_expression = clustered_expression.reset_index().set_index(['cluster', 'orf_name'])
		self.labels = labels
		self.medoid_indices = medoid_indices
		self.num_clusters = num_clusters

	def compute_medoids(self):
		medoids = np.zeros((len(self.medoid_indices), \
			self.normalized_high_ptr_expression.shape[1]))
		for i, medoid_index in enumerate(self.medoid_indices):
			medoids[i] = self.normalized_high_ptr_expression.iloc[medoid_index]
		self.medoids = medoids

	def load_chromatin_for_cluster(self, chromatin_dir, cluster):
		from src.deconv_data import load_f_files

		current_cluster_exp = self.clustered_expression.loc[cluster]
		all_gene_f_df = load_f_files(chromatin_dir, current_cluster_exp)
		self.current_cluster_chromatin = all_gene_f_df
		self.current_cluster_exp = current_cluster_exp
		self.selected_cluster = cluster

		# Flip the strand of the loaded chromatin
		current_genes = self.geneset[['gene', 'strand']].loc[self.current_cluster_exp.index]
		current_genes['gene_index'] = np.arange(len(current_genes))

		accumulated_gene_f_imgs = self.current_cluster_chromatin.reshape(
			(-1, 178, 23, 91)).astype(float)
		crick_genes = current_genes[current_genes.strand == '-'].gene_index
		accumulated_gene_f_imgs[crick_genes] = np.flip(accumulated_gene_f_imgs[crick_genes],\
			axis=3)

		self.current_cluster_chromatin = accumulated_gene_f_imgs

	def plot_clustered_heatmap(self):
		clustered_data = self.normalized_high_ptr_expression.copy()
		clustered_data['cluster'] = self.labels
		clustered_data = clustered_data.reset_index().set_index(['cluster', 'orf_name'])
		plt.imshow(clustered_data.sort_index().astype(float))

	def plot_chromatin_in_cluster(self):

		cluster = self.selected_cluster
		aligned_clustered_expression_data, aligned_shift  = align_to_medoid(
			self.current_cluster_exp, self.medoids[cluster])
		print(aligned_shift)

		chrom_dat = self.current_cluster_chromatin
		shifted_chrom_dat = chrom_dat.copy()
		for i in range(shifted_chrom_dat.shape[0]):
			shifted_chrom_dat[i] = np.roll(chrom_dat[i], aligned_shift.values[i], axis=0)

		aligned_chromatin = shifted_chrom_dat.mean(axis=0)

		fig = plt.figure(figsize=(6, 6))

		medoid = self.medoids[cluster]
		interesting_indices = retrieve_important_points(medoid)
		print(interesting_indices)

		k = len(interesting_indices)

		for i, selected_index in enumerate(interesting_indices):
			plt.subplot(k, 1, i+1)
			plt.imshow(aligned_chromatin[selected_index].astype(float),
					  origin='lower', cmap='magma_r', aspect='auto', vmax=10)
			plt.xticks([])


def circular_correlation(v1, v2, return_idx=False):
	"""Compute the circular correlation between two vectors and return the max correlation and the optimal shift."""
	# Ensure the vectors are numpy arrays
	v1 = np.array(v1)
	v2 = np.array(v2)
	
	# Get the length of the vectors
	n = len(v1)
	
	# Compute the cross-correlation
	corr = correlate(v1, v2, mode='full')
	
	# Normalize the correlation
	norm_factor = np.linalg.norm(v1) * np.linalg.norm(v2)
	normalized_corr = corr / norm_factor
	
	# Only consider the relevant part of the cross-correlation (circular part)
	circular_corr = np.concatenate([normalized_corr[-(n-1):], normalized_corr[:n]])
	
	# Find the maximum correlation and its corresponding shift
	max_corr = np.max(circular_corr)
	optimal_shift = np.argmax(circular_corr)
	
	# Since shifts beyond n-1 actually mean shifts in the opposite direction
	if optimal_shift >= n:
		optimal_shift = optimal_shift - n
	
	if return_idx:
		return max_corr, optimal_shift

	return max_corr


def circular_distance(series1, series2):
	"""Convert circular correlation to a distance measure."""
	return 1 - circular_correlation(series1, series2)


def cluster_timeseries_kmedoids(dist_matrix, n_clusters):
	"""
	Perform K-medoids clustering using the precomputed distance matrix.
	"""
	kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', 
						method='alternate', init='k-medoids++')
	kmedoids.fit(dist_matrix)
	return kmedoids, kmedoids.labels_, kmedoids.medoid_indices_


def align_to_medoid(data, medoid):
	aligned_data = []

	data_shift = data[[]].copy()
	data_shift[['shift']] = None
	for idx, series in data.iterrows():  
		corr, shift = circular_correlation(medoid, 
										series, return_idx=True)
		aligned_data.append(np.roll(series, shift))
		data_shift.loc[idx, 'shift'] = shift

	return pd.DataFrame(aligned_data, index=data.index), data_shift

