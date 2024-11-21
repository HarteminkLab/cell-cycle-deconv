
from src.gene_clustering import circular_distance
from scipy.stats import pearsonr
import numpy as np
from matplotlib import pyplot as plt


def compute_distance_from_tf_occupancy(time_series_data):

	# Compute the pairwise distance matrix using circular correlation
	n_samples = len(time_series_data)
	distance_matrix = np.zeros((n_samples, n_samples))

	for i in range(n_samples):
		for j in range(i + 1, n_samples):
			pearsonr_val, _ = pearsonr(time_series_data[i], time_series_data[j])
			distance = 1 - pearsonr_val
			distance_matrix[i, j] = distance
			distance_matrix[j, i] = distance

	return distance_matrix


class TFClustering():

	def __init__(self):
		pass

	def compute_tf_distances(self, analysis):

		from src.config import load_configs_by_config_type

		config, _ = load_configs_by_config_type('shared')
		t_indices = config.get_Hpositions_for_branch('t')

		# Distance matrix
		tf_occ_t = analysis.all_tf_occupancies[t_indices]
		from src.cluster_tf_sites import compute_distance_from_tf_occupancy

		tf_dist = compute_distance_from_tf_occupancy(tf_occ_t.values)
		original_tf_dist = tf_dist.copy()
		nonna_indices = np.arange(len(tf_dist))[~np.isnan(tf_dist[0])]
		nonna_dist = tf_dist[nonna_indices, :][:, nonna_indices]

		self.tf_occ_t = tf_occ_t
		self.original_tf_dist = tf_dist
		self.nonna_dist = nonna_dist
		self.nonna_indices = nonna_indices


	def cluster_tfs(self, n_clusters, filtered_tf_sites):

		from sklearn_extra.cluster import KMedoids

		nonna_tf_occ_t = self.tf_occ_t.loc[self.nonna_indices]
		kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', 
							method='alternate', init='k-medoids++', random_state=123)
		kmedoids.fit(self.nonna_dist.astype(float))

		nonnatf_cluster_assignments = nonna_tf_occ_t[[]].copy()
		nonnatf_cluster_assignments['cluster'] = kmedoids.labels_
		nonnatf_cluster_assignments = nonnatf_cluster_assignments.join(filtered_tf_sites, how='left')
		sorted_nonnatf_indices = nonnatf_cluster_assignments.sort_values('cluster').index

		self.sorted_nona_dist = self.nonna_dist[sorted_nonnatf_indices, :][:, sorted_nonnatf_indices]
		self.nonnatf_cluster_assignments = nonnatf_cluster_assignments
		self.nonna_tf_occ_t = nonna_tf_occ_t
		self.sorted_nonnatf_indices = sorted_nonnatf_indices

	def plot_distance_mat(self):
		plt.figure(figsize=(8, 3))
		plt.subplot(1, 2, 1)
		plt.imshow(self.nonna_dist)
		plt.colorbar()

		plt.subplot(1, 2, 2)
		plt.imshow(self.sorted_nona_dist)
		plt.colorbar()

	def plot_clusters(self):
		nonna_tf_occ_t = self.nonna_tf_occ_t
		normalized_cc = (nonna_tf_occ_t - nonna_tf_occ_t.mean(axis=1).values.reshape((-1, 1))) / \
		    nonna_tf_occ_t.std(axis=1).values.reshape((-1, 1))

		plt.figure(figsize=(8,6))

		plt.subplot(1, 2, 1)
		plt.imshow(nonna_tf_occ_t.loc[self.sorted_nonnatf_indices], 
		    aspect='auto', interpolation='none', vmax=150)
		plt.colorbar()

		plt.subplot(1, 2, 2)
		plt.imshow(normalized_cc.loc[self.sorted_nonnatf_indices], 
		    aspect='auto', interpolation='none')

	def plot_cluster_heatmaps(self, analysis):
		padding = 400
		nonnatf_cluster_assignments = self.nonnatf_cluster_assignments

		for cluster in range(9):
		    print("Cluster", cluster)
		    cluster_sites = nonnatf_cluster_assignments[nonnatf_cluster_assignments.cluster == cluster]
		    analysis.load_stacked_mnase_data_for_rossi_sites(cluster_sites, padding=padding)
		    fig = analysis.plot_deconvolved_result(analysis.all_tfs_mnase, 
		    mnase_span=(-padding, padding), figsize=(3, 6), vmax=10)
		    plt.suptitle(f"Cluster {cluster}, n={len(cluster_sites)}")
