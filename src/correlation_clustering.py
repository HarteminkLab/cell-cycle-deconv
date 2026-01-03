import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


class CorrelationClustering:
	"""
	Hierarchical clustering using correlation distance for time series data.
	
	Parameters
	----------
	data : np.ndarray
		2D array of shape (n_samples, n_features) where each row is a time series
	"""
	
	def __init__(self, data):
		"""
		Initialize with data matrix.
		
		Parameters
		----------
		data : np.ndarray
			2D array where rows are samples (e.g., genes) and columns are features (e.g., timepoints)
		"""
		self.data = np.array(data)
		if self.data.ndim != 2:
			raise ValueError("Data must be a 2D array")
		
		self.n_samples, self.n_features = self.data.shape
		self.distance_matrix = None
		self.linkage_matrix = None
		self.labels = None
	
	def correlation_distance(self, x, y):
		"""
		Compute correlation distance between two 1D arrays.
		
		Correlation distance = 1 - Pearson correlation coefficient
		
		Parameters
		----------
		x : np.ndarray
			First time series (1D array)
		y : np.ndarray
			Second time series (1D array)
		
		Returns
		-------
		float
			Correlation distance (0 = perfect correlation, 2 = perfect anti-correlation)
		"""
		# Handle edge cases
		if len(x) != len(y):
			raise ValueError("Arrays must have the same length")
		
		# Center the data
		x_centered = x - np.mean(x)
		y_centered = y - np.mean(y)
		
		# Compute correlation
		numerator = np.sum(x_centered * y_centered)
		denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
		
		# Handle zero variance case
		if denominator == 0:
			return 1.0  # Return maximum distance for constant signals
		
		correlation = numerator / denominator
		
		# Clip to handle numerical precision issues
		correlation = np.clip(correlation, -1.0, 1.0)
		
		return 1.0 - correlation
	
	def compute_distance_matrix(self):
		"""
		Compute pairwise correlation distance matrix for all rows in the data.
		
		Returns
		-------
		np.ndarray
			Symmetric distance matrix of shape (n_samples, n_samples)
		"""
		n = self.n_samples
		dist_matrix = np.zeros((n, n))
		
		for i in range(n):
			for j in range(i + 1, n):
				dist = self.correlation_distance(self.data[i], self.data[j])
				dist_matrix[i, j] = dist
				dist_matrix[j, i] = dist
		
		self.distance_matrix = dist_matrix
		return dist_matrix
	
	def cluster(self, n_clusters=None, method='average', criterion='maxclust'):
		"""
		Perform hierarchical clustering on the data using correlation distance.
		
		Parameters
		----------
		n_clusters : int, optional
			Number of clusters to form. If None, only computes linkage without cutting.
		method : str, default='average'
			Linkage method: 'single', 'complete', 'average', 'weighted', 'ward'
			Note: 'ward' is not recommended for correlation distance; use 'average' instead.
		criterion : str, default='maxclust'
			Criterion for forming flat clusters: 'maxclust', 'distance', 'inconsistent', etc.
		
		Returns
		-------
		labels : np.ndarray or None
			Cluster labels for each sample (only if n_clusters is specified)
		linkage_matrix : np.ndarray
			Hierarchical clustering linkage matrix
		"""
		# Compute distance matrix if not already computed
		if self.distance_matrix is None:
			self.compute_distance_matrix()
		
		# Convert to condensed distance matrix for scipy
		condensed_dist = squareform(self.distance_matrix)
		
		# Perform hierarchical clustering
		self.linkage_matrix = linkage(condensed_dist, method=method)
		
		# Cut tree to form flat clusters if n_clusters specified
		if n_clusters is not None:
			self.labels = fcluster(self.linkage_matrix, n_clusters, criterion=criterion)
		else:
			self.labels = None
		
		return self.labels, self.linkage_matrix
	
	def get_cluster_labels(self):
		"""
		Get the cluster labels from the last clustering operation.
		
		Returns
		-------
		np.ndarray or None
			Cluster labels for each sample
		"""
		return self.labels
	
	def get_distance_matrix(self):
		"""
		Get the computed distance matrix.
		
		Returns
		-------
		np.ndarray or None
			Pairwise distance matrix
		"""
		return self.distance_matrix
	
	def get_linkage_matrix(self):
		"""
		Get the linkage matrix from hierarchical clustering.
		
		Returns
		-------
		np.ndarray or None
			Linkage matrix
		"""
		return self.linkage_matrix
	
	def plot_dendrogram(self, figsize=(10, 7), title='Hierarchical Clustering Dendrogram',
					   color_threshold=None, n_clusters=None, above_threshold_color='gray',
					   orientation='left', labels=None, ax=None, **kwargs):
		"""
		Plot dendrogram of hierarchical clustering.
		
		Parameters
		----------
		figsize : tuple, default=(10, 7)
			Figure size (width, height)
		title : str, default='Hierarchical Clustering Dendrogram'
			Plot title
		color_threshold : float, optional
			Threshold for coloring clusters. If None, uses default coloring.
		above_threshold_color : str, default='gray'
			Color for links above the threshold
		orientation : str, default='top'
			Dendrogram orientation: 'top', 'bottom', 'left', 'right'
		labels : array-like, optional
			Labels for leaf nodes (e.g., gene names). If None, uses indices.
		**kwargs : dict
			Additional keyword arguments passed to scipy.cluster.hierarchy.dendrogram
		
		Returns
		-------
		fig : matplotlib.figure.Figure
			The figure object
		ax : matplotlib.axes.Axes
			The axes object
		dendro_dict : dict
			Dictionary containing dendrogram information including 'leaves' (the ordering)
		"""
		if self.linkage_matrix is None:
			raise ValueError("Must run cluster() first to compute linkage matrix")

		# If n_clusters is specified, compute the appropriate color threshold
		if n_clusters is not None:
			
			# The linkage matrix has merges in order
			# To get n_clusters, we cut just above the (n_samples - n_clusters)th merge
			merge_index = self.n_samples - n_clusters
			if merge_index >= 0 and merge_index < len(self.linkage_matrix):
				# Set threshold slightly above this merge distance
				color_threshold = self.linkage_matrix[merge_index, 2] + 1e-10
			else:
				color_threshold = None	

		if ax is None:
			fig, ax = plt.subplots(figsize=figsize)

		# Plot dendrogram
		dendro_dict = dendrogram(
			self.linkage_matrix,
			ax=ax,
			color_threshold=color_threshold,
			above_threshold_color=above_threshold_color,
			orientation=orientation,
			labels=labels,
			**kwargs
		)

		n = len(self.linkage_matrix)

		# Remove x-axis tick labels if requested
		ax.set_yticklabels([])
		ax.set_ylabel('')
		ax.set_ylim(n, 0)
		
		ax.set_title(title, fontsize=14, fontweight='bold')
		ax.set_xlabel('Correlation Distance', fontsize=12)

		self.dendrogram = dendrogram
		self.dendro_dict = dendro_dict
	
	def get_dendrogram_order(self):
		"""
		Get the ordering of samples as they appear in the dendrogram (left to right).
		This is useful for reordering your data matrix to match the dendrogram.
		
		Returns
		-------
		np.ndarray
			Array of indices representing the dendrogram leaf order
		
		Notes
		-----
		After calling this, you can reorder your data as: data_sorted = data[order, :]
		"""
		# Extract leaf order (this is the ordering from left to right in dendrogram)
		leaf_order = np.array(self.dendro_dict['leaves'])
		
		return leaf_order

