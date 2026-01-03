import numpy as np
from typing import Tuple, Optional


class MultiMetricGeneData:
	"""
	Container for gene data with multiple metrics across timepoints.
	Handles conversion between 3D (structured) and 2D (flat) representations.
	
	Structure:
	- 3D: (n_genes, n_metrics, n_timepoints)
	- 2D: (n_genes, n_metrics * n_timepoints) for ML algorithms
	"""
	
	def __init__(self, data_3d: np.ndarray, metric_names: Tuple[str, ...], 
			gene_row_index):
		"""
		Parameters
		----------
		data_3d : np.ndarray
			Shape: (n_genes, n_metrics, n_timepoints)
		metric_names : tuple of str
			Names of metrics, e.g., ('promoter_occupancy', 'nucleosome_occupancy', 
									  'nucleosome_entropy', 'expression')
		"""
		if data_3d.ndim != 3:
			raise ValueError(f"Expected 3D array (genes, metrics, timepoints), got {data_3d.ndim}D")
		
		self.data_3d = data_3d
		self.metric_names = metric_names
		self.n_genes = data_3d.shape[0]
		self.n_metrics = data_3d.shape[1]
		self.n_timepoints = data_3d.shape[2]
		self.gene_row_index = gene_row_index
		self.total_features = self.n_metrics * self.n_timepoints
		
		# Cache the 2D representation
		self._data_2d = None
	
	@property
	def data_2d(self) -> np.ndarray:
		"""Get flattened 2D representation (n_genes, n_metrics * n_timepoints)"""
		if self._data_2d is None:
			self._data_2d = self.data_3d.reshape(self.n_genes, -1)
		return self._data_2d
	

def load_from_chromatin_and_expression_processor(chromatin_processor, expression_processor):

	from src.config import load_default_chrom_configs
	from pipeline.cluster_metrics_data import MultiMetricGeneData
	from src.transcripts_dataset import load_transcripts_sets

	geneset, _ = load_transcripts_sets('output/draft4_run/')

	# Load the metric keys and place in a 4D array
	deconvolved_metrics = chromatin_processor.normalized_deconvolved_metrics

	config1, _ = load_default_chrom_configs()
	metric_keys = list(deconvolved_metrics.keys())

	t_indices = config1.t_indices()
	b_indices = config1.b_indices()

	def _select_mean_tb_data(data_df, config):
		t_data = data_df[config.t_indices()]
		b_data = data_df[config.b_indices()]
		return (t_data.values + b_data.values)/2

	# Convert each metric dataframe to numpy array and stack them
	# Assuming all dataframes have the same shape: (n_genes, n_features)
	metric_arrays = [_select_mean_tb_data(deconvolved_metrics[key].loc[geneset.index], config1) \
		for key in metric_keys]

	# Add expression data
	all_expression_data = expression_processor.expression_data.loc[geneset.index]
	metric_arrays.append(_select_mean_tb_data(all_expression_data, config1))
	metric_keys.append('expression')

	# Stack along a new axis to create 4D array
	# Shape will be: (n_genes, n_features_metric1, n_features_metric2, ...)
	# First, we need to add dimensions if each metric is 2D
	stacked_metrics_data_3d = np.stack(metric_arrays, axis=1)

	data_2d = stacked_metrics_data_3d.reshape((stacked_metrics_data_3d.shape[0], -1))

	# Remove any rows that contain nans
	mask_no_nans = ~np.isnan(data_2d).any(axis=1)
	clean_arr_3d = stacked_metrics_data_3d[mask_no_nans]
	clean_index = geneset.index[mask_no_nans]

	print(f"{len(stacked_metrics_data_3d) - mask_no_nans.sum()} genes with nan values were removed")

	# Create the MultiMetricGeneData object
	gene_data = MultiMetricGeneData(
		data_3d=clean_arr_3d,
		metric_names=tuple(metric_keys),
		gene_row_index = clean_index
	)

	return gene_data
