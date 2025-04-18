import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class ThreeDimensionalPTRAnalysis:
	"""
	A class for performing three-dimensional sensitivity analysis of gene expression
	and two chromatin PTR thresholds.
	"""
	
	def __init__(self, gene_expression_PTR, chromatin_metric1_PTR, chromatin_metric2_PTR, 
				 metric1_name="Nucleosome Entropy", metric2_name="Promoter Occupancy", 
				 gene_names=None):
		"""
		Initialize the 3D PTR analysis with three metrics.
		
		Parameters:
		-----------
		gene_expression_PTR : array-like
			PTR values for gene expression
		chromatin_metric1_PTR : array-like
			PTR values for first chromatin metric
		chromatin_metric2_PTR : array-like
			PTR values for second chromatin metric
		metric1_name : str
			Name of the first chromatin metric
		metric2_name : str
			Name of the second chromatin metric
		gene_names : list, optional
			List of gene names corresponding to the PTR values
		"""
		# Store metric names
		self.metric1_name = metric1_name
		self.metric2_name = metric2_name
		
		# Convert pandas Series to numpy arrays if needed
		self.gene_names = None
		if isinstance(gene_expression_PTR, pd.Series):
			if gene_names is None:
				self.gene_names = gene_expression_PTR.index.tolist()
			self.gene_expression_PTR = gene_expression_PTR.values
		else:
			self.gene_expression_PTR = gene_expression_PTR
			
		if isinstance(chromatin_metric1_PTR, pd.Series):
			if gene_names is None and self.gene_names is None:
				self.gene_names = chromatin_metric1_PTR.index.tolist()
			self.chromatin_metric1_PTR = chromatin_metric1_PTR.values
		else:
			self.chromatin_metric1_PTR = chromatin_metric1_PTR
			
		if isinstance(chromatin_metric2_PTR, pd.Series):
			if gene_names is None and self.gene_names is None:
				self.gene_names = chromatin_metric2_PTR.index.tolist()
			self.chromatin_metric2_PTR = chromatin_metric2_PTR.values
		else:
			self.chromatin_metric2_PTR = chromatin_metric2_PTR
		
		# Use passed gene_names if provided
		if gene_names is not None:
			self.gene_names = gene_names
			
		# If no gene names are available, create placeholder names
		if self.gene_names is None:
			self.gene_names = [f"Gene_{i}" for i in range(len(self.gene_expression_PTR))]
			
		# Check that we have the correct number of gene names
		if len(self.gene_names) != len(self.gene_expression_PTR):
			raise ValueError("Number of gene names must match the number of gene expression values")
			
		self.total_genes = len(self.gene_expression_PTR)
		self.results_df = None
		self.volume_data = None
		self.gene_sets = {}
		
	def run_3d_analysis(self, expr_percentiles=np.arange(0.5, 1.01, 0.1), 
					   chrom1_percentiles=np.arange(0.5, 1.01, 0.1),
					   chrom2_percentiles=np.arange(0.5, 1.01, 0.1)):
		"""
		Run a three-dimensional analysis across various expression and two chromatin thresholds.
		
		Parameters:
		-----------
		expr_percentiles : array-like
			Percentile thresholds for gene expression
		chrom1_percentiles : array-like
			Percentile thresholds for first chromatin metric
		chrom2_percentiles : array-like
			Percentile thresholds for second chromatin metric
			
		Returns:
		--------
		pandas.DataFrame
			DataFrame with analysis results
		"""
		# Sort percentiles to ensure they're in ascending order
		expr_percentiles = np.sort(expr_percentiles)
		chrom1_percentiles = np.sort(chrom1_percentiles)
		chrom2_percentiles = np.sort(chrom2_percentiles)
		
		# Initialize results container
		results = []
		
		# Initialize data structures for volumetric visualization
		p_values = np.zeros((len(expr_percentiles), len(chrom1_percentiles), len(chrom2_percentiles)))
		log10_p_values = np.zeros((len(expr_percentiles), len(chrom1_percentiles), len(chrom2_percentiles)))
		enrichment_ratios = np.zeros((len(expr_percentiles), len(chrom1_percentiles), len(chrom2_percentiles)))
		intersection_sizes = np.zeros((len(expr_percentiles), len(chrom1_percentiles), len(chrom2_percentiles)))
		
		# Store gene sets for key threshold combinations
		self.gene_sets = {}
		
		# Loop through all combinations of thresholds
		for i, expr_percentile in enumerate(expr_percentiles):
			expr_threshold = np.percentile(self.gene_expression_PTR, expr_percentile * 100)
			expr_genes = np.where(self.gene_expression_PTR >= expr_threshold)[0]
			expr_count = len(expr_genes)
			
			for j, chrom1_percentile in enumerate(chrom1_percentiles):
				chrom1_threshold = np.percentile(self.chromatin_metric1_PTR, chrom1_percentile * 100)
				chrom1_genes = np.where(self.chromatin_metric1_PTR >= chrom1_threshold)[0]
				chrom1_count = len(chrom1_genes)
				
				# Find intersection of first two sets
				intersection_1_2 = np.intersect1d(expr_genes, chrom1_genes)
				intersection_1_2_count = len(intersection_1_2)
				
				for k, chrom2_percentile in enumerate(chrom2_percentiles):
					chrom2_threshold = np.percentile(self.chromatin_metric2_PTR, chrom2_percentile * 100)
					chrom2_genes = np.where(self.chromatin_metric2_PTR >= chrom2_threshold)[0]
					chrom2_count = len(chrom2_genes)
					
					# Find intersection of all three sets
					intersection_all = np.intersect1d(intersection_1_2, chrom2_genes)
					intersection_count = len(intersection_all)
					
					# Expected overlap under null hypothesis (assuming independence)
					expected = (expr_count * chrom1_count * chrom2_count) / (self.total_genes ** 2)
					
					# Calculate significance - using hypergeometric approximation
					# This is an approximation, as the true 3-way intersection test is more complex
					# We'll use the intersection of first two sets as our "new population"
					if intersection_1_2_count > 0 and chrom2_count > 0:
						p_value = 1 - stats.hypergeom.cdf(
							intersection_count-1, 
							self.total_genes, 
							intersection_1_2_count, 
							chrom2_count
						)
					else:
						p_value = 1.0
					
					# Store results
					results.append({
						'expr_percentile': expr_percentile,
						'chrom1_percentile': chrom1_percentile,
						'chrom2_percentile': chrom2_percentile,
						'expr_threshold': expr_threshold,
						'chrom1_threshold': chrom1_threshold,
						'chrom2_threshold': chrom2_threshold,
						'expr_count': expr_count,
						'chrom1_count': chrom1_count,
						'chrom2_count': chrom2_count,
						'intersection_1_2': intersection_1_2_count,
						'intersection_all': intersection_count,
						'expected_overlap': expected,
						'enrichment_ratio': intersection_count / expected if expected > 0 else np.inf,
						'p_value': p_value,
						'-log10_p_value': -np.log10(p_value) if p_value > 0 else np.inf
					})
					
					# Store values for volumetric visualization
					p_values[i, j, k] = p_value
					log10_p_values[i, j, k] = -np.log10(p_value) if p_value > 0 else np.inf
					enrichment_ratios[i, j, k] = intersection_count / expected if expected > 0 else np.inf
					intersection_sizes[i, j, k] = intersection_count
					
					# Store gene sets for key combinations
					key = (expr_percentile, chrom1_percentile, chrom2_percentile)
					
					# Get genes unique to each set and their intersections
					self.gene_sets[key] = {
						'all_three': [self.gene_names[idx] for idx in intersection_all],
						'expr_chrom1_only': [self.gene_names[idx] for idx in np.setdiff1d(intersection_1_2, intersection_all)],
						'expr_chrom2_only': [self.gene_names[idx] for idx in np.setdiff1d(np.intersect1d(expr_genes, chrom2_genes), intersection_all)],
						'chrom1_chrom2_only': [self.gene_names[idx] for idx in np.setdiff1d(np.intersect1d(chrom1_genes, chrom2_genes), intersection_all)],
						'expr_only': [self.gene_names[idx] for idx in np.setdiff1d(expr_genes, np.union1d(chrom1_genes, chrom2_genes))],
						'chrom1_only': [self.gene_names[idx] for idx in np.setdiff1d(chrom1_genes, np.union1d(expr_genes, chrom2_genes))],
						'chrom2_only': [self.gene_names[idx] for idx in np.setdiff1d(chrom2_genes, np.union1d(expr_genes, chrom1_genes))]
					}
		
		# Convert to DataFrame
		self.results_df = pd.DataFrame(results)
		
		# Store volume data for visualization
		self.volume_data = {
			'expr_percentiles': expr_percentiles,
			'chrom1_percentiles': chrom1_percentiles,
			'chrom2_percentiles': chrom2_percentiles,
			'p_values': p_values,
			'log10_p_values': log10_p_values,
			'enrichment_ratios': enrichment_ratios,
			'intersection_sizes': intersection_sizes
		}
		
		return self.results_df
	
	def visualize_3d_data(self, metric='log10_p_value', threshold=3.0, 
						 colormap='viridis', alpha=0.7, figsize=(12, 10)):
		"""
		Visualize the 3D results using volumetric slicing and isosurfaces.
		
		Parameters:
		-----------
		metric : str
			Metric to visualize ('log10_p_value', 'enrichment_ratio', 'intersection_size')
		threshold : float
			Threshold value for isosurface
		colormap : str
			Colormap for visualization
		alpha : float
			Transparency level
		figsize : tuple
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			Figure with the visualizations
		"""
		if self.volume_data is None:
			raise ValueError("No volume data available. Run run_3d_analysis() first.")
			
		# Get data to visualize
		if metric == 'log10_p_value':
			data = self.volume_data['log10_p_values']
			title = '-log10(p-value) of Intersection'
		elif metric == 'enrichment_ratio':
			data = np.clip(self.volume_data['enrichment_ratios'], 0, 10)  # Cap extremely high values
			title = 'Enrichment Ratio (Observed/Expected)'
		elif metric == 'intersection_size':
			data = self.volume_data['intersection_sizes']
			title = 'Number of Genes in Intersection'
		else:
			raise ValueError("Invalid metric. Choose 'log10_p_value', 'enrichment_ratio', or 'intersection_size'")
			
		# Create figure
		fig = plt.figure(figsize=figsize)
		
		# 3D plots are best done with multiple views
		# 1. Orthogonal slices at midpoints
		ax1 = fig.add_subplot(221, projection='3d')
		self._plot_orthogonal_slices(ax1, data, threshold, colormap, alpha)
		ax1.set_title(f"Orthogonal Slices: {title}")
		
		# 2. Isosurface
		ax2 = fig.add_subplot(222, projection='3d')
		self._plot_isosurface(ax2, data, threshold, colormap, alpha)
		ax2.set_title(f"Isosurface (threshold={threshold}): {title}")
		
		# 3. 2D slices along each axis for complementary view
		ax3 = fig.add_subplot(223)
		self._plot_2d_slices(ax3, data, axis=0, colormap=colormap)
		ax3.set_title(f"2D Slices along Gene Expression Axis")
		
		# 4. 3D scatter plot of most significant points
		ax4 = fig.add_subplot(224, projection='3d')
		self._plot_significant_points(ax4, data, threshold, colormap)
		ax4.set_title(f"Significant Points (>{threshold})")
		
		plt.tight_layout()
		return fig
	
	def _plot_orthogonal_slices(self, ax, data, threshold, colormap, alpha):
		"""Plot orthogonal slices through the volume at midpoints"""
		x, y, z = np.indices(data.shape)
		
		# Get midpoints
		midx = data.shape[0] // 2
		midy = data.shape[1] // 2
		midz = data.shape[2] // 2
		
		# Create coordinate matrices for plotting
		X = self.volume_data['expr_percentiles']
		Y = self.volume_data['chrom1_percentiles']
		Z = self.volume_data['chrom2_percentiles']
		
		# Create meshgrids for plotting
		Xm, Ym = np.meshgrid(X, Y)
		Xm, Zm = np.meshgrid(X, Z)
		Ym, Zm = np.meshgrid(Y, Z)
		
		# Plot slices
		ax.contourf(Xm, Ym, data[:, :, midz], zdir='z', offset=Z[midz], 
					cmap=colormap, alpha=alpha, levels=20)
		ax.contourf(Xm, Zm, data[:, midy, :], zdir='y', offset=Y[midy], 
					cmap=colormap, alpha=alpha, levels=20)
		ax.contourf(Ym, Zm, data[midx, :, :], zdir='x', offset=X[midx], 
					cmap=colormap, alpha=alpha, levels=20)
		
		# Set axis labels and limits
		ax.set_xlabel('Gene Expression')
		ax.set_ylabel(f'{self.metric1_name}')
		ax.set_zlabel(f'{self.metric2_name}')
		ax.set_xlim(X.min(), X.max())
		ax.set_ylim(Y.min(), Y.max())
		ax.set_zlim(Z.min(), Z.max())
	
	def _plot_isosurface(self, ax, data, threshold, colormap, alpha):
		"""Plot isosurface at the specified threshold"""
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection
		from skimage import measure
		
		# Get coordinate matrices
		X = self.volume_data['expr_percentiles']
		Y = self.volume_data['chrom1_percentiles']
		Z = self.volume_data['chrom2_percentiles']
		
		# Create coordinate arrays
		x, y, z = np.indices(data.shape)
		x = x * (X.max() - X.min()) / (data.shape[0] - 1) + X.min()
		y = y * (Y.max() - Y.min()) / (data.shape[1] - 1) + Y.min()
		z = z * (Z.max() - Z.min()) / (data.shape[2] - 1) + Z.min()
		
		# Extract isosurface using scikit-image
		try:
			verts, faces, _, _ = measure.marching_cubes(data, threshold)
			
			# Convert vertices to actual coordinates
			verts[:, 0] = verts[:, 0] * (X.max() - X.min()) / (data.shape[0] - 1) + X.min()
			verts[:, 1] = verts[:, 1] * (Y.max() - Y.min()) / (data.shape[1] - 1) + Y.min()
			verts[:, 2] = verts[:, 2] * (Z.max() - Z.min()) / (data.shape[2] - 1) + Z.min()
			
			# Create mesh
			mesh = Poly3DCollection(verts[faces], alpha=alpha)
			face_color = plt.cm.get_cmap(colormap)(0.5)
			mesh.set_facecolor(face_color)
			mesh.set_edgecolor('k')
			ax.add_collection3d(mesh)
		except:
			# If marching cubes fails (e.g., no isosurface found), plot scatter of points above threshold
			self._plot_significant_points(ax, data, threshold, colormap)
		
		# Set axis labels and limits
		ax.set_xlabel('Gene Expression')
		ax.set_ylabel(f'{self.metric1_name}')
		ax.set_zlabel(f'{self.metric2_name}')
		ax.set_xlim(X.min(), X.max())
		ax.set_ylim(Y.min(), Y.max())
		ax.set_zlim(Z.min(), Z.max())
	
	def _plot_2d_slices(self, ax, data, axis=0, colormap='viridis'):
		"""Plot heat map slices along a specified axis"""
		# Get percentile values
		X = self.volume_data['expr_percentiles']
		Y = self.volume_data['chrom1_percentiles']
		Z = self.volume_data['chrom2_percentiles']
		
		# Get the middle slice along axis
		if axis == 0:  # Expression axis
			slice_idx = data.shape[0] // 2
			slice_data = data[slice_idx, :, :]
			x_label = f'{self.metric1_name}'
			y_label = f'{self.metric2_name}'
			x_vals = Y
			y_vals = Z
		elif axis == 1:  # Chrom1 axis
			slice_idx = data.shape[1] // 2
			slice_data = data[:, slice_idx, :]
			x_label = 'Gene Expression'
			y_label = f'{self.metric2_name}'
			x_vals = X
			y_vals = Z
		else:  # Chrom2 axis
			slice_idx = data.shape[2] // 2
			slice_data = data[:, :, slice_idx]
			x_label = 'Gene Expression'
			y_label = f'{self.metric1_name}'
			x_vals = X
			y_vals = Y
		
		# Create meshgrid
		X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
		
		# Plot heat map
		im = ax.pcolormesh(X_grid, Y_grid, slice_data.T, cmap=colormap)
		plt.colorbar(im, ax=ax)
		
		# Set labels
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
	
	def _plot_significant_points(self, ax, data, threshold, colormap):
		"""Plot significant points as a 3D scatter plot"""
		# Get coordinate matrices
		X = self.volume_data['expr_percentiles']
		Y = self.volume_data['chrom1_percentiles']
		Z = self.volume_data['chrom2_percentiles']
		
		# Find significant points
		x_idx, y_idx, z_idx = np.where(data > threshold)
		
		if len(x_idx) > 0:
			# Convert indices to percentile values
			x_vals = X[x_idx]
			y_vals = Y[y_idx]
			z_vals = Z[z_idx]
			data_vals = data[x_idx, y_idx, z_idx]
			
			# Normalize data for coloring
			norm = plt.Normalize(threshold, data_vals.max())
			colors = plt.cm.get_cmap(colormap)(norm(data_vals))
			
			# Plot 3D scatter
			scatter = ax.scatter(x_vals, y_vals, z_vals, c=colors, s=100*norm(data_vals), alpha=0.7)
			
			# Add colorbar
			cbar = plt.colorbar(scatter, ax=ax)
			cbar.set_label('Significance')
		else:
			ax.text(0.5, 0.5, 0.5, "No points above threshold", 
					ha='center', va='center', transform=ax.transAxes)
		
		# Set axis labels and limits
		ax.set_xlabel('Gene Expression')
		ax.set_ylabel(f'{self.metric1_name}')
		ax.set_zlabel(f'{self.metric2_name}')
		ax.set_xlim(X.min(), X.max())
		ax.set_ylim(Y.min(), Y.max())
		ax.set_zlim(Z.min(), Z.max())
	
	def find_optimal_thresholds(self, criteria='p_value', top_n=5):
		"""
		Find optimal threshold combinations based on a given criteria.
		
		Parameters:
		-----------
		criteria : str
			Criteria to optimize ('p_value', 'enrichment_ratio', or 'intersection_size')
		top_n : int
			Number of top combinations to return
			
		Returns:
		--------
		pandas.DataFrame
			DataFrame with top threshold combinations
		"""
		if self.results_df is None:
			raise ValueError("No results available. Run run_3d_analysis() first.")
			
		# Define sorting key based on criteria
		if criteria == 'p_value':
			sort_key = 'p_value'
			ascending = True
		elif criteria == 'enrichment_ratio':
			sort_key = 'enrichment_ratio'
			ascending = False
		elif criteria == 'intersection_size':
			sort_key = 'intersection_size'
			ascending = False
		else:
			raise ValueError("Invalid criteria. Choose 'p_value', 'enrichment_ratio', or 'intersection_size'")
			
		# Sort and return top combinations
		top_combinations = self.results_df.sort_values(by=sort_key, ascending=ascending).head(top_n)
		return top_combinations
	
	def visualize_optimal_regions(self, criteria='p_value', top_n=3, figsize=(15, 5)):
		"""
		Visualize the most significant regions with 2D projections.
		
		Parameters:
		-----------
		criteria : str
			Criteria to optimize ('p_value', 'enrichment_ratio', or 'intersection_size')
		top_n : int
			Number of top combinations to highlight
		figsize : tuple
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			Figure with the visualizations
		"""
		# Find optimal combinations
		top_combinations = self.find_optimal_thresholds(criteria, top_n)
		
		# Create figure with 3 subplots (one for each 2D projection)
		fig, axes = plt.subplots(1, 3, figsize=figsize)
		
		# Get data to visualize
		if criteria == 'p_value':
			data_3d = self.volume_data['log10_p_values']
			title = '-log10(p-value)'
		elif criteria == 'enrichment_ratio':
			data_3d = np.clip(self.volume_data['enrichment_ratios'], 0, 10)
			title = 'Enrichment Ratio'
		else:
			data_3d = self.volume_data['intersection_sizes']
			title = 'Intersection Size'
		
		# Get dimension values
		X = self.volume_data['expr_percentiles']
		Y = self.volume_data['chrom1_percentiles']
		Z = self.volume_data['chrom2_percentiles']
		
		# Plot each 2D projection
		# 1. Expression vs Chrom1 (max projection along Chrom2)
		ax = axes[0]
		projection = np.max(data_3d, axis=2)
		im = ax.pcolormesh(X, Y, projection.T, cmap='viridis')
		plt.colorbar(im, ax=ax)
		ax.set_xlabel('Gene Expression')
		ax.set_ylabel(self.metric1_name)
		ax.set_title(f"{title}: Expression vs {self.metric1_name}")
		
		# Mark top combinations on this projection
		for _, row in top_combinations.iterrows():
			ax.plot(row['expr_percentile'], row['chrom1_percentile'], 'ro', markersize=10)
		
		# 2. Expression vs Chrom2 (max projection along Chrom1)
		ax = axes[1]
		projection = np.max(data_3d, axis=1)
		im = ax.pcolormesh(X, Z, projection.T, cmap='viridis')
		plt.colorbar(im, ax=ax)
		ax.set_xlabel('Gene Expression')
		ax.set_ylabel(self.metric2_name)
		ax.set_title(f"{title}: Expression vs {self.metric2_name}")
		
		# Mark top combinations on this projection
		for _, row in top_combinations.iterrows():
			ax.plot(row['expr_percentile'], row['chrom2_percentile'], 'ro', markersize=10)
		
		# 3. Chrom1 vs Chrom2 (max projection along Expression)
		ax = axes[2]
		projection = np.max(data_3d, axis=0)
		im = ax.pcolormesh(Y, Z, projection.T, cmap='viridis')
		plt.colorbar(im, ax=ax)
		ax.set_xlabel(self.metric1_name)
		ax.set_ylabel(self.metric2_name)
		ax.set_title(f"{title}: {self.metric1_name} vs {self.metric2_name}")
		
		# Mark top combinations on this projection
		for _, row in top_combinations.iterrows():
			ax.plot(row['chrom1_percentile'], row['chrom2_percentile'], 'ro', markersize=10)
		
		plt.tight_layout()
		return fig
	
	def analyze_gene_sets(self, expr_percentile, chrom1_percentile, chrom2_percentile):
		"""
		Analyze gene sets at a specific threshold combination.
		
		Parameters:
		-----------
		expr_percentile : float
			Percentile threshold for gene expression
		chrom1_percentile : float
			Percentile threshold for first chromatin metric
		chrom2_percentile : float
			Percentile threshold for second chromatin metric
			
		Returns:
		--------
		dict
			Dictionary with gene set analysis
		"""
		key = (expr_percentile, chrom1_percentile, chrom2_percentile)
		
		if key not in self.gene_sets:
			raise ValueError("Gene sets not found for these thresholds. Try run_3d_analysis first.")
		
		gene_sets = self.gene_sets[key]
		
		# Count genes in each category
		set_counts = {category: len(genes) for category, genes in gene_sets.items()}
		
		# Venn diagram data structure
		venn_data = {
			'all_three': set_counts['all_three'],
			'expr_chrom1_only': set_counts['expr_chrom1_only'],
			'expr_chrom2_only': set_counts['expr_chrom2_only'],
			'chrom1_chrom2_only': set_counts['chrom1_chrom2_only'],
			'expr_only': set_counts['expr_only'],
			'chrom1_only': set_counts['chrom1_only'],
			'chrom2_only': set_counts['chrom2_only']
		}
		
		# Create summary
		summary = {
			'gene_sets': gene_sets,
			'set_counts': set_counts,
			'venn_data': venn_data
		}
		
		return summary
	
	def visualize_venn_diagram(self, expr_percentile, chrom1_percentile, chrom2_percentile, figsize=(10, 8)):
		"""
		Visualize the gene set overlaps as a Venn diagram.
		
		Parameters:
		-----------
		expr_percentile : float
			Percentile threshold for gene expression
		chrom1_percentile : float
			Percentile threshold for first chromatin metric
		chrom2_percentile : float
			Percentile threshold for second chromatin metric
		figsize : tuple
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			Figure with the Venn diagram
		"""
		from matplotlib_venn import venn3, venn3_circles
		
		# Get gene set analysis
		analysis = self.analyze_gene_sets(expr_percentile, chrom1_percentile, chrom2_percentile)
		
		# Extract counts for Venn diagram
		# Order: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
		venn_counts = (
			analysis['set_counts']['expr_only'],
			analysis['set_counts']['chrom1_only'],
			analysis['set_counts']['expr_chrom1_only'],
			analysis['set_counts']['chrom2_only'],
			analysis['set_counts']['expr_chrom2_only'],
			analysis['set_counts']['chrom1_chrom2_only'],
			analysis['set_counts']['all_three']
		)
		
		# Create figure
		fig, ax = plt.subplots(figsize=figsize)
		
		# Create Venn diagram
		v = venn3(subsets=venn_counts, set_labels=('Gene Expression', self.metric1_name, self.metric2_name), ax=ax)
		venn3_circles(subsets=venn_counts, ax=ax, linewidth=1)
		
		# Add title with threshold information

# Add title with threshold information
		title = (f"Gene Set Distribution at Thresholds:\n"
				f"Expression: {expr_percentile:.2f}, "
				f"{self.metric1_name}: {chrom1_percentile:.2f}, "
				f"{self.metric2_name}: {chrom2_percentile:.2f}")
		plt.title(title)
		
		return fig
	
	def compare_multiple_thresholds(self, threshold_combinations, figsize=(15, 15)):
		"""
		Compare gene sets across different threshold combinations.
		
		Parameters:
		-----------
		threshold_combinations : list of tuples
			List of (expr, chrom1, chrom2) threshold combinations to compare
		figsize : tuple
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			Figure with the comparison visualizations
		"""
		# Ensure we have valid combinations
		if len(threshold_combinations) < 2:
			raise ValueError("Need at least two combinations to compare")
		
		# Analyze each combination
		analyses = {}
		for combo in threshold_combinations:
			expr, chrom1, chrom2 = combo
			analyses[combo] = self.analyze_gene_sets(expr, chrom1, chrom2)
		
		# Create a figure with subplots
		n_combos = len(threshold_combinations)
		n_cols = min(3, n_combos)
		n_rows = (n_combos + n_cols - 1) // n_cols
		
		fig = plt.figure(figsize=figsize)
		
		# 1. Plot Venn diagrams for each combination
		for i, combo in enumerate(threshold_combinations):
			ax = fig.add_subplot(n_rows, n_cols, i+1)
			expr, chrom1, chrom2 = combo
			analysis = analyses[combo]
			
			# Create mini Venn diagram
			from matplotlib_venn import venn3, venn3_circles
			
			# Extract counts for Venn diagram (Abc, aBc, ABc, abC, AbC, aBC, ABC)
			venn_counts = (
				analysis['set_counts']['expr_only'],
				analysis['set_counts']['chrom1_only'],
				analysis['set_counts']['expr_chrom1_only'],
				analysis['set_counts']['chrom2_only'],
				analysis['set_counts']['expr_chrom2_only'],
				analysis['set_counts']['chrom1_chrom2_only'],
				analysis['set_counts']['all_three']
			)
			
			v = venn3(subsets=venn_counts, set_labels=('Expr', 'Chr1', 'Chr2'), ax=ax)
			venn3_circles(subsets=venn_counts, ax=ax, linewidth=1)
			
			# Add title
			ax.set_title(f"Threshold: ({expr:.2f}, {chrom1:.2f}, {chrom2:.2f})")
		
		# 2. Plot set sizes comparison
		ax = fig.add_subplot(n_rows+1, 1, n_rows+1)
		
		# Prepare data for bar chart
		set_types = ['all_three', 'expr_chrom1_only', 'expr_chrom2_only', 
					'chrom1_chrom2_only', 'expr_only', 'chrom1_only', 'chrom2_only']
		
		labels = ['All Three', 'Expr & Chr1', 'Expr & Chr2', 'Chr1 & Chr2', 
				 'Expr Only', 'Chr1 Only', 'Chr2 Only']
		
		# Position of bars
		x = np.arange(len(labels))
		width = 0.8 / n_combos
		
		# Plot bars for each combination
		for i, combo in enumerate(threshold_combinations):
			counts = [analyses[combo]['set_counts'][t] for t in set_types]
			offset = width * i - width * (n_combos - 1) / 2
			ax.bar(x + offset, counts, width, label=f"Threshold {i+1}")
		
		# Add labels and legend
		ax.set_ylabel('Number of Genes')
		ax.set_title('Comparison of Gene Set Sizes Across Thresholds')
		ax.set_xticks(x)
		ax.set_xticklabels(labels, rotation=45, ha='right')
		ax.legend()
		
		plt.tight_layout()
		return fig
	
	def identify_core_genes(self, threshold_combinations=None, min_occurrence=2):
		"""
		Identify core genes that appear in multiple threshold combinations.
		
		Parameters:
		-----------
		threshold_combinations : list of tuples, optional
			List of (expr, chrom1, chrom2) threshold combinations to analyze
			If None, use all calculated combinations
		min_occurrence : int
			Minimum number of combinations a gene must appear in
			
		Returns:
		--------
		dict
			Dictionary with core gene analysis
		"""
		# If no combinations provided, use optimal combinations
		if threshold_combinations is None:
			# Get top combinations by p-value
			top_df = self.find_optimal_thresholds(criteria='p_value', top_n=5)
			threshold_combinations = [
				(row['expr_percentile'], row['chrom1_percentile'], row['chrom2_percentile']) 
				for _, row in top_df.iterrows()
			]
		
		# Track gene occurrence
		gene_occurrence = {}
		
		# Loop through combinations
		for combo in threshold_combinations:
			# Skip if this combination doesn't exist in gene_sets
			if combo not in self.gene_sets:
				continue
				
			# Get genes in the three-way intersection
			intersection_genes = self.gene_sets[combo]['all_three']
			
			# Record occurrence
			for gene in intersection_genes:
				if gene not in gene_occurrence:
					gene_occurrence[gene] = []
				gene_occurrence[gene].append(combo)
		
		# Filter for core genes
		core_genes = {gene: occurrences for gene, occurrences in gene_occurrence.items() 
					 if len(occurrences) >= min_occurrence}
		
		# Count genes by frequency
		frequency_counts = {}
		for gene, occurrences in gene_occurrence.items():
			freq = len(occurrences)
			if freq not in frequency_counts:
				frequency_counts[freq] = 0
			frequency_counts[freq] += 1
		
		# Sort by frequency
		frequency_df = pd.DataFrame([
			{'frequency': k, 'gene_count': v} 
			for k, v in frequency_counts.items()
		]).sort_values('frequency')
		
		return {
			'core_genes': core_genes,
			'frequency_distribution': frequency_df,
			'total_unique_genes': len(gene_occurrence),
			'core_gene_count': len(core_genes)
		}
	
	def visualize_core_genes(self, core_gene_data=None, figsize=(12, 5)):
		"""
		Visualize the core gene analysis.
		
		Parameters:
		-----------
		core_gene_data : dict, optional
			Core gene analysis from identify_core_genes
			If None, run identify_core_genes
		figsize : tuple
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			Figure with the visualizations
		"""
		# Get core gene data if not provided
		if core_gene_data is None:
			core_gene_data = self.identify_core_genes()
		
		# Create figure with subplots
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
		
		# Plot 1: Frequency distribution
		freq_df = core_gene_data['frequency_distribution']
		ax1.bar(freq_df['frequency'], freq_df['gene_count'])
		ax1.set_xlabel('Number of Threshold Combinations')
		ax1.set_ylabel('Number of Genes')
		ax1.set_title('Distribution of Gene Occurrence Frequency')
		
		# Plot 2: Top genes and their threshold combinations
		# Sort genes by occurrence count
		top_genes = sorted(
			core_gene_data['core_genes'].items(), 
			key=lambda x: len(x[1]), 
			reverse=True
		)[:10]  # Top 10 genes
		
		# Create a matrix for visualization
		if top_genes:
			# Get all unique combinations
			all_combos = set()
			for _, combos in top_genes:
				all_combos.update(combos)
			all_combos = sorted(all_combos)
			
			# Create matrix
			matrix = np.zeros((len(top_genes), len(all_combos)))
			
			for i, (gene, combos) in enumerate(top_genes):
				for j, combo in enumerate(all_combos):
					if combo in combos:
						matrix[i, j] = 1
			
			# Plot matrix
			im = ax2.imshow(matrix, cmap='Blues', aspect='auto')
			
			# Add labels
			ax2.set_yticks(np.arange(len(top_genes)))
			ax2.set_yticklabels([gene for gene, _ in top_genes])
			
			# Format threshold combinations for display
			combo_labels = [f"({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})" for c in all_combos]
			ax2.set_xticks(np.arange(len(all_combos)))
			ax2.set_xticklabels(combo_labels, rotation=90, fontsize=8)
			
			ax2.set_title('Top Core Genes and Their Threshold Combinations')
		else:
			ax2.text(0.5, 0.5, "No core genes found", ha='center', va='center')
		
		plt.tight_layout()
		return fig
