
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class PromoterExpressionShiftAnalysis:
	"""Analysis of timing of promoter occupancy
	 and expression of deconvolution data"""

	def __init__(self, expression_analysis, chromatin_analysis, config1):

		self.config1 = config1
		self.chromatin_analysis = chromatin_analysis
		self.expression_analysis = expression_analysis

		# Determine set of gene expression genes
		unchanging_orfs, mg1_only, mg1_and_dg1, dg1_only, postg1_only = \
			expression_analysis.retrieve_mg1_dg1_specific_genes(config1, 
				proportion_threshold=0.1)

		self.gene_subsets = {
			'MG1 only': mg1_only,
			'DG1 only': dg1_only,
			'MG1 and DG1': mg1_and_dg1,
			'Post G1': postg1_only,
			'Uncategorized': unchanging_orfs
		}

	def compute_high_std_promoter_orfs_set(self):

		def _filter_to_high_std_genes(small_promoter_occupancies_df, indices, percentile):
				small_stds = small_promoter_occupancies_df[indices].std(1)
				qval = np.quantile(small_stds, q=percentile)
				high_std_genes = small_stds[small_stds >= qval].index
				return small_stds, high_std_genes, qval

		percentile = 0.5
		config = self.config1
		small_promoter_occupancies_df = self.chromatin_analysis.small_promoter_occupancies_df
		t_indices = config.t_indices()
		b_indices = config.b_indices()

		b_stds, high_b_std_genes, b_qv = _filter_to_high_std_genes(small_promoter_occupancies_df, 
																  b_indices, percentile=percentile)
		t_stds, high_t_std_genes, t_qv = _filter_to_high_std_genes(small_promoter_occupancies_df, 
																  t_indices, percentile=percentile)

		plt.figure(figsize=(9, 3))
		plt.subplot(1, 2, 1)
		plt.hist(t_stds, bins=60)
		plt.xlabel("Standard deviation of occupancy")
		plt.ylabel("Frequency")
		plt.title(f"Mother branch, n={len(high_b_std_genes)}/{len(small_promoter_occupancies_df)}")

		plt.axvline(t_qv, c='red')
		plt.subplot(1, 2, 2)
		plt.hist(b_stds, bins=60)
		plt.axvline(b_qv, c='red')
		plt.title(f"Daughter branch, n={len(high_t_std_genes)}/{len(small_promoter_occupancies_df)}")

		high_std_genes = set(high_b_std_genes).intersection(high_t_std_genes)

		print(f"Percentile values for threshold (top/bottom): {t_qv:.2f}, {b_qv:.2f}")

		plt.suptitle(f"Promoter occupancy $\sigma$ bottom branch, {percentile*100} percentile",
					 y=1.1, fontsize=16)

		# The individual standard deviations and sets of genes
		# with high enough variation to be included in the analysis
		# Top and bottom branches are computed independently
		self.b_stds = b_stds
		self.high_b_std_genes = high_b_std_genes
		self.b_qv = b_qv

		self.t_stds = t_stds
		self.high_t_std_genes = high_t_std_genes
		self.t_qv = t_qv


	def filter_gene_subsets_for_high_std(self):
		def _filter_gene_subsets(gene_subsets, high_std_genes, suffix="high std"):
			"""
			Filter gene subsets to include only genes that are also in high_std_genes.
			
			Parameters:
			-----------
			gene_subsets : dict
				Dictionary mapping subset names to lists of genes
			high_std_genes : list or set
				List/set of genes with high standard deviation
			suffix : str, optional
				Suffix to append to the filtered subset names (default: "high std")
			
			Returns:
			--------
			dict
				Dictionary of filtered gene subsets
			"""
			# Create filtered gene subsets
			filtered_gene_subsets = {}
			
			for subset_name, gene_list in gene_subsets.items():
				# Find the intersection between this subset and high_std_genes
				filtered_genes = [gene for gene in gene_list if gene in high_std_genes]
				
				# Only include the subset if it has genes after filtering
				if filtered_genes:
					filtered_gene_subsets[f'{subset_name} ({suffix})'] = filtered_genes
			
			# Print the number of genes in each filtered subset
			for name, genes in filtered_gene_subsets.items():
				print(f"{name}: {len(genes)} genes")
			
			return filtered_gene_subsets

		print("todo: This step allows us to justify the minimum std to classify a promoter"
			  " with binding dynamics as activating or repressing")
		# self.b_filtered_gene_subsets = _filter_gene_subsets(self.gene_subsets, 
		# 	self.high_b_std_genes)
		# self.t_filtered_gene_subsets = _filter_gene_subsets(self.gene_subsets, 
		# 	self.high_t_std_genes)
		self.b_filtered_gene_subsets = self.gene_subsets
		self.t_filtered_gene_subsets = self.gene_subsets


	def compute_promoter_expression_correlations(self, debug=False):

		from src.chromatin_expression_optimal_shift import ExpressionCorrelationAnalyzer
		from src.timer import Timer

		expressions_df = self.expression_analysis.deconvolved_genes_F
		small_occupancies = self.chromatin_analysis.small_promoter_occupancies_df

		if debug:
			expressions_df = expressions_df.iloc[:100]
			small_occupancies = small_occupancies.iloc[:100]

		max_shift = len(self.config1.t_indices())//2

		timer = Timer()

		self.correlation_calculator = ExpressionCorrelationAnalyzer(expressions_df, 
			small_occupancies, 
		    max_shift=max_shift)

		# Compute the correlations for each gene
		self.correlation_results = self.correlation_calculator.analyze_branches(self.config1)
		timer.print_time()


	def plot_expression_quantiles(self, normalize=True, figsize=(20, 6)):
		"""
		Plot expression profiles with median and 25-75% quantiles for multiple gene sets.
		
		Parameters:
		-----------
		normalize : bool, default=False
			If True, normalize each gene to have mean=0 and std=1 before plotting
		figsize : tuple
			Figure size (width, height)
		"""

		config = self.config1
		expressions_df = self.expression_analysis.deconvolved_genes_F
		t_indices = config.t_indices()
		b_indices = config.b_indices()
		gene_subsets = self.gene_subsets

		# Get timepoints for branches
		t_tps = config.get_timepoints_for_branch('t')
		
		# Create a copy of the expression data to avoid modifying the original
		if normalize:
			# Create a normalized copy of the expression data
			# We'll normalize each gene (row) across all conditions
			all_indices = expressions_df.columns
			expressions_norm = expressions_df.copy()
			
			# For each gene in the dataframe
			for gene in expressions_norm.index:
				gene_data = expressions_norm.loc[gene, all_indices]
				gene_mean = gene_data.mean()
				gene_std = gene_data.std()
				if gene_std > 0:  # Avoid division by zero
					expressions_norm.loc[gene, all_indices] = (gene_data - gene_mean) / gene_std
				else:
					expressions_norm.loc[gene, all_indices] = 0  # Set to zero if std is zero
					
			plot_data = expressions_norm
		else:
			plot_data = expressions_df
		
		# Create the figure with columns and 2 rows
		fig, axes = plt.subplots(2, len(gene_subsets), figsize=figsize, sharex='col')
		plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing
		
		# Find global min and max for consistent y-limits
		y_min = float('inf')
		y_max = float('-inf')
		
		# First pass to calculate global min and max
		for set_name, gene_set in gene_subsets.items():
			if len(gene_set) == 0:
				continue
				
			# Get data for both branches
			data_t = plot_data.loc[gene_set][t_indices].T
			data_b = plot_data.loc[gene_set][b_indices].T
			
			# Calculate quantiles
			q25_t = data_t.quantile(0.25, axis=1)
			q75_t = data_t.quantile(0.75, axis=1)
			q25_b = data_b.quantile(0.25, axis=1)
			q75_b = data_b.quantile(0.75, axis=1)
			
			# Update global min and max
			y_min = min(y_min, q25_t.min(), q25_b.min())
			y_max = max(y_max, q75_t.max(), q75_b.max())
		
		# Add a small buffer to the limits (10%)
		y_range = y_max - y_min
		y_min = y_min - 0.05 * y_range
		y_max = y_max + 0.1 * y_range
		
		# Loop through each gene subset
		for col, (title, gene_set) in enumerate(gene_subsets.items()):
			# Skip if gene set is empty
			if len(gene_set) == 0:
				axes[0, col].text(0.5, 0.5, "No genes in set", 
								 ha='center', va='center', transform=axes[0, col].transAxes)
				axes[1, col].text(0.5, 0.5, "No genes in set", 
								 ha='center', va='center', transform=axes[1, col].transAxes)
				continue
			
			# Top branch (t_indices)
			# Calculate median and quantiles
			data_t = plot_data.loc[gene_set][t_indices].T
			median_t = data_t.median(axis=1)
			q25_t = data_t.quantile(0.25, axis=1)
			q75_t = data_t.quantile(0.75, axis=1)
			
			# Plot median line and fill between quantiles
			axes[0, col].plot(t_tps, median_t, linewidth=2, color='blue')
			axes[0, col].fill_between(t_tps, q25_t, q75_t, alpha=0.3, color='blue')
			axes[0, col].set_title(f"{title}\n(n={len(gene_set)})")
			
			# Set ylabel based on whether data is normalized
			if normalize:
				axes[0, col].set_ylabel("Normalized Expression (z-score)")
			else:
				axes[0, col].set_ylabel("Expression")
				
			axes[0, col].set_ylim(y_min, y_max)  # Set consistent y-limits
			
			# Add a label for the top row
			if col == 0:
				axes[0, col].text(-0.3, 0.5, "Top Branch", 
								 transform=axes[0, col].transAxes, 
								 rotation=90, va='center', fontweight='bold')
			
			# Bottom branch (b_indices)
			# Calculate median and quantiles
			data_b = plot_data.loc[gene_set][b_indices].T
			median_b = data_b.median(axis=1)
			q25_b = data_b.quantile(0.25, axis=1)
			q75_b = data_b.quantile(0.75, axis=1)
			
			# Plot median line and fill between quantiles
			axes[1, col].plot(t_tps, median_b, linewidth=2, color='red')
			axes[1, col].fill_between(t_tps, q25_b, q75_b, alpha=0.3, color='red')
			axes[1, col].set_xlabel("Time")
			
			# Set ylabel based on whether data is normalized
			if normalize:
				axes[1, col].set_ylabel("Normalized Expression (z-score)")
			else:
				axes[1, col].set_ylabel("Expression")
				
			axes[1, col].set_ylim(y_min, y_max)  # Set consistent y-limits
			
			# Add a label for the bottom row
			if col == 0:
				axes[1, col].text(-0.3, 0.5, "Bottom Branch", 
								 transform=axes[1, col].transAxes, 
								 rotation=90, va='center', fontweight='bold')
		
		# Add a main title
		title_suffix = " (Normalized)" if normalize else ""
		plt.suptitle(f"Expression Profiles Across Different Gene Sets{title_suffix}", fontsize=16, y=1.05)
		
		# Adjust the layout
		plt.tight_layout()
		
		return fig, axes