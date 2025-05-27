import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from src.plot_helpers import color_for_key, plot_rect2
from src.timer import Timer
from src.utils import mkdir_safe, print_fl
from src.figure_configs import save_figure_for_paper
from src.config import load_cloccs_configs
from src.helpers import interpolate_increase_length


class FindAlphaSweepDS:
	"""
	Class for performing alpha parameter sweeps for daughter-specific genes.
	
	This class helps determine the optimal alpha parameter (which controls the end of mitosis/start of G1 timing) 
	by running deconvolution with different alpha values on daughter-specific genes and analyzing correlations
	between replicates.
	"""
	
	def __init__(self, output_directory, gene_names=['DSE1', 'DSE2', 'DSE3', 'DSE4'],
			 selected_origins=['oridb_10', 'oridb_295', 'oridb_289', 'oridb_193', 
							  'oridb_558', 'oridb_498', 'oridb_68', 'oridb_817'],
			 verbose=True):
		"""
		Initialize the FindAlphaSweepDS class.
		
		Parameters:
		-----------
		output_directory : str
			Directory where results will be saved
		gene_names : list, optional
			List of daughter-specific genes to analyze (default: DSE1-4)
		selected_origins : list, optional
			List of origin IDs to analyze for footprint deconvolution
		verbose : bool, optional
			Whether to print verbose output (default: True)
		"""
		self.output_directory = output_directory
		self.save_dir = f"{output_directory}/alpha_search"
		self.verbose = verbose
		self.timer = Timer()
		self.gene_names = gene_names
		self.selected_origins = selected_origins
		
		# Chromatin deconvolution parameters
		self.origin_gamma = 0.14
		self.origin_kappa = 0.01
		self.gene_kappa = 0.0
		
		# Load CLOCCS configs and initialize runner
		from pipeline.CombinedDeconvolveGeneExpressionRunner import CombinedDeconvolveGeneExpressionRunner
		self.config1, self.config2 = load_cloccs_configs()
		self.runner = CombinedDeconvolveGeneExpressionRunner(output_directory)

		# Create output directory
		mkdir_safe(self.save_dir)
		
		# Initialize results dataframes
		self.all_alpha_results_df = None
		self.dg1_mg1_ratios = None
		self.correlation_results = None
		self.thresholded_ratios = None
		self.combined_gene_results = None
		self.combined_origin_results = None

	def gene_alpha_search(self, gene_name, kappa, replicate, alphas):
		"""
		Search for optimal alpha parameter for a specific gene and replicate.
		
		Parameters:
		-----------
		gene_name : str
			Name of the gene to analyze
		kappa : float
			Kappa parameter for deconvolution
		replicate : int
			Replicate number (1 or 2)
		alphas : array-like
			Alpha values to test
			
		Returns:
		--------
		pd.DataFrame
			DataFrame containing results for each alpha value
		"""
		solutions = []
		if self.verbose:
			print_fl(f"Searching replicate {replicate} alphas: {gene_name}...")

		for alpha in alphas:
			if self.verbose:
				print_fl(alpha, end=",")
				
			# Modify alpha in appropriate config
			if replicate == 1:
				self.runner.config1.modify_alpha(alpha)
			elif replicate == 2:
				self.runner.config2.modify_alpha(alpha)
				
			# Run deconvolution with current alpha
			find_gamma = self.runner.deconvolve_gene_find_gamma(
				gene_name, kappa=kappa, replicate=replicate
			)
			
			solution = find_gamma.retrieve_solution()
			solutions.append(solution)
			
		if self.verbose:
			print_fl("")

		# Create DataFrame with results
		df = pd.DataFrame(solutions, index=alphas)
		df.index.name = 'alpha'
		df['gene'] = gene_name
		df['replicate'] = replicate
		df = df.reset_index().set_index(['replicate', 'alpha', 'gene'])
		
		return df

	def run_alpha_sweep(self, alphas=None, kappa=0.0):
		"""
		Run alpha parameter sweep for all genes and replicates.
		
		Parameters:
		-----------
		alphas : array-like, optional
			Alpha values to test. If None, uses default range.
		kappa : float, optional
			Kappa parameter for deconvolution (default: 0.0)
			
		Returns:
		--------
		pd.DataFrame
			Combined DataFrame with results for all genes and replicates
		"""
		if alphas is None:
			alphas = np.arange(4, 50, 2)  # Default from notebook

		if self.verbose:
			print_fl(f"Number of alphas: {len(alphas)}")
			print_fl(f"Testing alpha values: {alphas}")
			print_fl(f"Running for genes: {self.gene_names}")

		# Run alpha sweep for each gene and replicate
		solutions = []
		for replicate in [1, 2]:
			for gene_name in self.gene_names:
				solution_result = self.gene_alpha_search(
					gene_name, kappa=kappa, replicate=replicate, alphas=alphas
				)
				solutions.append(solution_result)
				
				if self.verbose:
					self.timer.print_time()

		# Combine all results
		self.all_alpha_results_df = pd.concat(solutions)
		return self.all_alpha_results_df

	def load_existing_results(self, filepath=None):
		"""
		Load existing alpha search results from disk.
		
		Parameters:
		-----------
		filepath : str, optional
			Path to the CSV file. If None, uses default path.
		"""
		if filepath is None:
			filepath = f"{self.save_dir}/alphas_repl1.csv"
		
		if self.verbose:
			print_fl(f"Loading existing results from {filepath}")
		
		self.all_alpha_results_df = pd.read_csv(filepath).set_index(['replicate', 'alpha', 'gene'])
		self.all_alpha_results_df.columns = self.all_alpha_results_df.columns.astype(int)
		
		if self.verbose:
			print_fl(f"Loaded {len(self.all_alpha_results_df)} rows of results")
		
		return self.all_alpha_results_df

	def compute_dg1_mg1_ratios(self):
		"""
		Compute DG1/MG1 ratios for each entry in the results.
		
		Returns:
		--------
		pd.DataFrame
			DataFrame with DG1/MG1 ratios for each alpha-gene combination
		"""
		if self.all_alpha_results_df is None:
			raise ValueError("No results loaded. Run alpha sweep or load existing results first.")
		
		if self.verbose:
			print_fl("Computing DG1/MG1 ratios...")
		
		results = self.all_alpha_results_df
		self.dg1_mg1_ratios = results[[]].copy()
		self.dg1_mg1_ratios['dm_ratio'] = None
		
		alphas = self.dg1_mg1_ratios.index.get_level_values(1).unique()
		
		for replicate in [1, 2]:
			for alpha in alphas:
				config = self.config1 if replicate == 1 else self.config2
				config.modify_alpha(alpha)
				
				gene_results = results.loc[replicate].loc[alpha]
				gene_dg1_cg1_ratios = (gene_results[config.get_Hpositions_for_phase('DG1')].mean(1) / 
									   gene_results[config.get_Hpositions_for_phase('CG1')].mean(1))
				
				for gene_idx in gene_dg1_cg1_ratios.index:
					self.dg1_mg1_ratios.at[(replicate, alpha, gene_idx), 'dm_ratio'] = \
						gene_dg1_cg1_ratios[gene_idx]
		
		if self.verbose:
			print_fl("DG1/MG1 ratios computed successfully")
		
		return self.dg1_mg1_ratios

	def threshold_ratios(self, threshold=1.6):
		"""
		Threshold the ratios to identify feasible alpha ranges for each replicate.
		
		Parameters:
		-----------
		threshold : float, optional
			Minimum DG1/MG1 ratio threshold (default: 1.6)
			
		Returns:
		--------
		tuple
			(mean_ratios, thresholded_ratios, rep1_range, rep2_range)
		"""
		if self.dg1_mg1_ratios is None:
			raise ValueError("DG1/MG1 ratios not computed. Run compute_dg1_mg1_ratios() first.")
		
		if self.verbose:
			print_fl(f"Applying threshold of {threshold} to ratios...")
		
		mean_ratios = self.dg1_mg1_ratios.groupby(['replicate', 'alpha']).mean()
		thresholded_mean_ratios = mean_ratios[mean_ratios.dm_ratio > threshold]
		
		# Get feasible ranges for each replicate
		try:
			rep1_range = (thresholded_mean_ratios.loc[1].index.min(), 
						  thresholded_mean_ratios.loc[1].index.max())
		except KeyError:
			rep1_range = (None, None)
			if self.verbose:
				print_fl("Warning: No feasible alpha values found for replicate 1")
		
		try:
			rep2_range = (thresholded_mean_ratios.loc[2].index.min(), 
						  thresholded_mean_ratios.loc[2].index.max())
		except KeyError:
			rep2_range = (None, None)
			if self.verbose:
				print_fl("Warning: No feasible alpha values found for replicate 2")
		
		self.thresholded_ratios = {
			'mean_ratios': mean_ratios,
			'thresholded': thresholded_mean_ratios,
			'rep1_range': rep1_range,
			'rep2_range': rep2_range,
			'threshold': threshold
		}
		
		if self.verbose:
			print_fl(f"Replicate 1 feasible range: {rep1_range}")
			print_fl(f"Replicate 2 feasible range: {rep2_range}")
		
		return mean_ratios, thresholded_mean_ratios, rep1_range, rep2_range

	def plot_ratio_analysis(self, threshold=1.6, save=True):
		"""
		Plot the DG1/MG1 ratio analysis for both replicates.
		
		Parameters:
		-----------
		threshold : float, optional
			Threshold value to display (default: 1.6)
		save : bool, optional
			Whether to save the figure (default: True)
		"""
		if self.thresholded_ratios is None:
			_, _, rep1_range, rep2_range = self.threshold_ratios(threshold)
		else:
			rep1_range = self.thresholded_ratios['rep1_range']
			rep2_range = self.thresholded_ratios['rep2_range']
			mean_ratios = self.thresholded_ratios['mean_ratios']
		
		plt.figure(figsize=(9, 3))
		
		# Plot replicate 1
		plt.subplot(1, 2, 1)
		plt.plot(mean_ratios.loc[1].index, mean_ratios.loc[1].dm_ratio)
		plt.title("Replicate 1")
		plt.xlabel("$\\alpha$ value")
		plt.ylabel("DG1 / MG1 ratio")
		plt.ylim(0, 4)
		plt.axhline(threshold, c='black', ls='dotted', lw=1)
		if rep1_range[0] is not None:
			plt.axvspan(rep1_range[0], rep1_range[1], 0, 4, color='#f0f0f0')
		
		# Plot replicate 2
		plt.subplot(1, 2, 2)
		plt.plot(mean_ratios.loc[2].index, mean_ratios.loc[2].dm_ratio)
		plt.title("Replicate 2")
		plt.xlabel("$\\alpha$ value")
		plt.ylim(0, 4)
		plt.axhline(threshold, c='black', ls='dotted', lw=1)
		if rep2_range[0] is not None:
			plt.axvspan(rep2_range[0], rep2_range[1], 0, 4, color='#f0f0f0')
		
		plt.suptitle("Range of $\\alpha$ values with high ratio of DG1/MG1 expression",
					fontweight='demi', fontsize=16, y=1.1)
		plt.tight_layout()
		
		if save:
			save_path = f"{self.save_dir}/dg1_mg1_ratio_analysis.png"
			save_figure_for_paper(save_path, dpi=80)
			if self.verbose:
				print_fl(f"Ratio analysis plot saved to {save_path}")

	def compute_correlations(self, alphas=None):
		"""
		Compute Pearson correlations between all alpha combinations across replicates.
		
		Parameters:
		-----------
		alphas : array-like, optional
			Alpha values to include in correlation analysis. If None, uses all available.
			
		Returns:
		--------
		pd.DataFrame
			DataFrame with correlation results for each alpha pair and gene
		"""
		if self.all_alpha_results_df is None:
			raise ValueError("No results available. Run alpha sweep or load existing results first.")
		
		if alphas is None:
			alphas = self.all_alpha_results_df.index.get_level_values(1).unique()
		
		if self.verbose:
			print_fl("Computing correlations between alpha combinations...")
			print_fl(f"Testing {len(alphas)}x{len(alphas)} = {len(alphas)**2} combinations")
		
		results = self.all_alpha_results_df
		genes = results.index.get_level_values(2).unique()
		correlation_results = []
		
		timer = Timer()
		
		for alpha1 in alphas:
			for alpha2 in alphas:
				self.config1.modify_alpha(alpha1)
				self.config2.modify_alpha(alpha2)
				
				gene_results1 = results.loc[1].loc[alpha1]
				gene_results2 = results.loc[2].loc[alpha2]
				
				for gene_idx in genes:
					# Get branch data for both replicates
					b1, b2 = self._retrieve_branch_data(gene_results1, gene_results2, gene_idx, 'b')
					t1, t2 = self._retrieve_branch_data(gene_results1, gene_results2, gene_idx, 't')
					r1, r2 = self._retrieve_branch_data(gene_results1, gene_results2, gene_idx, 'i')
					
					# Concatenate and correlate all branches
					gene1_data = np.concatenate([r1, b1, t1])
					gene2_data = np.concatenate([r2, b2, t2])
					
					correlation, p_value = pearsonr(gene1_data, gene2_data)
					
					correlation_results.append({
						'alpha1': alpha1,
						'alpha2': alpha2,
						'gene': gene_idx,
						'correlation': correlation,
						'p_value': p_value,
						'n_points': min(len(gene1_data), len(gene2_data))
					})
			
			if self.verbose:
				timer.print_time()
		
		# Convert to DataFrame
		self.correlation_results = pd.DataFrame(correlation_results)
		self.correlation_results = self.correlation_results.set_index(['alpha1', 'alpha2', 'gene'])
		
		if self.verbose:
			print_fl("Correlation computation completed")
		
		return self.correlation_results

	def _retrieve_branch_data(self, gene_results1, gene_results2, gene_idx, branch):
		"""
		Helper method to retrieve and interpolate branch data for correlation analysis.
		"""
		config1_indices = getattr(self.config1, f'get_Hpositions_for_branch')(branch)
		config2_indices = getattr(self.config2, f'get_Hpositions_for_branch')(branch)
		
		gene1_data = gene_results1.loc[gene_idx][config1_indices]
		gene2_data = gene_results2.loc[gene_idx][config2_indices]
		
		# Interpolate if lengths don't match
		if len(gene1_data) > len(gene2_data):
			gene2_data = interpolate_increase_length(gene2_data, len(gene1_data))
		elif len(gene1_data) < len(gene2_data):
			gene1_data = interpolate_increase_length(gene1_data, len(gene2_data))
		
		return gene1_data, gene2_data

	def plot_correlation_heatmap(self, save=True):
		"""
		Plot correlation heatmap with threshold annotations.
		
		Parameters:
		-----------
		save : bool, optional
			Whether to save the figure (default: True)
		"""
		if self.correlation_results is None:
			raise ValueError("Correlations not computed. Run compute_correlations() first.")
		
		if self.thresholded_ratios is None:
			raise ValueError("Thresholded ratios not computed. Run threshold_ratios() first.")
		
		# Compute mean correlations across genes
		correlation_means = self.correlation_results.groupby(['alpha1', 'alpha2']).mean()[['correlation']]
		correlation_pivot = correlation_means.reset_index().pivot(
			index='alpha1', columns='alpha2', values='correlation'
		)
		
		fig, ax = plt.subplots(figsize=(5, 4))
		
		# Display the heatmap
		im = ax.imshow(correlation_pivot, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1,
					   origin='lower',
					   extent=[correlation_pivot.columns[0]-1, correlation_pivot.columns[-1]+1, 
							   correlation_pivot.index[0]-1, correlation_pivot.index[-1]+1])
		
		# Set the ticks and labels
		ax.set_xticks(correlation_pivot.columns[::2])  # Show every other tick to avoid crowding
		ax.set_yticks(correlation_pivot.index[::2])
		ax.tick_params(axis='both', which='major', labelsize=8)
		
		# Add colorbar
		cbar = ax.figure.colorbar(im, ax=ax)
		cbar.ax.set_ylabel('Pearson Correlation', rotation=-90, va="bottom")
		
		ax.set_title("Correlation of replicate\\ndaughter-specific gene expression", 
					fontweight='demi', fontsize=13, y=1.02)
		ax.set_xlabel('Replicate 2 $\\alpha$')
		ax.set_ylabel('Replicate 1 $\\alpha$')
		ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
		
		# Add threshold region rectangle
		rep1_range = self.thresholded_ratios['rep1_range']
		rep2_range = self.thresholded_ratios['rep2_range']
		
		if rep1_range[0] is not None and rep2_range[0] is not None:
			plot_rect2(ax, rep2_range[0]-0.8, rep1_range[0]-0.8, 
					   rep2_range[1]+0.8, rep1_range[1]+0.8, 
					   edgecolor='black', lw=2, ls='solid', facecolor='none')
		
		plt.tight_layout()
		
		if save:
			save_path = f"{self.save_dir}/correlation_heatmap.png"
			save_figure_for_paper(save_path, dpi=80)
			if self.verbose:
				print_fl(f"Correlation heatmap saved to {save_path}")

	def find_optimal_alpha_pairs(self, top_n=20):
		"""
		Find optimal alpha pairs with highest correlation within feasible ranges.
		
		Parameters:
		-----------
		top_n : int, optional
			Number of top pairs to return (default: 20)
			
		Returns:
		--------
		pd.DataFrame
			Top alpha pairs sorted by correlation
		"""
		if self.correlation_results is None:
			raise ValueError("Correlations not computed. Run compute_correlations() first.")
		
		if self.thresholded_ratios is None:
			raise ValueError("Thresholded ratios not computed. Run threshold_ratios() first.")
		
		# Get mean correlations
		correlation_means = self.correlation_results.groupby(['alpha1', 'alpha2']).mean()[['correlation']]
		correlation_means = correlation_means.sort_values('correlation', ascending=False)
		
		# Get feasible alpha values
		thresholded_ratios1 = self.thresholded_ratios['thresholded'].loc[1] if 1 in self.thresholded_ratios['thresholded'].index else pd.Series(dtype=float)
		thresholded_ratios2 = self.thresholded_ratios['thresholded'].loc[2] if 2 in self.thresholded_ratios['thresholded'].index else pd.Series(dtype=float)
		
		# Filter to only feasible combinations
		feasible_indices = correlation_means.index[
			correlation_means.reset_index()['alpha1'].isin(thresholded_ratios1.index) &
			correlation_means.reset_index()['alpha2'].isin(thresholded_ratios2.index)
		]
		
		optimal_pairs = correlation_means.loc[feasible_indices].head(top_n)
		
		if self.verbose:
			print_fl(f"Top {len(optimal_pairs)} optimal alpha pairs within feasible ranges:")
			print_fl(optimal_pairs)
		
		return optimal_pairs

	def run_correlation_analysis(self, threshold=1.6, alphas=None, save_plots=True):
		"""
		Run the complete correlation analysis workflow.
		
		Parameters:
		-----------
		threshold : float, optional
			DG1/MG1 ratio threshold (default: 1.6)
		alphas : array-like, optional
			Alpha values to analyze (default: all available)
		save_plots : bool, optional
			Whether to save generated plots (default: True)
			
		Returns:
		--------
		dict
			Dictionary containing all analysis results
		"""
		if self.verbose:
			print_fl("Starting correlation analysis workflow...")
		
		# Compute ratios if not already done
		if self.dg1_mg1_ratios is None:
			self.compute_dg1_mg1_ratios()
		
		# Threshold ratios
		mean_ratios, thresholded_ratios, rep1_range, rep2_range = self.threshold_ratios(threshold)
		
		# Plot ratio analysis
		self.plot_ratio_analysis(threshold, save=save_plots)
		
		# Compute correlations
		self.compute_correlations(alphas)
		
		# Plot correlation heatmap
		self.plot_correlation_heatmap(save=save_plots)
		
		# Find optimal pairs
		optimal_pairs = self.find_optimal_alpha_pairs()
		
		results = {
			'mean_ratios': mean_ratios,
			'thresholded_ratios': thresholded_ratios,
			'rep1_range': rep1_range,
			'rep2_range': rep2_range,
			'correlations': self.correlation_results,
			'optimal_pairs': optimal_pairs
		}
		
		if self.verbose:
			print_fl("Correlation analysis completed successfully")
		
		return results

	def _plot_single_alpha_row(self, ax_row, alpha, alpha_data, config1,
		config2=None, genes=None):
		"""
		Base function to plot a single row (all genes + average) for given alpha value(s).
		
		Parameters:
		-----------
		ax_row : array of matplotlib axes
			Row of axes to plot on
		alpha_data : pd.DataFrame
			Data for this alpha combination, indexed by gene
		config1 : object
			Primary configuration object
		config2 : object, optional
			Second configuration object for combined model
		genes : list, optional
			List of genes to plot. If None, uses all genes in alpha_data
		"""
		if genes is None:
			genes = sorted(alpha_data.index.unique())
		
		n_genes = len(genes)
		
		# Get timepoints and indices (use config1 as primary, assuming indices match for combined)
		i_tps = config1.get_timepoints_for_branch('i')
		t_tps = config1.get_timepoints_for_branch('t') 
		b_tps = config1.get_timepoints_for_branch('b')
		
		i_indices = config1.get_Hpositions_for_branch('i')
		t_indices = config1.get_Hpositions_for_branch('t')
		b_indices = config1.get_Hpositions_for_branch('b')
		
		# Colors
		i_color = '#a0a0a0'
		t_color = 'black'
		b_color = color_for_key('DG1')
		
		# Store for averaging
		all_i_values = []
		all_t_values = []
		all_b_values = []
		
		# Plot each gene
		for j, gene in enumerate(genes):
			gene_data = alpha_data.loc[gene]
			
			# Extract branch data
			i_values = gene_data[i_indices].values
			t_values = gene_data[t_indices].values
			b_values = gene_data[b_indices].values
			
			# Store for averaging
			all_i_values.append(i_values)
			all_t_values.append(t_values)
			all_b_values.append(b_values)
			
			# Plot on corresponding subplot
			ax = ax_row[j]
			ax.plot(i_tps, i_values, lw=3, color=i_color, label='Recovery Branch')
			ax.plot(t_tps, t_values, lw=3, color=t_color, label='Top Branch')
			ax.plot(b_tps, b_values, lw=3, color=b_color, label='Bottom Branch')
			
			ax.axvline(0, c='#ddd', lw=1)
			ax.legend()
		
		# Plot average in last column
		if all_i_values and all_t_values and all_b_values:
			avg_i_values = np.mean(all_i_values, axis=0)
			avg_t_values = np.mean(all_t_values, axis=0)
			avg_b_values = np.mean(all_b_values, axis=0)
			
			ax = ax_row[n_genes]  # Last column for average
			ax.plot(i_tps, avg_i_values, lw=5, color=i_color, label='Recovery Branch')
			ax.plot(t_tps, avg_t_values, lw=5, color=t_color, label='Top Branch')
			ax.plot(b_tps, avg_b_values, lw=5, color=b_color, label='Bottom Branch')
			ax.axvline(0, c='#ddd', lw=1)
			ax.legend()

	def plot_alpha_genes_subplots(self, plot_alphas, replicate):
		"""
		Plot data for a specific replicate showing values for each alpha and gene combination.
		
		Parameters:
		-----------
		plot_alphas : array-like
			Alpha values to plot
		replicate : int
			The replicate number to plot (1 or 2)
		
		Returns:
		--------
		matplotlib.figure.Figure
			The figure containing all subplots
		"""
		if self.all_alpha_results_df is None:
			raise ValueError("No results available. Run alpha sweep or load existing results first.")
		
		# Select data for the specified replicate
		replicate_data = self.all_alpha_results_df.loc[replicate]
		genes = sorted(replicate_data.index.get_level_values('gene').unique())
		
		n_alphas = len(plot_alphas)
		n_genes = len(genes)
		
		# Create figure
		fig, axes = plt.subplots(n_alphas, n_genes + 1, figsize=(3. * (n_genes + 1), 3. * n_alphas))
		
		# Handle axis indexing for edge cases
		if n_alphas == 1 and n_genes == 1:
			axes = np.array([[axes[0], axes[1]]])
		elif n_alphas == 1:
			axes = axes.reshape(1, -1)
		elif n_genes == 1:
			axes = axes.reshape(-1, 2)
		
		# Set consistent y-limits
		all_data_no_nans = replicate_data.copy().fillna(0)
		ylims = 0, all_data_no_nans.values.max() * 1.2
		
		config = self.config1 if replicate == 1 else self.config2
		
		# Plot each alpha row
		for i, alpha in enumerate(plot_alphas):
			alpha_data = replicate_data.xs(alpha, level='alpha')
			ax_row = axes[i] if n_alphas > 1 else axes[0]

			config.modify_alpha(alpha)
			
			# Use base plotting function
			self._plot_single_alpha_row(ax_row, alpha, alpha_data, config, genes=genes)
			
			# Set labels and formatting for this row
			for j in range(n_genes + 1):
				ax = ax_row[j]
				ax.set_ylim(*ylims)
				
				if j == 0:  # First column - add y-label with alpha
					ax.set_ylabel(f"α={alpha}", fontsize=24)
				else:
					ax.set_yticks([])
				
				if i == 0:  # Top row - add gene titles
					if j < n_genes:
						ax.set_title('$\\it{' + genes[j] + '}$', fontsize=24, pad=5)
					else:
						ax.set_title('Average', fontsize=16, pad=5)
				
				if i == n_alphas - 1:  # Bottom row - add x-label
					if j == 0:  # Only on first subplot to avoid clutter
						ax.set_xlabel('Average single cell time, min', fontsize=16)
		
		# Adjust layout and add title
		plt.tight_layout(rect=[0, 0, 1, 0.96])
		fig.suptitle(f'Alpha search, replicate {replicate}', fontsize=38, fontweight='demi', y=1.05)
		
		return fig


	def run_full_analysis(self, alphas=None, kappa=0.0, threshold=1.6, load_existing=False, existing_filepath=None):
		"""
		Run the complete alpha sweep and correlation analysis workflow.
		
		Parameters:
		-----------
		alphas : array-like, optional
			Alpha values to test
		kappa : float, optional
			Kappa parameter for deconvolution
		threshold : float, optional
			DG1/MG1 ratio threshold
		load_existing : bool, optional
			Whether to load existing results instead of running new sweep
		existing_filepath : str, optional
			Path to existing results file
			
		Returns:
		--------
		dict
			Dictionary with comprehensive analysis results
		"""
		if load_existing:
			self.load_existing_results(existing_filepath)
		else:
			# Run alpha sweep
			self.run_alpha_sweep(alphas=alphas, kappa=kappa)
			# Plot and save initial results
			self.plot_and_save_results(alphas=alphas)
		
		# Run correlation analysis
		correlation_results = self.run_correlation_analysis(threshold=threshold, alphas=alphas)
		
		return correlation_results

	def plot_combined_alpha_results(self, alpha_pairs, combined_data, italics_title=True):
		"""
		Plot combined model results for alpha pair combinations.
		
		Parameters:
		-----------
		alpha_pairs : list of tuples
			List of (alpha1, alpha2) pairs to plot
		combined_data : pd.DataFrame  
			DataFrame with MultiIndex [(alpha1, alpha2), gene] containing combined model results
		
		Returns:
		--------
		matplotlib.figure.Figure
			The figure containing all subplots
		"""
		# Get genes from the data
		genes = sorted(combined_data.index.get_level_values(1).unique())
		
		n_pairs = len(alpha_pairs)
		n_genes = len(genes)
		
		# Create figure
		fig, axes = plt.subplots(n_pairs, n_genes + 1, figsize=(3. * (n_genes + 1), 3. * n_pairs))
		
		# Handle axis indexing for edge cases
		if n_pairs == 1 and n_genes == 1:
			axes = np.array([[axes[0], axes[1]]])
		elif n_pairs == 1:
			axes = axes.reshape(1, -1)
		elif n_genes == 1:
			axes = axes.reshape(-1, 2)
		
		# Set consistent y-limits
		all_data_no_nans = combined_data.copy().fillna(0)
		ylims = 0, all_data_no_nans.values.max() * 1.2
		
		# Plot each alpha pair row
		for i, (alpha1, alpha2) in enumerate(alpha_pairs):
			alpha_data = combined_data.xs((alpha1, alpha2), level=0)
			ax_row = axes[i] if n_pairs > 1 else axes[0]

			num_g1 = alpha1+alpha2
			self.config1.modify_alpha(alpha1, num_g1)
			self.config2.modify_alpha(alpha2, num_g1)
			
			# Use base plotting function
			self._plot_single_alpha_row(ax_row, (alpha1, alpha2), alpha_data, 
									   self.config1, self.config2, genes=genes)
			
			# Set labels and formatting for this row
			for j in range(n_genes + 1):
				ax = ax_row[j]
				ax.set_ylim(*ylims)
				
				if j == 0:  # First column - add y-label with alpha pair
					ax.set_ylabel(f"α₁={alpha1}, α₂={alpha2}", fontsize=20)
				else:
					ax.set_yticks([])
				
				if i == 0:  # Top row - add gene titles
					if j < n_genes:

						if italics_title:
							ax.set_title('$\\it{' + genes[j] + '}$', fontsize=24)
						else:
							ax.set_title(genes[j], fontsize=24)
					else:
						ax.set_title('Average', fontsize=16, pad=5)
				
				if i == n_pairs - 1:  # Bottom row - add x-label
					if j == 0:  # Only on first subplot to avoid clutter
						ax.set_xlabel('Average single cell time, min', fontsize=16)
		
		# Adjust layout and add title
		plt.tight_layout(rect=[0, 0, 1, 0.96])
		fig.suptitle('Combined model alpha search', fontsize=38, fontweight='demi', y=1.05)
		
		return fig

	def plot_and_save_results(self, alphas=None, save=True):
		"""
		Plot and save the results of the alpha sweep.
		
		Parameters:
		-----------
		alphas : array-like, optional
			Alpha values to plot. If None, uses all alphas in results.
		save : bool, optional
			Whether to save results and plots (default: True)
		"""
		if self.all_alpha_results_df is None:
			raise ValueError("No results to plot. Run 'run_alpha_sweep' first.")
			
		if alphas is None:
			# Extract unique alphas from the results
			alphas = self.all_alpha_results_df.reset_index()['alpha'].unique()

		# Plot and save for replicate 1
		fig1 = self.plot_alpha_genes_subplots(alphas, 1)

		if save:
			self.all_alpha_results_df.to_csv(f"{self.save_dir}/alphas_repl1.csv")
			save_figure_for_paper(f"{self.save_dir}/alphas_repl1.png", dpi=80)
		
		# Plot and save for replicate 2
		fig2 = self.plot_alpha_genes_subplots(alphas, 2)

		if save:
			self.all_alpha_results_df.to_csv(f"{self.save_dir}/alphas_repl2.csv")
			save_figure_for_paper(f"{self.save_dir}/alphas_repl2.png", dpi=80)
		
		if self.verbose:
			print_fl(f"Results saved to {self.save_dir}")


	def run_combined_gene_deconvolution(self, top_n_pairs=8):
		"""
		Run combined gene deconvolution for optimal alpha pairs.
		
		Parameters:
		-----------
		top_n_pairs : int, optional
			Number of top alpha pairs to analyze (default: 8)
			
		Returns:
		--------
		pd.DataFrame
			Combined gene deconvolution results for all alpha pairs and genes
		"""
		if self.correlation_results is None:
			raise ValueError("Correlations not computed. Run correlation analysis first.")
		
		# Get optimal alpha pairs
		optimal_pairs = self.find_optimal_alpha_pairs(top_n_pairs)
		alpha_pairs = optimal_pairs.index[:top_n_pairs]
		
		if self.verbose:
			print_fl(f"Running combined gene deconvolution for {len(alpha_pairs)} alpha pairs...")
		
		timer = Timer()
		all_alpha_gene_solutions = []
		
		for alphas in alpha_pairs:
			if self.verbose:
				print_fl(f"Deconvolving combined alphas: {alphas}")
			
			gene_solutions = []
			num_g1 = alphas[0] + alphas[1]
			
			for gene_name in self.gene_names:
				if self.verbose:
					print_fl(f"{gene_name}...", end='')
				
				# Run combined deconvolution
				self.runner.deconvolve_gene_find_gamma(
					gene_name, alphas=alphas, num_g1_indices=num_g1
				)
				F = self.runner.expression_find_gamma.retrieve_solution()
				gene_solutions.append(F)
				
				if self.verbose:
					timer.print_time()
			
			# Create DataFrame for this alpha pair
			gene_solutions_df = pd.DataFrame(gene_solutions, index=self.gene_names)
			gene_solutions_df.index.name = 'gene'
			gene_solutions_df['alphas'] = [alphas] * len(gene_solutions)
			gene_solutions_df = gene_solutions_df.reset_index().set_index(['alphas', 'gene'])
			
			all_alpha_gene_solutions.append(gene_solutions_df)
		
		self.combined_gene_results = pd.concat(all_alpha_gene_solutions)
		
		if self.verbose:
			print_fl("Combined gene deconvolution completed")
		
		return self.combined_gene_results


	def run_combined_origin_deconvolution(self, top_n_pairs=8, copy_correct=False):
		"""
		Run combined origin deconvolution for optimal alpha pairs.
		
		Parameters:
		-----------
		top_n_pairs : int, optional
			Number of top alpha pairs to analyze (default: 8)
		copy_correct : bool, optional
			Whether to apply copy correction (default: False)
			
		Returns:
		--------
		pd.DataFrame
			Combined origin deconvolution results for all alpha pairs and origins
		"""
		if self.correlation_results is None:
			raise ValueError("Correlations not computed. Run correlation analysis first.")
		
		# Import origin sweep class
		from pipeline.origin_alpha_sweep import OriginAlphaSweep
		origin_sweep = OriginAlphaSweep(self.output_directory)
		
		# Get optimal alpha pairs
		optimal_pairs = self.find_optimal_alpha_pairs(top_n_pairs)
		alpha_pairs = optimal_pairs.index[:top_n_pairs]
		
		if self.verbose:
			print_fl(f"Running combined origin deconvolution for {len(alpha_pairs)} alpha pairs...")
		
		timer = Timer()
		all_alpha_origin_results = []
		
		for alphas in alpha_pairs:
			if self.verbose:
				print_fl(f"Alphas: {alphas}")
			
			origin_footprint_results = []
			num_g1 = alphas[0] + alphas[1]
			origin_sweep.load_configs(modify_alphas=alphas, num_g1_indices=num_g1)
			
			for oridb in self.selected_origins:
				if self.verbose:
					print_fl(f"Deconvolving origin: {oridb}")
				
				# Analyze the origin
				footprint = origin_sweep.analyze_origin(
					oridb, copy_correct=copy_correct, verbose=False
				)
				
				origin_sweep.setup_footprint_deconvolution(
					replicate='combined', copy_correct=copy_correct
				)
				origin_sweep.deconvolve_footprint(
					gamma=self.origin_gamma, kappa=self.origin_kappa, verbose=False
				)
				
				footprint_F_mean = origin_sweep.footprint_F_imgs.mean((1, 2))
				origin_footprint_results.append(footprint_F_mean)
				
				if self.verbose:
					timer.print_time("\n")
			
			# Create DataFrame for this alpha pair
			origin_footprints_df = pd.DataFrame(
				origin_footprint_results, index=self.selected_origins
			)
			origin_footprints_df['alphas'] = [alphas] * len(origin_footprints_df)
			origin_footprints_df.index.name = 'origin_id'
			origin_footprints_df = origin_footprints_df.reset_index().set_index(
				['alphas', 'origin_id']
			)
			
			all_alpha_origin_results.append(origin_footprints_df)
		
		self.combined_origin_results = pd.concat(all_alpha_origin_results)
		
		if self.verbose:
			print_fl("Combined origin deconvolution completed")
		
		return self.combined_origin_results


	def save_combined_gene_results(self, filename=None):
		"""
		Save combined gene deconvolution results to CSV.
		
		Parameters:
		-----------
		filename : str, optional
			Custom filename. If None, uses default naming.
		"""
		if self.combined_gene_results is None:
			raise ValueError("No combined gene results to save. Run combined gene deconvolution first.")
		
		if filename is None:
			filename = f"{self.save_dir}/combined_gene_results.csv"
		
		self.combined_gene_results.to_csv(filename)
		
		if self.verbose:
			print_fl(f"Combined gene results saved to {filename}")


	def save_combined_origin_results(self, filename=None):
		"""
		Save combined origin deconvolution results to CSV.
		
		Parameters:
		-----------
		filename : str, optional
			Custom filename. If None, uses default naming.
		"""
		if self.combined_origin_results is None:
			raise ValueError("No combined origin results to save. Run combined origin deconvolution first.")
		
		if filename is None:
			filename = f"{self.save_dir}/combined_origin_results.csv"
		
		self.combined_origin_results.to_csv(filename)
		
		if self.verbose:
			print_fl(f"Combined origin results saved to {filename}")


	def save_combined_gene_figures(self, alpha_pairs=None, filename=None, **kwargs):
		"""
		Save combined gene deconvolution figures.
		
		Parameters:
		-----------
		alpha_pairs : list, optional
			Specific alpha pairs to plot. If None, uses all available.
		filename : str, optional
			Custom filename. If None, uses default naming.
		**kwargs : dict
			Additional arguments passed to plot_combined_alpha_results
		"""
		if self.combined_gene_results is None:
			raise ValueError("No combined gene results to plot. Run combined gene deconvolution first.")
		
		if alpha_pairs is None:
			alpha_pairs = self.combined_gene_results.index.get_level_values(0).unique()
		
		fig = self.plot_combined_alpha_results(alpha_pairs, self.combined_gene_results, **kwargs)
		
		if filename is None:
			filename = f"{self.save_dir}/combined_gene_deconvolution.png"
		
		save_figure_for_paper(filename, dpi=80)
		
		if self.verbose:
			print_fl(f"Combined gene figures saved to {filename}")
		
		return fig


	def save_combined_origin_figures(self, alpha_pairs=None, filename=None, **kwargs):
		"""
		Save combined origin deconvolution figures.
		
		Parameters:
		-----------
		alpha_pairs : list, optional
			Specific alpha pairs to plot. If None, uses all available.
		filename : str, optional
			Custom filename. If None, uses default naming.
		**kwargs : dict
			Additional arguments passed to plot_combined_alpha_results
		"""
		if self.combined_origin_results is None:
			raise ValueError("No combined origin results to plot. Run combined origin deconvolution first.")
		
		if alpha_pairs is None:
			alpha_pairs = self.combined_origin_results.index.get_level_values(0).unique()
		
		fig = self.plot_combined_alpha_results(alpha_pairs, self.combined_origin_results, 
											  italics_title=False, **kwargs)
		
		if filename is None:
			filename = f"{self.save_dir}/combined_origin_deconvolution.png"
		
		save_figure_for_paper(filename, dpi=80)
		
		if self.verbose:
			print_fl(f"Combined origin figures saved to {filename}")
		
		return fig


	def run_full_combined_analysis(self, top_n_pairs=8, copy_correct=False, 
								  save_results=True, save_figures=True):
		"""
		Run the complete combined analysis workflow including genes and origins.
		
		Parameters:
		-----------
		top_n_pairs : int, optional
			Number of top alpha pairs to analyze (default: 8)
		copy_correct : bool, optional
			Whether to apply copy correction for origins (default: False)
		save_results : bool, optional
			Whether to save results to CSV (default: True)
		save_figures : bool, optional
			Whether to save figures (default: True)
			
		Returns:
		--------
		dict
			Dictionary containing all combined analysis results
		"""
		if self.verbose:
			print_fl("Starting full combined analysis workflow...")
		
		# Run combined gene deconvolution
		gene_results = self.run_combined_gene_deconvolution(top_n_pairs)
		
		# Run combined origin deconvolution  
		origin_results = self.run_combined_origin_deconvolution(top_n_pairs, copy_correct)
		
		# Save results if requested
		if save_results:
			self.save_combined_gene_results()
			self.save_combined_origin_results()
		
		# Save figures if requested
		if save_figures:
			# Get alpha pairs for plotting (limit to reasonable number for figures)
			plot_pairs = min(4, top_n_pairs)  # Limit to 4 pairs for readability
			alpha_pairs_to_plot = gene_results.index.get_level_values(0).unique()[:plot_pairs]
			
			self.save_combined_gene_figures(alpha_pairs_to_plot)
			self.save_combined_origin_figures(alpha_pairs_to_plot)
		
		results = {
			'gene_results': gene_results,
			'origin_results': origin_results,
			'n_pairs_analyzed': top_n_pairs,
			'copy_correct': copy_correct
		}
		
		if self.verbose:
			print_fl("Full combined analysis completed successfully")
		
		return results


	def load_combined_gene_results(self, filename=None):
		"""
		Load combined gene results from CSV file.
		
		Parameters:
		-----------
		filename : str, optional
			Path to CSV file. If None, uses default path.
		"""
		if filename is None:
			filename = f"{self.save_dir}/combined_gene_results.csv"
		
		self.combined_gene_results = pd.read_csv(filename).set_index(['alphas', 'gene'])
		# Convert column names to integers if they represent timepoints
		try:
			self.combined_gene_results.columns = self.combined_gene_results.columns.astype(int)
		except (ValueError, TypeError):
			pass  # Keep original column names if conversion fails
		
		if self.verbose:
			print_fl(f"Combined gene results loaded from {filename}")
		
		return self.combined_gene_results


	def load_combined_origin_results(self, filename=None):
		"""
		Load combined origin results from CSV file.
		
		Parameters:
		-----------
		filename : str, optional
			Path to CSV file. If None, uses default path.
		"""
		if filename is None:
			filename = f"{self.save_dir}/combined_origin_results.csv"
		
		self.combined_origin_results = pd.read_csv(filename).set_index(['alphas', 'origin_id'])
		# Convert column names to integers if they represent timepoints
		try:
			self.combined_origin_results.columns = self.combined_origin_results.columns.astype(int)
		except (ValueError, TypeError):
			pass  # Keep original column names if conversion fails
		
		if self.verbose:
			print_fl(f"Combined origin results loaded from {filename}")
		
		return self.combined_origin_results