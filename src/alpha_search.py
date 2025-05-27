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
		verbose=True):
		"""
		Initialize the FindAlphaSweepDS class.
		
		Parameters:
		-----------
		output_directory : str
			Directory where results will be saved
		gene_names : list, optional
			List of daughter-specific genes to analyze (default: DSE1-4)
		verbose : bool, optional
			Whether to print verbose output (default: True)
		"""
		self.output_directory = output_directory
		self.save_dir = f"{output_directory}/alpha_search"
		self.verbose = verbose
		self.timer = Timer()
		self.gene_names = gene_names
			
		# Load CLOCCS configs and initialize runner
		from pipeline.CombinedDeconvolveGeneExpressionRunner import CombinedDeconvolveGeneExpressionRunner
		self.config1, self.config2 = load_cloccs_configs()
		self.runner = CombinedDeconvolveGeneExpressionRunner(output_directory, False)

		# Create output directory
		mkdir_safe(self.save_dir)
		
		# Initialize results dataframes
		self.all_alpha_results_df = None
		self.dg1_mg1_ratios = None
		self.correlation_results = None
		self.thresholded_ratios = None

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
		
		fig, ax = plt.subplots(figsize=(5, 4.5))
		
		# Display the heatmap
		im = ax.imshow(correlation_pivot, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1,
					   origin='lower',
					   extent=[correlation_pivot.columns[0]-1, correlation_pivot.columns[-1]+1, 
							   correlation_pivot.index[0]-1, correlation_pivot.index[-1]+1])
		
		# Set the ticks and labels
		ax.set_xticks(correlation_pivot.columns)
		ax.set_yticks(correlation_pivot.index)
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
		fig1 = plot_alpha_genes_subplots(self.config1, alphas, self.all_alpha_results_df, 1)

		if save:
			self.all_alpha_results_df.to_csv(f"{self.save_dir}/alphas_repl1.csv")
			save_figure_for_paper(f"{self.save_dir}/alphas_repl1.png", dpi=80)
		
		# Plot and save for replicate 2
		fig2 = plot_alpha_genes_subplots(self.config2, alphas, self.all_alpha_results_df, 2)

		if save:
			self.all_alpha_results_df.to_csv(f"{self.save_dir}/alphas_repl2.csv")
			save_figure_for_paper(f"{self.save_dir}/alphas_repl2.png", dpi=80)
		
		if self.verbose:
			print_fl(f"Results saved to {self.save_dir}")

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


def plot_alpha_genes_subplots(config, plot_alphas, all_alpha_results_df, replicate):
	"""
	Plot data for a specific replicate showing values for t_indices and b_indices for each alpha and gene combination.
	
	Parameters:
	-----------
	config : object
		Configuration object with methods get_Hpositions_for_branch() to get top and bottom branch indices
	plot_alphas : array-like
		Alpha values to plot
	all_alpha_results_df : pandas.DataFrame
		DataFrame with MultiIndex ['replicate', 'alpha', 'gene'] and numeric value columns
	replicate : int
		The replicate number to plot
	
	Returns:
	--------
	matplotlib.figure.Figure
		The figure containing all subplots
	"""
	
	# Select data for the specified replicate using loc
	replicate_data = all_alpha_results_df.loc[replicate]
	
	# Get unique alpha values and genes from the index
	alphas = plot_alphas
	genes = sorted(replicate_data.index.get_level_values('gene').unique())
	
	# Create a figure with subplots for each alpha-gene combination plus averages
	n_alphas = len(plot_alphas)
	n_genes = len(genes)
	
	# Create a grid of subplots
	fig, axes = plt.subplots(n_alphas, n_genes + 1, figsize=(3. * (n_genes + 1), 2. * n_alphas))
	
	# Make sure axes is 2D for consistent indexing
	if n_alphas == 1 and n_genes == 1:
		axes = np.array([[axes[0], axes[1]]])
	elif n_alphas == 1:
		axes = axes.reshape(1, -1)
	elif n_genes == 1:
		axes = axes.reshape(-1, 2)

	# Make all the ylims the same and set colors
	all_data_no_nans = replicate_data.copy().fillna(0)
	ylims = 0, all_data_no_nans.values.max()*1.2

	# Process each alpha value
	for i, alpha in enumerate(alphas):
		config.modify_alpha(alpha)

		# Timepoints for branch
		i_tps = config.get_timepoints_for_branch('i')
		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')

		# Get indices for branches
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		# Get data for this alpha
		alpha_data = replicate_data.xs(alpha, level='alpha')
		
		# Store for averaging
		all_i_values = []
		all_t_values = []
		all_b_values = []

		i_color = '#a0a0a0'
		t_color = 'black'
		b_color = color_for_key('DG1')

		# Process each gene
		for j, gene in enumerate(genes):
			# Get the row for this gene
			gene_data = alpha_data.loc[gene]
			
			# Select the columns for branches
			i_values = gene_data[i_indices].values
			t_values = gene_data[t_indices].values
			b_values = gene_data[b_indices].values
			
			# Store for averaging
			all_i_values.append(i_values)
			all_t_values.append(t_values)
			all_b_values.append(b_values)
			
			# Plot on the corresponding subplot
			ax = axes[i, j]
			ax.plot(i_tps, i_values, lw=3, color=i_color, label='Recovery Branch')
			ax.plot(t_tps, t_values, lw=3, color=t_color, label='Top Branch')
			ax.plot(b_tps, b_values, lw=3, color=b_color, label='Bottom Branch')
			
			if j == 0:
				ax.set_ylabel(f"α={alpha}", fontsize=24)
			else:
				ax.set_yticks([])

			if i == 0:
				ax.set_title('$\\it{' + gene + '}$', fontsize=24, pad=5)

			if i == n_alphas-1:
				if i == 0:
					ax.set_xlabel('Average single cell time, min', fontsize=16)

			ax.set_ylim(*ylims)
			ax.axvline(0, c='#ddd', lw=1)
			ax.legend()
		
		# Calculate and plot averages across genes
		if all_t_values and all_b_values:
			# Calculate average for each position
			avg_i_values = np.mean(all_i_values, axis=0)
			avg_t_values = np.mean(all_t_values, axis=0)
			avg_b_values = np.mean(all_b_values, axis=0)
			
			# Plot averages
			ax = axes[i, n_genes]
			ax.plot(i_tps, avg_i_values, lw=5, color=i_color, label='Recovery Branch')
			ax.plot(t_tps, avg_t_values, lw=5, color=t_color, label='Top Branch')
			ax.plot(b_tps, avg_b_values, lw=5, color=b_color, label='Bottom Branch')
			ax.set_yticks([])
			ax.axvline(0, c='#ddd', lw=1)
			ax.set_ylim(*ylims)
			
			# Set title and labels
			ax.set_title(f'Average', fontsize=16, pad=5)
			ax.legend()
	
	# Adjust layout
	plt.tight_layout(rect=[0, 0, 1, 0.96])
	
	# Set overall title
	fig.suptitle(f'Alpha search, replicate {replicate}', fontsize=32, fontweight='demi')
	
	return fig