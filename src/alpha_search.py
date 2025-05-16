import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.plot_helpers import color_for_key
from src.timer import Timer
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper
from src.config import load_cloccs_configs


class FindAlphaSweepDS:
    """
    Class for performing alpha parameter sweeps for daughter-specific genes.
    
    This class helps determine the optimal alpha parameter (which controls the end of mitosis/start of G1 timing) 
    by running deconvolution with different alpha values on daughter-specific genes.
    """
    
    def __init__(self, output_directory, gene_names=None, verbose=False):
        """
        Initialize the FindAlphaSweepDS class.
        
        Parameters:
        -----------
        output_directory : str
            Directory where results will be saved
        gene_names : list, optional
            List of daughter-specific genes to analyze (default: DSE1-4)
        verbose : bool, optional
            Whether to print verbose output (default: False)
        """
        self.output_directory = output_directory
        self.save_dir = f"{output_directory}/alpha_search"
        self.verbose = verbose
        self.timer = Timer()
        
        # Default daughter-specific genes if none provided
        if gene_names is None:
            self.gene_names = ['DSE1', 'DSE2', 'DSE3', 'DSE4']
        else:
            self.gene_names = gene_names
            
        # Load CLOCCS configs and initialize runner
        from pipeline.CombinedDeconvolveGeneExpressionRunner import CombinedDeconvolveGeneExpressionRunner
        self.config1, self.config2 = load_cloccs_configs()
        self.runner = CombinedDeconvolveGeneExpressionRunner(output_directory, False)
        
        # Create output directory
        mkdir_safe(self.save_dir)
        
        # Initialize results dataframe
        self.all_alpha_results_df = None

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
            print(f"Searching replicate {replicate} alphas: {gene_name}...")

        for alpha in alphas:
            if self.verbose:
                print(alpha, end=",")
                
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
            print()

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
            alphas = np.arange(4, 49, 16)  # Default from notebook
            
        if self.verbose:
            print(f"Number of alphas: {len(alphas)}")
            print(f"Testing alpha values: {alphas}")
            print(f"Running for genes: {self.gene_names}")

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

    def plot_and_save_results(self, alphas=None):
        """
        Plot and save the results of the alpha sweep.
        
        Parameters:
        -----------
        alphas : array-like, optional
            Alpha values to plot. If None, uses all alphas in results.
        """
        if self.all_alpha_results_df is None:
            raise ValueError("No results to plot. Run 'run_alpha_sweep' first.")
            
        if alphas is None:
            # Extract unique alphas from the results
            alphas = self.all_alpha_results_df.reset_index()['alpha'].unique()

        # Plot and save for replicate 1
        fig1 = plot_alpha_genes_subplots(self.config1, alphas, self.all_alpha_results_df, 1)
        self.all_alpha_results_df.to_csv(f"{self.save_dir}/alphas_repl1.csv")
        save_figure_for_paper(f"{self.save_dir}/alphas_repl1.png", dpi=80)
        
        # Plot and save for replicate 2
        fig2 = plot_alpha_genes_subplots(self.config2, alphas, self.all_alpha_results_df, 2)
        self.all_alpha_results_df.to_csv(f"{self.save_dir}/alphas_repl2.csv")
        save_figure_for_paper(f"{self.save_dir}/alphas_repl2.png", dpi=80)
        
        if self.verbose:
            print(f"Results saved to {self.save_dir}")
            
    def determine_optimal_alpha(self, criterion='TBD'):
        """
        Determine the optimal alpha for each replicate based on specified criterion.
        
        Parameters:
        -----------
        criterion : str
            Criterion to use for determining optimal alpha (TBD)
            
        Returns:
        --------
        dict
            Dictionary with optimal alpha values for each replicate
        """
        # Placeholder for future implementation
        # This would analyze self.all_alpha_results_df to find optimal alpha
        # based on criteria like fit quality, smoothness, or biological relevance
        
        if self.verbose:
            print("Optimal alpha determination not yet implemented")
            print("Criterion will be: " + criterion)
            
        return {
            'replicate1': None,
            'replicate2': None
        }
        
    def run_full_analysis(self, alphas=None, kappa=0.0, criterion='TBD'):
        """
        Run the complete alpha sweep analysis workflow.
        
        Parameters:
        -----------
        alphas : array-like, optional
            Alpha values to test
        kappa : float, optional
            Kappa parameter for deconvolution
        criterion : str, optional
            Criterion for determining optimal alpha
            
        Returns:
        --------
        dict
            Dictionary with optimal alpha values for each replicate
        """
        # Run alpha sweep
        self.run_alpha_sweep(alphas=alphas, kappa=kappa)
        
        # Plot and save results
        self.plot_and_save_results(alphas=alphas)


def plot_alpha_genes_subplots(config, plot_alphas, all_alpha_results_df, replicate):
	"""
	Plot data for a specific replicate showing values for t_indices and b_indices for each alpha and gene combination.
	
	Parameters:
	-----------
	config1 : object
		Configuration object with methods get_Hpositions_for_branch() to get top and bottom branch indices
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


	"""make all the ylims the same
	color the average differently"""

	all_data_no_nans = replicate_data.copy().fillna(0)
	ylims = 0, all_data_no_nans.values.max()*1.2

	# Process each alpha value
	for i, alpha in enumerate(alphas):

		config.modify_alpha(alpha)

		# Timepoints for branch
		i_tps = config.get_timepoints_for_branch('i')
		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')

		# Get t_indices and b_indices for top and bottom branches
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		# Get data for this alpha using xs (cross-section)
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
			
			# Directly select the columns for top and bottom branches
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
					ax.set_xlabel('Average single cell time, min', 
						fontsize=16)

			ax.set_ylim(*ylims)
			ax.axvline(0, c='#ddd', lw=1)
			ax.legend()
		
		# Calculate and plot averages across genes
		if all_t_values and all_b_values:

			# Calculate average for each position (ignoring NaN values)
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
			
			# Add legend
			ax.legend()
	
	# Adjust layout
	plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

	# Set overall title
	fig.suptitle(f'Alpha search, replicate {replicate}', fontsize=32, fontweight='demi')
	
	return fig


