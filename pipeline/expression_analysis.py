
# Let's analyze the first set of genes that have been deconvolved. Create a volcano plot
# get get an understanding of the mother daughter differences

from src.geneset import get_deconvolved_geneset
from glob import glob
from src.sgd import get_gene_name_orf_name
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from src.config import load_default_expression_configs


class ExpressionAnalysis(object):
	"""docstring for ExpressionAnalysis"""
	def __init__(self, output_directory):
		self.output_directory = output_directory
		self.deconvolved_genes_F = load_deconvolved_gene_expression(self.output_directory)
		self.config1, self.config2 = load_default_expression_configs()


	def plot_volcano_cg1_dg1(self, genes_callout=['DSE1', 'DSE2', 'DSE3', 'DSE4', 'SIC1', 'CLN3', 'SSK22',
			'CBF1']):

		self.cg1_dg1_data = plot_volcano_cg1_dg1(self.deconvolved_genes_F, self.config1,
										   genes_callout=genes_callout)
		plt.suptitle("Mother vs Daughter\nexpressed genes")

		thresh_x_min, thresh_y_min = 0.25, 20

		thresholded_data = filter_threshold_genes(self.cg1_dg1_data, cg1_dg1_boundary_func,
			min_y=thresh_y_min, min_x=thresh_x_min)

		# create boundaries to separate, DG1, CG1 and no difference genes
		xs = np.linspace(0.0001, 4, 10000)
		ys = cg1_dg1_boundary_func(xs, thresh_y_min, thresh_x_min)

		plt.plot(-xs, ys, c='red', lw=0.75, ls='solid')
		plt.plot(xs, ys, c='red', lw=0.75, ls='solid')
		plt.xlim(-4, 4)

		self.dg1_dat = thresholded_data.loc[thresholded_data.meets_dg1_threshold]
		self.cg1_dat = thresholded_data.loc[thresholded_data.meets_cg1_threshold]
		self.control_dat = thresholded_data.loc[(~thresholded_data.meets_dg1_threshold) &
										   (~thresholded_data.meets_cg1_threshold)]

	def retrieve_mg1_dg1_specific_genes(self, config1, proportion_threshold):
		"""Retrieve G1 specific expression genes using a threshold. Separates into
		DG1 only, MG1 only and DG1 and MG1 expressed genes."""

		expressions_F = self.deconvolved_genes_F

		# Compute the number of MG1, DG1 specific genes and G1 specific genes.
		mg1_i = config1.cg1_indices()
		dg1_i = config1.dg1_indices()
		postg1_i = config1.postg1_indices()

		mg1_expression = expressions_F[mg1_i]
		dg1_expression = expressions_F[dg1_i]

		mean_mg1_expression = mg1_expression.mean(1)
		mean_dg1_expression = dg1_expression.mean(1)
		mean_postg1_expression = expressions_F[postg1_i].mean(1)

		orfs = expressions_F.index

		# How many genes have g1 expression mg1 union dg1
		mg1_select = mean_mg1_expression > mean_postg1_expression*(1+proportion_threshold)
		dg1_select = mean_dg1_expression > mean_postg1_expression*(1+proportion_threshold)

		# M and D have similar expression levels, use the l2 norm and set to threshold
		# these to a low value, can look at the histogram to decide on the threshold
		cg1_dg1_l2 = (((mg1_expression.values-dg1_expression.values)**2).sum(1)**0.5)
		mg1_and_dg1_similar_expression = cg1_dg1_l2 < 1

		# MG1 and DG1 are similarly expressed and G1-specific
		mg1_and_dg1_select = mg1_select & dg1_select & mg1_and_dg1_similar_expression
		mg1_and_dg1 = orfs[mg1_and_dg1_select]

		# MG1 is transcribed
		mg1_only_select = mg1_select & ~mg1_and_dg1_select & ~dg1_select
		mg1_only = orfs[mg1_only_select]

		# DG1 is transcribed
		dg1_only_select = dg1_select & ~mg1_and_dg1_select & ~mg1_select
		dg1_only = orfs[dg1_only_select]

		# How many genes have g1 expression mg1 union dg1
		postg1_select = (mean_mg1_expression*(1+proportion_threshold) < mean_postg1_expression) & \
						(mean_dg1_expression*(1+proportion_threshold) < mean_postg1_expression)
		postg1_orfs = orfs[postg1_select & ~mg1_and_dg1_select & ~mg1_only_select & ~dg1_only_select]

		# Unchanging orfs
		unchanging_orfs = orfs[~postg1_select & ~mg1_and_dg1_select &\
							   ~mg1_only_select & ~dg1_only_select]

		print(f"Using a threshold of {1+proportion_threshold}")

		print(f"  {len(postg1_orfs)} post G1 expressed genes")
		
		print(f"  {len(mg1_only)} Mother G1 specific genes")
		print(f"  {len(dg1_only)} Daughter G1 specific genes")
		print(f"  {len(mg1_and_dg1)} Mother and Daughter expressed genes")
		print(f"  {len(unchanging_orfs)} All others")
		print(f"  Total:  {len(postg1_orfs)+len(mg1_only)+len(dg1_only)+len(mg1_and_dg1)+len(unchanging_orfs)}")

		return unchanging_orfs, mg1_only, mg1_and_dg1, dg1_only,  postg1_orfs

def plot_expression_quantiles(config, expressions_df, gene_subsets, t_indices, b_indices, 
                              normalize=False, figsize=(20, 6)):
    """
    Plot expression profiles with median and 25-75% quantiles for multiple gene sets.
    
    Parameters:
    -----------
    config : object
        Configuration object with method get_timepoints_for_branch
    expressions_df : pandas.DataFrame
        DataFrame containing gene expression data
    gene_subsets : list of tuples
        List of (gene_set, title) tuples where gene_set contains gene identifiers
    t_indices : list
        Column indices for top branch
    b_indices : list
        Column indices for bottom branch
    normalize : bool, default=False
        If True, normalize each gene to have mean=0 and std=1 before plotting
    figsize : tuple
        Figure size (width, height)
    """
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


def load_deconvolved_gene_expression(output_directory):
	genes = get_deconvolved_geneset()

	deconvolved_expression_filenames = glob(f"{output_directory}/genes_deconvolution/*.npy")

	gene_expression_Fs_list = []
	gene_names = []
	for i, filename in enumerate(deconvolved_expression_filenames):
		expression_F = np.load(filename)
		gene_name = filename.split('/')[-1].split('_')[0]
		orf_name, gene_name = get_gene_name_orf_name(gene_name)        
		gene_expression_Fs_list.append(expression_F)
		gene_names.append(orf_name)

	expression_Fs_df = pd.DataFrame(gene_expression_Fs_list, index=gene_names)

	return expression_Fs_df


def plot_volcano_cg1_dg1(expression_Fs_df, config1, genes_callout=[]):

	from scipy.stats.distributions import norm

	# Unlog the expression data for plotting
	average_TPM = 2**np.mean(expression_Fs_df, axis=1)
	cg1_dg1_max_ratio = np.log2(expression_Fs_df[config1.cg1_indices()].mean(axis=1) / \
		expression_Fs_df[config1.dg1_indices()].mean(axis=1))

	plot_data = pd.DataFrame({'average_TPM': average_TPM, 
		'max_ratio': cg1_dg1_max_ratio+ norm.rvs(0, 0.002, len(average_TPM))})

	from src.marginal_scatter_plot import ScatterChromatinPlot
	marginal_scatter_plot = ScatterChromatinPlot()

	from src.plot_helpers import create_sub_colormap
	cmap = create_sub_colormap('Purples', 0.25, 1., 'Purples_darker')

	fig = marginal_scatter_plot.plot(
		dat=plot_data,
		x_key='max_ratio',
		y_key='average_TPM',
		highlight_genes=genes_callout,
		xlim=(-4, 4),
		ylim=(-10, 600),
		orf_groups=[],
		plot_fit=False,
		cmap=cmap,
		bw=[0.2, 0.03],
		xlabel="$\\log_2$ [ CG1 occupancy / DG1 occupancy ]",
		ylabel="Average deconvolved, TPM"
	)

	return plot_data


def cg1_dg1_boundary_func(xs, y_offset=10, x_offset=0.5,
	scale=0.2, multiplier=100):
	"""Boundary function to place threshold on cg1/dg1 analysis plot"""

	from scipy.stats.distributions import gamma
	
	ys = gamma.pdf(xs-x_offset, 1, loc=0, scale=scale)*multiplier+y_offset
	ys[xs < +x_offset] = 1e9 # If less than the offset, set to some max value

	return ys


def filter_threshold_genes(cg1_dg1_data, cg1_dg1_boundary_func, min_y=10, min_x=0.5):
	xs = cg1_dg1_data.max_ratio

	neg_ratios = xs < 0
	pos_ratios = xs > 0

	neg_ys = cg1_dg1_boundary_func(-xs[neg_ratios], min_y, min_x)
	pos_ys = cg1_dg1_boundary_func(xs[pos_ratios], min_y, min_x)

	boundary_check = cg1_dg1_data.copy()
	boundary_check.loc[neg_ratios, 'boundary_y'] = neg_ys
	boundary_check.loc[pos_ratios, 'boundary_y'] = pos_ys

	boundary_check['meets_dg1_threshold'] = False
	boundary_check['meets_cg1_threshold'] = False

	boundary_check.loc[(neg_ratios) & (boundary_check.average_TPM > boundary_check.boundary_y), 
		'meets_dg1_threshold'] = True

	boundary_check.loc[(pos_ratios) & (boundary_check.average_TPM > boundary_check.boundary_y), 
		'meets_cg1_threshold'] = True

	boundary_check[boundary_check.meets_dg1_threshold]
	boundary_check[boundary_check.meets_cg1_threshold]
	return boundary_check

