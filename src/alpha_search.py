import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.plot_helpers import color_for_key


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
		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')

		# Get t_indices and b_indices for top and bottom branches
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		# Get data for this alpha using xs (cross-section)
		alpha_data = replicate_data.xs(alpha, level='alpha')
		
		# Store for averaging
		all_t_values = []
		all_b_values = []

		t_color = 'black'
		b_color = color_for_key('DG1')

		# Process each gene
		for j, gene in enumerate(genes):

			# Get the row for this gene
			gene_data = alpha_data.loc[gene]
			
			# Directly select the columns for top and bottom branches
			t_values = gene_data[t_indices].values
			b_values = gene_data[b_indices].values
			
			# Store for averaging
			all_t_values.append(t_values)
			all_b_values.append(b_values)
			
			# Plot on the corresponding subplot
			ax = axes[i, j]
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
			# Determine maximum length for each branch
			max_t_len = max([len(vals) for vals in all_t_values]) if all_t_values else 0
			max_b_len = max([len(vals) for vals in all_b_values]) if all_b_values else 0
			
			# Create arrays for averaging, filling with NaN for missing values
			t_array = np.full((len(all_t_values), max_t_len), np.nan)
			b_array = np.full((len(all_b_values), max_b_len), np.nan)
			
			# Fill the arrays with actual values
			for k, vals in enumerate(all_t_values):
				t_array[k, :len(vals)] = vals
			
			for k, vals in enumerate(all_b_values):
				b_array[k, :len(vals)] = vals
			
			# Calculate average for each position (ignoring NaN values)
			avg_t_values = np.nanmean(t_array, axis=0)
			avg_b_values = np.nanmean(b_array, axis=0)
			
			# Clean up NaNs for plotting
			avg_t_values = avg_t_values[~np.isnan(avg_t_values)]
			avg_b_values = avg_b_values[~np.isnan(avg_b_values)]
			
			# Plot averages
			ax = axes[i, n_genes]
			ax.plot(t_tps, avg_t_values, lw=5, color=t_color, label='Top Branch')
			ax.plot(b_tps, avg_b_values, lw=5, color=b_color, label='Bottom Branch')
			ax.set_yticks([])
			ax.axvline(0, c='#ddd', lw=1)
			
			# Set title and labels
			ax.set_title(f'Average', fontsize=16, pad=5)
			
			# Add legend
			ax.legend()
	
	# Adjust layout
	plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

	# Set overall title
	fig.suptitle(f'Alpha search, replicate {replicate}', fontsize=32, fontweight='demi')
	
	return fig


