
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from src.plot_helpers import color_for_key


def create_subplot_grid(fig_size=(4, 4), main_size=(0.9, 0.9), 
						margin=0.05, cbar_ax=True):
	"""
	Create a custom subplot grid with a main plot and three marginal plots
	(one on the left, one on top, and one on the right).
	"""

	left_padding = -0.05
	main_right_padding = 0.05

	# Create figure
	fig = plt.figure(figsize=fig_size)
	
	# Calculate sizes
	main_width, main_height = main_size
	top_height = 1 - main_height - margin
	left_width = 1 - main_width - margin

	if cbar_ax:
		right_width = left_width  # Make right width same as left width
	else:
		right_width = 0
		main_right_padding = 0
	
	# Calculate positions
	left_pos = margin
	bottom_pos = margin
	
	# Create axes with specific positions and sizes
	# Format: [left, bottom, width, height]
	main_ax = plt.axes([left_padding+left_pos + left_width, bottom_pos, main_width, main_height])
	left_ax = plt.axes([left_padding+margin, bottom_pos, left_width, main_height])
	top_ax = plt.axes([left_padding+left_pos + left_width, bottom_pos + main_height, main_width, top_height])
	
	# Add the right axis
	right_ax = plt.axes([left_padding+left_pos + left_width + main_width+main_right_padding, bottom_pos, right_width, main_height])
	
	# Remove ticks from marginal plots
	for ax in [left_ax, top_ax, right_ax, main_ax]:
		ax.set_xticks([])
		ax.set_yticks([])

	top_ax.xaxis.set_label_position('top')
	
	# Store axes in a dictionary for easy access
	axes = {'main': main_ax, 'left': left_ax, 'top': top_ax, 'right': right_ax}
	
	# Remove spacing between subplots
	plt.subplots_adjust(wspace=0, hspace=0)
	
	return fig, axes


def plot_peak_idx_heatmap(matrix,
						 figsize=(4.5, 4), fig=None, ax=None, vmax=10,
						 cmap='YlGnBu', tick_interval=10):
	"""
	Plot a heatmap of the peak index co-occurrence matrix using pcolormesh.
	Shows the full matrix with proper axis labels and x-axis on top.
	"""
	
	if ax is None:
		fig, ax = plt.subplots(figsize=figsize)
	
	# Get unique x and y values
	y_values = matrix.index.tolist()
	x_values = matrix.columns.tolist()
	
	# Create the meshgrid for pcolormesh
	x_edges = np.arange(len(x_values) + 1)
	y_edges = np.arange(len(y_values) + 1)
	X, Y = np.meshgrid(x_edges, y_edges)
	
	# Convert matrix values to 2D array for pcolormesh
	Z = matrix.values.copy()
	
	# Set up color normalization
	norm = colors.Normalize(vmin=0, vmax=vmax)
	
	# Create the pcolormesh with origin at top-left
	mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
	
	# Set the axis limits
	ax.set_xlim(0, len(x_values))
	ax.set_ylim(len(y_values), 0)  # Reversed to have origin at top-left

	ax.set_xticks([])
	ax.set_yticks([])
	
	return fig, ax, mesh


def create_peak_idx_co_occurrence_matrix(df, bin_key, tp_key, min_idx=0, max_idx=63):
	"""
	Create a co-occurrence matrix from gene peak indices.
	"""

	tx_col = bin_key + '_tx'
	small_col = bin_key + '_small'

	# Create a copy of the dataframe to avoid modifying the original
	df_copy = df.copy()
	
	# Replace NaN with a special value (e.g., 'NaN') in both columns
	df_copy[tx_col] = df_copy[tx_col].fillna('NaN')
	df_copy[small_col] = df_copy[small_col].fillna('NaN')
	
	# Create a cross-tabulation (this is essentially our co-occurrence matrix)
	matrix = pd.crosstab(df_copy[tx_col], df_copy[small_col])
	
	# Create the full range of numeric indices
	all_numeric_indices = list(range(min_idx, max_idx + 1))
	
	# For rows, use numeric indices and add 'NaN' if it exists
	all_row_indices = all_numeric_indices.copy()
	if 'NaN' in matrix.index:
		all_row_indices.append('NaN')
	
	# For columns, use numeric indices and add 'NaN' if it exists
	all_col_indices = all_numeric_indices.copy()
	if 'NaN' in matrix.columns:
		all_col_indices.append('NaN')
	
	# Create a new DataFrame with all indices and fill with zeros
	full_matrix = pd.DataFrame(0, index=all_row_indices, columns=all_col_indices)
	
	# Fill in the actual counts from the original matrix
	for row in matrix.index:
		for col in matrix.columns:
			full_matrix.loc[row, col] = matrix.loc[row, col]
	
	# Move 'NaN' to the end in both dimensions if they exist
	if 'NaN' in full_matrix.index:
		# Get the NaN row
		nan_row = full_matrix.loc['NaN'].copy()
		# Drop the NaN row
		full_matrix = full_matrix.drop('NaN')
		# Add it back at the end
		full_matrix.loc['NaN'] = nan_row
	
	if 'NaN' in full_matrix.columns:
		# Get the NaN column
		nan_col = full_matrix['NaN'].copy()
		# Drop the NaN column
		full_matrix = full_matrix.drop('NaN', axis=1)
		# Add it back at the end
		full_matrix['NaN'] = nan_col
	
	return full_matrix


def count_peak_idx_by_bin(df, bins_to_include, bin_key, tp_key,
						 min_idx=0, max_idx=64):
	"""
	Count the number of entries per peak_idx for specified bins.
	"""

	# Filter the dataframe to include only the specified bins
	filtered_df = df[df[bin_key].isin(bins_to_include)]
	
	# Group by peak_idx and bin, then count
	counts = filtered_df.groupby([tp_key]).size()
	
	for idx in np.arange(0, max_idx):
		if idx not in counts.index:
			counts[idx] = 0

	return filtered_df[[tp_key]]


def plot_cooccurences(expression_combined_polar_data_df, small_combined_polar_data_df,
	branch='mother', metric='metric'):

	if branch == 'mother':
		bins_to_include = ["MG1", "postG1"]
		g1_key = 'MG1'
		bin_key = 'bin_t'
		tp_key = 'peak_idx_t'
	else:
		g1_key = 'DG1'
		bins_to_include = ["DG1", "postG1"]
		bin_key = 'bin_b'
		tp_key = 'peak_idx_b'

	expression_peak_counts = count_peak_idx_by_bin(expression_combined_polar_data_df, bins_to_include,
		bin_key, tp_key)
	small_peak_counts = count_peak_idx_by_bin(small_combined_polar_data_df, bins_to_include,
		bin_key, tp_key)
	joined_counts_data = expression_peak_counts.join(small_peak_counts, lsuffix='_tx', 
		rsuffix='_small', how='outer')

	counts_mat = create_peak_idx_co_occurrence_matrix(joined_counts_data, tp_key, bin_key)

	fig, axes = create_subplot_grid()

	ax = axes['main']

	fig, ax, mesh = plot_peak_idx_heatmap(counts_mat, cmap='YlGnBu', fig=fig, ax=ax, vmax=3)
	ax.axvline(42, c='black', lw=0.25)
	ax.axhline(42, c='black', lw=0.25)
	ax.axvline(64, c='black', lw=0.25)
	ax.axhline(64, c='black', lw=0.25)

	def retrieve_quadrant_counts(counts_mat):
		"""Get the counts per quadrant"""
		quadrant_counts = df = pd.DataFrame([
			(0, 42, 0, 42, 0),
			(42, 64, 0, 42, 0),
			(0, 42, 42, 64, 0),
			(42, 64, 42, 64, 0),
			
		], columns=['row_start', 'row_end', 'col_start', 'col_end', 'count'])

		for idx, entry in quadrant_counts.iterrows():
			count = counts_mat.iloc[entry.row_start:entry.row_end, 
							entry.col_start:entry.col_end].sum().sum()
			entry['count'] = count

		quadrant_counts['text_y'] = (quadrant_counts.row_start+quadrant_counts.row_end)/2
		quadrant_counts['text_x'] = (quadrant_counts.col_start+quadrant_counts.col_end)/2
		return quadrant_counts

	quadrant_counts = retrieve_quadrant_counts(counts_mat)

	for idx, entry in quadrant_counts.iterrows():
		ax.text(entry.text_x, entry.text_y, f"{entry['count']:.0f}",
			fontfamily='Open Sans', fontsize=12, fontweight='demi', color='black', alpha=0.25)

	# Plot annotations along top and left

	from src.plot_helpers import color_for_key

	g1_color = color_for_key(g1_key)
	postg1_color = color_for_key('postG1')

	annotate_top_ax(axes['top'])
	annotate_left_ax(axes['left'])

	right_ax = axes['right']

	# Add a colorbar
	cbar = fig.colorbar(mesh, cax=right_ax)
	cbar.set_label('Gene Count', rotation=270, va='bottom', labelpad=0)

	yticks = [0, 1, 2, 3]
	yticklabels = [str(y) for y in yticks]
	yticklabels[-1] = yticklabels[-1]+"+"
	cbar.ax.set_yticks(yticks)
	cbar.ax.set_yticklabels(yticklabels)

	plt.suptitle(f"{branch.title()} branch expression/{metric} timing", y=1.12)

	return fig, joined_counts_data, counts_mat

def _add_annotation_text(ax, x, y, text, rotate=False):
	rotation = 90 if rotate else 0
	ax.text(x, y, text, rotation=rotation, va='center', ha='center', color='white',
				fontfamily='Open Sans', fontweight='demi')

def annotate_top_ax(top_ax, label="Peak time", g1_key='CG1'):
	top_ax.set_xlim(0, 64.5)
	top_ax.set_ylim(0, 1)

	g1_color = color_for_key(g1_key)
	postg1_color = color_for_key('postG1')

	top_ax.fill_between([0, 41.5], [1, 1], [0, 0], color=g1_color, linewidth=0)
	top_ax.fill_between([41.5, 63.5], [1, 1], [0, 0], color=postg1_color, linewidth=0)
	top_ax.fill_between([63.5, 64.5], [1, 1], [0, 0], color='#555', linewidth=0)
	top_ax.set_xlabel(label, labelpad=5)

	g1_name = g1_key.replace("D", ' Daughter ').replace("M", " Mother")

	_add_annotation_text(top_ax, 22, 0.45, g1_name)
	_add_annotation_text(top_ax, 52, 0.45, "S/G2/M")

def annotate_left_ax(left_ax, label='Peak expression time', g1_key='CG1'):
	left_ax.set_ylim(64.5, 0)
	left_ax.set_xlim(0, 1)

	g1_color = color_for_key(g1_key)
	postg1_color = color_for_key('postG1')

	g1_name = g1_key.replace("D", ' Daughter ').replace("M", " Mother")

	left_ax.fill_betweenx([0, 41.5], [1, 1], [0, 0], color=g1_color, linewidth=0)
	left_ax.fill_betweenx([41.5, 63.5], [1, 1], [0, 0], color=postg1_color, linewidth=0)
	left_ax.fill_betweenx([63.5, 64.5], [1, 1], [0, 0], color='#555', linewidth=0)
	left_ax.set_ylabel("Peak expression time", labelpad=5)

	_add_annotation_text(left_ax, 0.54, 22, g1_name, rotate=True)
	_add_annotation_text(left_ax, 0.54, 52, "S/G2/M", rotate=True)
