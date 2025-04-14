
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.plot_helpers import hide_spines
from src.plot_helpers import color_for_key
from src.config_utils import get_sample_indices
from src.orf_plotter import load_default_orf_plotter

import pandas as pd
import numpy as np

# Category colors for comparison plots

color_map = {
	"expression_only": '#755b96',
	"promoter_only": plt.cm.Spectral(0.25), 
	"entropy_only": plt.cm.Spectral(0.75), 
	"both": "#7a7a7a",
	"neither": "#d3d3d3"
}

text_color_map = {
	"expression_only": 'white',
	"promoter_only": 'black',
	"entropy_only": 'black',
	"both": 'white',
	"neither": 'black',
}


def categorize_cell_cycle_genes(expression_df, chromatin_df):
	"""
	Categorize genes based on their cell cycle activity in expression and chromatin data.
	
	Parameters:
	-----------
	expression_df : pandas.DataFrame
		DataFrame containing expression data with 'bin' column indicating cell cycle phase
	chromatin_df : pandas.DataFrame
		DataFrame containing chromatin data with 'bin' column indicating cell cycle phase
		
	Returns:
	--------
	dict
		Dictionary containing categorized ORF lists and a summary DataFrame
	"""
	# Define cell cycle categories
	cell_cycle_bins = ['MG1', 'DG1', 'postG1']
	
	# Create mask for cell cycle genes in each dataset
	expr_cell_cycle_mask = expression_df['bin'].isin(cell_cycle_bins)
	chrom_cell_cycle_mask = chromatin_df['bin'].isin(cell_cycle_bins)
	
	# Get unique ORFs from both datasets
	expr_orfs = set(expression_df.index)
	chrom_orfs = set(chromatin_df.index)
	all_orfs = expr_orfs.union(chrom_orfs)
	
	# Initialize results dictionary
	results = {
		'expression_only': [],
		'chromatin_only': [],
		'both': [],
		'either': [],
		'neither': [],
		'details': {}
	}
	
	# Create a dictionary to store detailed information for each ORF
	details = {}
	
	# Analyze each ORF
	for orf in all_orfs:
		expr_status = "Not found"
		chrom_status = "Not found"
		
		# Check if ORF is in expression dataset
		if orf in expr_orfs:
			if expr_cell_cycle_mask.loc[orf]:
				expr_status = expression_df.loc[orf, 'bin']
			else:
				expr_status = "Not cell cycle"
		
		# Check if ORF is in chromatin dataset
		if orf in chrom_orfs:
			if chrom_cell_cycle_mask.loc[orf]:
				chrom_status = chromatin_df.loc[orf, 'bin']
			else:
				chrom_status = "Not cell cycle"
		
		# Store detailed information
		details[orf] = {
			'expression': expr_status,
			'chromatin': chrom_status
		}
		
		# Categorize the ORF
		is_expr_cycling = expr_status in cell_cycle_bins
		is_chrom_cycling = chrom_status in cell_cycle_bins
		
		if is_expr_cycling and is_chrom_cycling:
			results['both'].append(orf)
		elif is_expr_cycling:
			results['expression_only'].append(orf)
		elif is_chrom_cycling:
			results['chromatin_only'].append(orf)
		else:
			results['neither'].append(orf)
		
		if is_expr_cycling or is_chrom_cycling:
			results['either'].append(orf)
	
	# Convert details to DataFrame for easy analysis
	results['details'] = pd.DataFrame(details).T
	results['details'].index.name = 'ORF'
	
	# Generate summary statistics
	results['summary'] = {
		'total_orfs': len(all_orfs),
		'expression_only': len(results['expression_only']),
		'chromatin_only': len(results['chromatin_only']),
		'both': len(results['both']),
		'either': len(results['either']),
		'neither': len(results['neither'])
	}
	
	return results



def categorize_genes(expression_df, chromatin_df, suffix='_t',
	cat_1_name="expression_only", cat_2_name="chromatin_only"):
	"""
	Categorize genes based on their cell cycle classification in both datasets.
	
	Parameters:
	-----------
	expression_df : pandas.DataFrame
		Expression dataset with phase column
	chromatin_df : pandas.DataFrame
		Chromatin dataset with phase column
	suffix : str
		Suffix to identify mother (_t) or daughter (_b) data
		
	Returns:
	--------
	dict
		Dictionary with counts for each category
	"""
	phase_col = f'bin{suffix}'
	
	# Create merged dataframe with only the relevant columns
	merged_df = expression_df[[phase_col]].join(chromatin_df[[phase_col]], 
		how='outer',
		lsuffix=f'_{cat_1_name}',
		rsuffix=f'_{cat_2_name}')
	
	# Create flags for cell cycle vs. not cell cycle
	is_cat_1_key = f'is_{cat_1_name}'
	is_cat_2_key = f'is_{cat_2_name}'
	merged_df[is_cat_1_key] = merged_df[f'{phase_col}_{cat_1_name}'] != 'Not cell cycle'
	merged_df[is_cat_2_key] = merged_df[f'{phase_col}_{cat_2_name}'] != 'Not cell cycle'

	# Handle misclassification
	merged_df = merged_df.fillna('Not cell cycle')
	
	# Count genes in each category
	cat_1_only = sum(merged_df[is_cat_1_key] & ~merged_df[is_cat_2_key])
	cat_2_only = sum(~merged_df[is_cat_1_key] & merged_df[is_cat_2_key])
	both = sum(merged_df[is_cat_1_key] & merged_df[is_cat_2_key])
	neither = sum(~merged_df[is_cat_1_key] & ~merged_df[is_cat_2_key])

	return merged_df, {
		cat_1_name: cat_1_only,
		cat_2_name: cat_2_only,
		'both': both,
		'neither': neither
	}


# Categorize genes into three bins
def categorize_genes3(expression_df, promoter_df, entropy_df, suffix='_t',
					 cat_1_name="expression_only", cat_2_name="promoter_only", cat_3_name="entropy_only"):
	"""
	Categorize genes based on their cell cycle classification in three datasets.
	"""
	phase_col = f'bin{suffix}'
	
	# Create merged dataframe with only the relevant columns
	# First merge expression and chromatin
	merged_df = expression_df[[phase_col]].join(
		promoter_df[[phase_col]], 
		how='outer',
		lsuffix=f'_{cat_1_name}',
		rsuffix=f'_{cat_2_name}'
	)
	
	# Then merge with entropy
	merged_df = merged_df.join(
		entropy_df[[phase_col]],
		how='outer',
		rsuffix=f'_{cat_3_name}'
	)
	
	# Rename the column from the first dataframe that didn't get a suffix
	if f'{phase_col}_{cat_3_name}' not in merged_df.columns:
		merged_df = merged_df.rename(columns={phase_col: f'{phase_col}_{cat_3_name}'})
	
	# Create flags for cell cycle vs. not cell cycle
	is_cat_1_key = f'is_{cat_1_name}'
	is_cat_2_key = f'is_{cat_2_name}'
	is_cat_3_key = f'is_{cat_3_name}'
	
	merged_df[is_cat_1_key] = merged_df[f'{phase_col}_{cat_1_name}'] != 'Not cell cycle'
	merged_df[is_cat_2_key] = merged_df[f'{phase_col}_{cat_2_name}'] != 'Not cell cycle'
	merged_df[is_cat_3_key] = merged_df[f'{phase_col}_{cat_3_name}'] != 'Not cell cycle'

	# Handle misclassification
	merged_df = merged_df.fillna('Not cell cycle')
	
	# Count genes in each individual category (including overlaps)
	cat_1_total = sum(merged_df[is_cat_1_key])
	cat_2_total = sum(merged_df[is_cat_2_key])
	cat_3_total = sum(merged_df[is_cat_3_key])
	
	# Count genes in pairwise overlaps
	cat_1_2_overlap = sum(merged_df[is_cat_1_key] & merged_df[is_cat_2_key])
	cat_1_3_overlap = sum(merged_df[is_cat_1_key] & merged_df[is_cat_3_key])
	cat_2_3_overlap = sum(merged_df[is_cat_2_key] & merged_df[is_cat_3_key])
	
	# Count genes in triple overlap
	cat_1_2_3_overlap = sum(merged_df[is_cat_1_key] & merged_df[is_cat_2_key] & merged_df[is_cat_3_key])
	
	# Count genes in exclusive categories (only in one category)
	cat_1_only = sum(merged_df[is_cat_1_key] & ~merged_df[is_cat_2_key] & ~merged_df[is_cat_3_key])
	cat_2_only = sum(~merged_df[is_cat_1_key] & merged_df[is_cat_2_key] & ~merged_df[is_cat_3_key])
	cat_3_only = sum(~merged_df[is_cat_1_key] & ~merged_df[is_cat_2_key] & merged_df[is_cat_3_key])
	
	# Count genes in exactly two categories
	cat_1_2_only = sum(merged_df[is_cat_1_key] & merged_df[is_cat_2_key] & ~merged_df[is_cat_3_key])
	cat_1_3_only = sum(merged_df[is_cat_1_key] & ~merged_df[is_cat_2_key] & merged_df[is_cat_3_key])
	cat_2_3_only = sum(~merged_df[is_cat_1_key] & merged_df[is_cat_2_key] & merged_df[is_cat_3_key])
	
	# Count genes in none of the categories
	none = sum(~merged_df[is_cat_1_key] & ~merged_df[is_cat_2_key] & ~merged_df[is_cat_3_key])

	# Create results dictionary compatible with the three-set Venn diagram function
	results = {
		# Individual categories (totals)
		cat_1_name: cat_1_total,
		cat_2_name: cat_2_total,
		cat_3_name: cat_3_total,
		
		# Pairwise overlaps
		f"{cat_1_name}_{cat_2_name}": cat_1_2_overlap,
		f"{cat_1_name}_{cat_3_name}": cat_1_3_overlap,
		f"{cat_2_name}_{cat_3_name}": cat_2_3_overlap,
		
		# Triple overlap
		f"{cat_1_name}_{cat_2_name}_{cat_3_name}": cat_1_2_3_overlap,
		
		# None category
		'none': none,
		
		# For convenience, also include exclusive counts
		f"{cat_1_name}_exclusive": cat_1_only,
		f"{cat_2_name}_exclusive": cat_2_only,
		f"{cat_3_name}_exclusive": cat_3_only,
		f"{cat_1_name}_{cat_2_name}_exclusive": cat_1_2_only,
		f"{cat_1_name}_{cat_3_name}_exclusive": cat_1_3_only,
		f"{cat_2_name}_{cat_3_name}_exclusive": cat_2_3_only
	}

	return merged_df, results


def plot_category_counts(t_category_counts, b_category_counts, 
	category_keys=['expression_only', 'chromatin_only', 'both', 'neither'],
	category_names=['Expression\nonly', 'Chromatin\nonly', 'Both', 'Neither'], title=""):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

	plot_cell_cycle_category_counts(ax1, t_category_counts, ylim=500,
		category_keys=category_keys, category_names=category_names)
	ax1.set_title("Mother branch", fontsize=16, pad=15,
				 fontfamily='Open Sans', fontweight='demi')
	ax1.set_ylabel("# of genes")

	plot_cell_cycle_category_counts(ax2, b_category_counts, ylim=500,
		category_keys=category_keys, category_names=category_names)
	ax2.set_title("Daughter branch", fontsize=16, pad=15,
				 fontfamily='Open Sans', fontweight='demi')
	plt.suptitle(title, fontsize=24,
				 fontfamily='Open Sans', fontweight='demi', y=1.15)


def plot_cell_cycle_category_counts(ax1, results, ylim=None,
	category_keys=['expression_only', 'chromatin_only', 'both', 'neither'],
	category_names=['Expression\nonly', 'Chromatin\nonly', 'Both', 'Neither']):
	"""
	Create a bar plot for cell cycle classification results with a broken y-axis
	to properly display the large difference between "Neither" and other categories.
	"""

	values = []
	colors = []

	for i, key in enumerate(category_keys):
		values.append(results[key])
		colors.append(color_map[key])
	
	# Create the bars in both axes
	bars1 = ax1.bar(category_names, values, color=colors, alpha=1.)

	if ylim is None:
		ymax = 0
		for cat in category_keys[:-1]:
			ymax = results[cat] if results[cat] > ymax else ymax
		ylim = ymax*1.7

	ax1.set_ylim(0, ylim)
	
	# Add count labels on top of bars
	for i, bar in enumerate(bars1):
		height = bar.get_height()
		count = height

		if height < ylim:
			y_pos = height*1.01
		else:
			y_pos = ylim*0.85

		ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
			   f'{int(count)}', ha='center', va='bottom', 
			   fontsize=12, fontweight='demi', fontfamily='Open Sans')

def plot_category_venn_diagram(t_category_counts, b_category_counts, 
	category_keys=['expression_only', 'chromatin_only', 'both', 'neither'],
	category_names=['Expression\nonly', 'Chromatin\nonly', 'Both', 'Neither'], title=""):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

	plot_cell_cycle_venn_diagram(ax1, t_category_counts,
		category_keys=category_keys, category_names=category_names)
	ax1.set_title("Mother branch", fontsize=16, pad=15,
				 fontfamily='Open Sans', fontweight='demi')
	ax1.set_ylabel("# of genes")

	plot_cell_cycle_venn_diagram(ax2, b_category_counts,
		category_keys=category_keys, category_names=category_names)
	ax2.set_title("Daughter branch", fontsize=16, pad=15,
				 fontfamily='Open Sans', fontweight='demi')
	plt.suptitle(title, fontsize=24,
				 fontfamily='Open Sans', fontweight='demi', y=1.15)
	plt.subplots_adjust(wspace=0., bottom=0.2)


def plot_cell_cycle_venn_diagram(ax, results, 
								category_keys=['expression_only', 'chromatin_only', 'both', 'neither'],
								category_names=['Expression\nonly', 'Chromatin\nonly', 'Both', 'Neither'],
								colors=None):
	"""
	Create a Venn diagram for cell cycle classification results showing the overlap
	between expression and chromatin categories.
	"""
	from matplotlib_venn import venn2, venn2_circles
	
	# Default colors if none provided
	colors = [color_map[key] for key in category_keys]
	text_colors = [text_color_map[key] for key in category_keys]
	category_names = [cat.replace('\n', ' ') for cat in category_names]
	
	# Map the categories to set notation for venn diagram
	# For a Venn diagram we need:
	# - Set A: Expression (expression_only + both)
	# - Set B: Chromatin (chromatin_only + both)
	# - Both: (both)
	
	# Extract values
	cat_1_only = results.get(category_keys[0], 0)
	cat_2_only = results.get(category_keys[1], 0)
	both = results.get(category_keys[2], 0)
	neither = results.get(category_keys[3], 0)
	total = cat_1_only + cat_2_only + both + neither

	# Calculate set sizes for venn diagram
	set_a = cat_1_only + both
	set_b = cat_2_only + both
	
	# Create the Venn diagram
	v = venn2(subsets=(cat_1_only, cat_2_only, both), 
			  set_labels=("Expression", "Chromatin"),
			  ax=ax,
			  alpha=1.,
			  set_colors=(colors[0], colors[1]))
	
	# Add circles with a slightly thicker line
	venn2_circles(subsets=(cat_1_only, cat_2_only, both), 
				  ax=ax, 
				  linewidth=1)
	
	# Customize the labels with counts
	# Format label for Expression only (10)
	v.get_label_by_id('10').set_text(f'{category_names[0]}\n{cat_1_only} ({cat_1_only/total*100:.1f}%)')
	v.get_label_by_id('10').set_color(text_colors[0])
	v.get_patch_by_id('10').set_color(colors[0])
	
	# Format label for Chromatin only (01)
	v.get_label_by_id('01').set_text(f'{category_names[1]}\n{cat_2_only} ({cat_2_only/total*100:.1f}%)')
	v.get_patch_by_id('01').set_color(colors[1])
	v.get_label_by_id('01').set_color(text_colors[1])
	
	# Format label for both (11)
	v.get_label_by_id('11').set_text(f'{both}\n({both/total*100:.1f}%)')
	v.get_patch_by_id('11').set_color(colors[2])
	v.get_label_by_id('11').set_color(text_colors[2])
	
	# Add 'neither' count as text annotation
	ax.text(0.5, -0.15, 
			f"Neither: {neither} ({neither/total*100:.1f}%)", 
			ha='center', 
			fontsize=12, 
			transform=ax.transAxes, 
			bbox=dict(facecolor=colors[3], alpha=1, edgecolor='black',
				lw=0.5, boxstyle='square,pad=0.5'))
	
	# Add title showing total count
	ax.set_title(f'Cell Cycle Classification (Total: {total})', 
				fontsize=14, 
				fontweight='bold')
	
	# Remove axes
	ax.axis('off')


def plot_three_category_venn_diagram(ax, results, 
									category_keys=['expression_only', 'promoter_only', 'entropy_only'],
									category_names=['Expression', 'Promoter', 'Entropy'],
									colors=None):
	"""
	Create a three-set Venn diagram showing the overlap between three different categories.

	Returns:
	--------
	matplotlib.figure.Figure
		The figure with the Venn diagram
	"""
	from matplotlib_venn import venn3, venn3_circles
	import matplotlib.pyplot as plt
	import numpy as np
	
	# Default colors if none provided
	colors = [color_map[key] for key in category_keys]
	text_colors = [text_color_map[key] for key in category_keys]
	
	# Extract all possible region counts from the results dictionary
	# For a 3-set Venn diagram, we need counts for:
	# (100): Only in set A
	# (010): Only in set B
	# (001): Only in set C
	# (110): In sets A and B but not C
	# (101): In sets A and C but not B
	# (011): In sets B and C but not A
	# (111): In all three sets
	
	# Define helper function to get value with default
	def get_value(key, default=0):
		return results.get(key, default)
	
	# Get individual category counts (may include overlaps)
	set_a_total = get_value(category_keys[0])
	set_b_total = get_value(category_keys[1])
	set_c_total = get_value(category_keys[2])
	
	# Get pairwise overlaps
	ab_overlap = get_value(f"{category_keys[0]}_{category_keys[1]}")
	ac_overlap = get_value(f"{category_keys[0]}_{category_keys[2]}")
	bc_overlap = get_value(f"{category_keys[1]}_{category_keys[2]}")
	
	# Get triple overlap
	abc_overlap = get_value(f"{category_keys[0]}_{category_keys[1]}_{category_keys[2]}")
	
	# Calculate exclusive regions
	# Only in A = A - (A∩B) - (A∩C) + (A∩B∩C)
	only_a = set_a_total - ab_overlap - ac_overlap + abc_overlap
	
	# Only in B = B - (A∩B) - (B∩C) + (A∩B∩C)
	only_b = set_b_total - ab_overlap - bc_overlap + abc_overlap
	
	# Only in C = C - (A∩C) - (B∩C) + (A∩B∩C)
	only_c = set_c_total - ac_overlap - bc_overlap + abc_overlap
	
	# A∩B exclusive = (A∩B) - (A∩B∩C)
	only_ab = ab_overlap - abc_overlap
	
	# A∩C exclusive = (A∩C) - (A∩B∩C)
	only_ac = ac_overlap - abc_overlap
	
	# B∩C exclusive = (B∩C) - (A∩B∩C)
	only_bc = bc_overlap - abc_overlap
	
	# Triple intersection = A∩B∩C
	only_abc = abc_overlap
	
	# Get 'none' category count
	none = get_value('none')
	
	# Prepare the subsets for venn3 in the order: (100, 010, 110, 001, 101, 011, 111)
	# This corresponds to: (A, B, AB, C, AC, BC, ABC)
	subsets = (only_a, only_b, only_ab, only_c, only_ac, only_bc, only_abc)
	
	# Create the Venn diagram
	v = venn3(subsets=subsets, 
			  set_labels=category_names,
			  ax=ax,
			  alpha=1,
			  set_colors=colors)
	
	# Add circles with a slightly thicker line
	venn3_circles(subsets=subsets, 
				  ax=ax, 
				  linewidth=1)
	
	# Calculate total elements
	total = sum(subsets) + none
	
	# Add 'none' count as text annotation
	ax.text(0.9, 0.1, 
			f"None: {none} ({none/total*100:.1f}%)", 
			ha='center', 
			fontsize=12, 
			transform=ax.transAxes, 
			bbox=dict(facecolor='#d3d3d3', alpha=0.2, boxstyle='square,pad=0.3'))

	format_ids = ['100', '010', '001']
	format_counts = [only_a, only_b, only_c]

	def get_patch_centroid(patch):
		path = patch.get_path()
		vertices = path.vertices
		return vertices.mean(axis=0)

	for i, fid in enumerate(format_ids):
		label = v.get_label_by_id(fid)
		label.set_text(f'{category_names[i]} only\n{format_counts[i]} ({format_counts[i]/total*100:.1f}%)')
		label.set_color(text_colors[i])
		label.set_fontsize(12)
		v.get_patch_by_id(fid).set_color(colors[i])

		# Adjust the label position to be closer to the centroid
		patch = v.get_patch_by_id(fid)
		centroid = get_patch_centroid(patch)
		label_position = label.get_position()
		new_label_position = (label_position[0]+centroid[0])/2., (label_position[1]+centroid[1])/2.
		label.set_position(new_label_position)

	intersection_ids = ['110', '101', '011', '111']
	intersection_colors = ['#624b57', '#4d5166', '#948f80', '#505050']
		
	for i, region_id in enumerate(intersection_ids):
		patch = v.get_patch_by_id(region_id)
		if patch is None: continue

		patch.set_color(intersection_colors[i])
		label = v.get_label_by_id(region_id)
		label.set_color('white')
		count = int(label.get_text())
		label.set_text(f"{count}\n({count/total*100:.1f}%)")


def plot_scatter_timing(ax, expression_df, promoter_df, entropy_df, suffix='_t',
						xlim=(0, 64), ylim=(0, 64), marker_size=10, alpha=1,
						plot_intersection_genes=False, plot_trough_indices=True):
	"""
	Plot a scatter plot comparing expression timing vs promoter occupancy and expression vs entropy timing.
	"""

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	
	# Set default colors if not provided
	colors = color_map['promoter_only'], color_map['entropy_only']
	
	# Column names with suffix
	bin_col = f'bin{suffix}'

	peak_tx_tp_col = f'peak_idx{suffix}'
	trough_tx_tp_col = f'trough_idx{suffix}'
	chrom_tp_col = f'peak_idx{suffix}'

	# For the cell cycle genes we'll keep track if the 
	# gene is peaky or troughy
	tx_skew_class_key = f'cell_cycle_skew{suffix}'
	
	def retrieve_intersection_cc_genes(expression_df, chromatin_df, na_idx=65,
		plot_trough_indices=False):
		"""Retrieve the cell cycle gene data for the intersection between expression and a
		chromatin dataframe"""

		expression_df = expression_df.copy()
		chromatin_df = chromatin_df.copy()

		expression_df.loc[expression_df[bin_col] == "Not cell cycle", peak_tx_tp_col] = na_idx
		expression_df.loc[expression_df[bin_col] == "Not cell cycle", trough_tx_tp_col] = na_idx
		chromatin_df.loc[chromatin_df[bin_col] == "Not cell cycle", chrom_tp_col] = na_idx

		common_expr_chrom = expression_df[[]].join(chromatin_df[[]], how='inner').index
		expr_for_chrom = expression_df.loc[common_expr_chrom]
		chrom_filtered = chromatin_df.loc[common_expr_chrom]

		df = pd.DataFrame({
			'x': chrom_filtered[chrom_tp_col],
			'y': expr_for_chrom[peak_tx_tp_col],
			'skew_classification': expr_for_chrom[tx_skew_class_key]
		})

		# For the trough genes, use the trough tx timepoint
		if plot_trough_indices:
			df.loc[df.skew_classification == 'trough' , 'y'] = expr_for_chrom[trough_tx_tp_col]

		return df

	def retrieve_all_intersection_genes():
		"""Retrieve the set of genes that intersect as cell cycle genes for all three categories"""
		cc_tx = expression_df[expression_df[bin_col] != 'Not cell cycle']
		cc_sm = promoter_df[promoter_df[bin_col] != 'Not cell cycle']
		cc_en = entropy_df[entropy_df[bin_col] != 'Not cell cycle']
		intersection_orfs = cc_tx[[]].join(cc_sm[[]], how='inner').join(cc_en[[]], how='inner').index    
		return intersection_orfs

	# Retrieve the dataset to plot, in the case of promoter binding,
	# we have hte option to plot the trough index of expression, where binding may be repressive
	cc_prom_tx_df = retrieve_intersection_cc_genes(expression_df, promoter_df, na_idx=64-0.25,
		plot_trough_indices=plot_trough_indices)

	# For entropy, we expect a direct comparison, so we should not need to plot the trough expression values
	cc_entropy_tx_df = retrieve_intersection_cc_genes(expression_df, entropy_df, na_idx=64+0.25,
		plot_trough_indices=False)

	# Plot scatter for expression vs promoter
	import matplotlib.markers as mmarkers

	# Plot non-trough genes
	non_trough_rows = cc_prom_tx_df[cc_prom_tx_df.skew_classification != 'trough']
	prom_scatter = ax.scatter(
		non_trough_rows.x, 
		non_trough_rows.y,
		color=colors[0], 
		alpha=alpha, 
		s=marker_size, 
		marker=mmarkers.MarkerStyle('o', fillstyle='none'),
		lw=0.75,
		label='Promoter Occupancy'
	)

	import matplotlib.markers as mmarkers

	# Trough genes
	trough_rows = cc_prom_tx_df[cc_prom_tx_df.skew_classification == 'trough']
	trough_prom = ax.scatter(trough_rows.x, trough_rows.y, facecolor=colors[0],
		edgecolor=colors[0],
		# marker=mmarkers.MarkerStyle('o', fillstyle='bottom'), s=marker_size, lw=0,
		marker=mmarkers.MarkerStyle('v', fillstyle='full'), s=marker_size, lw=0,
		label='Promoter Occupancy trough gene')

	# Plot scatter for expression vs entropy
	entr_scatter = ax.scatter(
		cc_entropy_tx_df.x,
		cc_entropy_tx_df.y,
		color=colors[1],
		alpha=alpha, 
		marker=mmarkers.MarkerStyle('x', fillstyle='full'),
		lw=0.75,
		s=marker_size,
		label='Nucleosome Entropy'
	)

	# For the genes that intersect all groups, annotate
	if plot_intersection_genes:
		all_cc_genes = retrieve_all_intersection_genes()

		from src.sgd import get_gene_title_name

		if len(all_cc_genes) > 0:

			for cc_orf in all_cc_genes:
				prom_row = cc_prom_tx_df.loc[cc_orf]
				entropy_row = cc_entropy_tx_df.loc[cc_orf]

				ax.plot([prom_row.x, entropy_row.x], [prom_row.y, entropy_row.y],
					color='#555', ls='dotted', lw=0.75)

				ax.scatter(prom_row.x, prom_row.y, edgecolor=colors[0], marker='D',
					facecolor='none', lw=1, s=11)
				ax.scatter(entropy_row.x, entropy_row.y, edgecolor=colors[1], marker='D',
					facecolor='none', lw=1, s=11)

				gene_name = get_gene_title_name(cc_orf, include_system=False)
				ax.text((prom_row.x+entropy_row.x)/2.5, prom_row.y+-.25, gene_name,
					ha='center', color='#555', fontsize=8)
	
	# Set axis limits
	ax.set_xlim(0, 64.5)
	ax.set_ylim(64.5, 0)
	
	ax.axvline(41.5, c='black', lw=0.25)
	ax.axhline(41.5, c='black', lw=0.25)
	ax.axvline(63.5, c='black', lw=0.25)
	ax.axhline(63.5, c='black', lw=0.25)

	return prom_scatter, trough_prom, entr_scatter


def plot_example_gene_cycles(cc_orfs, chromatin_analysis, analysis,
							config1):

	from src.sgd import get_gene_title_name

	plt.figure(figsize=(8, 4))
	plt.subplot(1, 2, 1)
	for orf in cc_orfs:

		sm_dat = chromatin_analysis.small_promoter_occupancies_df.loc[orf][
			config1.t_indices()].values
		tx_dat = analysis.deconvolved_genes_F.loc[orf][config1.t_indices()].values

		sm_dat = np.concatenate([sm_dat, sm_dat[:1]])
		tx_dat = np.concatenate([tx_dat, tx_dat[:1]])

		indices = np.arange(len(config1.t_indices())+1)
		plt.plot(sm_dat, tx_dat, lw=0.1, c='black')
		plt.scatter(sm_dat, tx_dat, c=indices, cmap='twilight', s=10)

		gene_name = get_gene_title_name(orf, include_system=False)
		plt.text(sm_dat[0], tx_dat[0], gene_name)

	plt.xlabel("Promoter occupancy")
	plt.ylabel("Expression")


	plt.subplot(1, 2, 2)
	for orf in cc_orfs:

		ent_dat = chromatin_analysis.nucleosome_genebody_entropies_df.loc[orf][config1.t_indices()].values
		tx_dat = analysis.deconvolved_genes_F.loc[orf][config1.t_indices()].values

		ent_dat = np.concatenate([ent_dat, ent_dat[:1]])
		tx_dat = np.concatenate([tx_dat, tx_dat[:1]])

		indices = np.arange(len(config1.t_indices())+1)
		plt.plot(ent_dat, tx_dat, lw=0.1, c='black')
		plt.scatter(ent_dat, tx_dat, c=indices, cmap='twilight', s=10)

		gene_name = get_gene_title_name(orf, include_system=False)
		plt.text(ent_dat[0], tx_dat[0], gene_name)

	plt.xlabel("Nucleosome entropy")
	plt.ylabel("Expression")
	plt.xlim(2.5, 5.5)

	plt.suptitle("Example cell cycle genes")
	plt.colorbar()