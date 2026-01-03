import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

def plot_pairs_index(data, color=plt.cm.Reds(0.5), expanded_lims=False):
	
	# Create triangle layout matching plot_pairs_of_ptrs
	fig, triangle_axes = generate_metrics_comparison_plot(figsize=(6, 6))
	
	# Define the pairs to plot (index_x, index_y)
	# Order matches the triangle layout
	pairs = [
		(0, 3),  # Promoter vs Expression
		(1, 3),  # Nuc. Occupancy vs Expression
		(2, 3),  # Nuc. entropy vs Expression
		(0, 2),  # Promoter vs Nuc. entropy
		(1, 2),  # Nuc. Occupancy vs Nuc. entropy
		(0, 1),  # Promoter vs Nuc. occupancy
	]

	variable_names = ['Promoter\noccupancy', 'Nucleosome\noccupancy', 'Nucleosome\nentropy', 
		'Expression']

	if expanded_lims:
		lims = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]
	else:
		lims = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]

	from src.config import load_default_chrom_configs
	config, _ = load_default_chrom_configs()

	# Create scatter plots for each pair
	for idx, (i, j) in enumerate(pairs):
		ax = triangle_axes[idx]

		# Mean center
		mean_centered_x = data[i] - data[i].mean()
		mean_centered_y = data[j] - data[j].mean()

		from src.plot_helpers import plot_trajectory_deconvolved_values

		plot_trajectory_deconvolved_values(config, mean_centered_x,
			mean_centered_y, x_key=variable_names[i],
			y_key=variable_names[j], ax=ax, plot_arrows=True)

		# Set limits
		ax.set_xlim(lims[i])
		ax.set_ylim(lims[j])
		ax.set_xticks([])
		ax.set_yticks([])
		
		# Labels matching triangle layout pattern
		if idx < 3:  # Top row - set titles (x-axis labels)
			ax.set_title(variable_names[i], fontsize=11)
		
		if idx in [0, 3, 5]:  # Left column - set y-axis labels
			ax.set_ylabel(variable_names[j], fontsize=11)


def plot_meta_gene_3d(cluster_meta_gene):
	mean_center = cluster_meta_gene.mean(1)

	centered_data = cluster_meta_gene - mean_center[:, None]

	# Close the data to create closed trajectory loops
	centered_data_closed = np.hstack([centered_data, centered_data[:, 0:1]])

	x, y, z, c = centered_data_closed

	fig = plt.figure(figsize=(7, 6))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_proj_type('ortho') 

	# Create the 3D line plot with color mapping
	points = np.array([x, y, z]).T.reshape(-1, 1, 3)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)

	from mpl_toolkits.mplot3d.art3d import Line3DCollection
	lc = Line3DCollection(segments, color='#999')
	lc.set_array(c)
	lc.set_linewidth(3)
	ax.add_collection(lc)
	
	lc = Line3DCollection(segments, cmap='Purples_r')
	lc.set_array(c)
	lc.set_linewidth(2)
	ax.add_collection(lc)

	# Add colorbar
	#fig.colorbar(lc, ax=ax, label='Color dimension (c)')

	ax.set_xlabel('Promoter occupancy')
	ax.set_ylabel('Nucleosome occupancy')
	ax.set_zlabel('Nucleosome entropy')

	# Add projections onto the 2D faces
	projection_alpha = 0.3
	projection_linewidth = 1

	lim_val = 1
	lims = [
		(-lim_val, lim_val),
		(-lim_val, lim_val),
		(-lim_val, lim_val),
		(-lim_val, lim_val),
	]
	def plot_on_plane(x, y, z, flattened_axis):
		
		if flattened_axis == 'z':
			z = np.zeros_like(x)+lims[2][0]
		elif flattened_axis == 'y':
			y = np.zeros_like(x)+lims[1][0]
		elif flattened_axis == 'x':
			x = np.zeros_like(x)+lims[0][0]

		# Create the 3D line plot with color mapping
		points = np.array([x, y, z]).T.reshape(-1, 1, 3)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
	
		lc = Line3DCollection(segments, color='#ddd')
		lc.set_linewidth(3)
		ax.add_collection(lc)
		
		lc = Line3DCollection(segments, cmap='Greys_r', alpha=0.5)
		lc.set_array(c)
		lc.set_linewidth(2)
		ax.add_collection(lc)

	plot_on_plane(x, y, z, flattened_axis='x')
	plot_on_plane(x, y, z, flattened_axis='y')
	plot_on_plane(x, y, z, flattened_axis='z')

	# Set axes face colors (panes)
	ax.xaxis.pane.fill = True
	ax.yaxis.pane.fill = True
	ax.zaxis.pane.fill = True
	ax.xaxis.pane.set_facecolor('#f9f9f9')  # Light gray for x-axis pane
	ax.yaxis.pane.set_facecolor('#f9f9f9')  # Light blue for y-axis pane
	ax.zaxis.pane.set_facecolor('#f9f9f9')  # Light yellow for z-axis pane

	ax.set_xticks(np.linspace(lims[0][0], lims[0][1], 5))
	ax.set_yticks(np.linspace(lims[1][0], lims[1][1], 5))
	ax.set_zticks(np.linspace(lims[2][0], lims[2][1], 5))

	# Set axis limits
	ax.set_xlim(lims[0])
	ax.set_ylim(lims[1])
	ax.set_zlim(lims[2])

	# Customize grid
	ax.xaxis._axinfo["grid"].update({"linewidth": 0.75, "linestyle": "solid", "color": "#ddd"})
	ax.yaxis._axinfo["grid"].update({"linewidth": 0.75, "linestyle": "solid", "color": "#ddd"})
	ax.zaxis._axinfo["grid"].update({"linewidth": 0.75, "linestyle": "solid", "color": "#ddd"})

	ax.set_box_aspect(None, zoom=0.75)
	ax.view_init(elev=35.25, azim=45)

def compute_diameter(data):
	"""
	Compute the true diameter (max pairwise distance) for each gene's trajectory.
	
	Parameters:
	-----------
	data : np.ndarray
		Shape (n_genes, 4, m_timepoints)
		
	Returns:
	--------
	diameters : np.ndarray
		Shape (n_genes,) containing the diameter for each gene
	"""
	n_genes, n_metrics, m_timepoints = data.shape
	
	# Reshape to (n_genes, m_timepoints, 4) for easier distance computation
	data_transposed = np.transpose(data, (0, 2, 1))  # Shape: (n_genes, 128, 4)
	
	# Compute pairwise distances for all genes at once
	# Expand dimensions for broadcasting
	# data_transposed[:, :, np.newaxis, :] has shape (n_genes, 128, 1, 4)
	# data_transposed[:, np.newaxis, :, :] has shape (n_genes, 1, 128, 4)
	diff = data_transposed[:, :, np.newaxis, :] - data_transposed[:, np.newaxis, :, :]
	# diff has shape (n_genes, 128, 128, 4)
	
	# Compute Euclidean distances
	distances = np.linalg.norm(diff, axis=3)  # Shape: (n_genes, 128, 128)
	
	# Find maximum distance for each gene
	diameters = np.max(distances, axis=(1, 2))  # Shape: (n_genes,)
	
	return diameters

def generate_metrics_comparison_plot(figsize=(4, 4)):
	fig, axs = plt.subplots(3, 3, figsize=figsize)
	triangle_axes = []

	# Hide all axes first
	for ax in np.array(axs).flatten():
		ax.set_visible(False)

	# Show descending pattern
	for i in range(3):
		for j in range(3 - i):  # Show first (3-i) columns in row i
			axs[i, j].set_visible(True)
			axs[i, j].set_xticks([])
			axs[i, j].set_yticks([])
			triangle_axes.append(axs[i, j])

	plt.subplots_adjust(wspace=0, hspace=0)

	triangle_axes
	return fig, triangle_axes
