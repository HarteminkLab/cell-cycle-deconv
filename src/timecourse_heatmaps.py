import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from src.gene_expression import load_gene_expression_data


class CellCycleHeatmapPlotter:
	"""
	A class for creating and plotting heatmaps for cell cycle gene expression data.
	
	This class encapsulates functionality for:
	- Subsetting gene expression data into different cell cycle phases
	- Plotting heatmaps with customizable row heights
	- Creating separate colorbar figures
	"""
	
	def __init__(self, analysis, config, expression_combined_polar_data_df, wt_data=None):
		"""
		Initialize the CellCycleHeatmapPlotter with necessary data.
		
		Parameters:
		-----------
		analysis : object
			The analysis object containing deconvolved_genes_F data
		config : object
			Configuration object with functions to get indices
		expression_combined_polar_data_df : pandas.DataFrame
			DataFrame containing cell cycle annotation data
		wt_data : pandas.DataFrame, optional
			Wild-type gene expression data. If None, it will be loaded from source.
		"""
		self.analysis = analysis
		self.config = config
		self.expression_df = expression_combined_polar_data_df
		
		# Get indices for different branches
		self.i_indices = config.i_indices()
		self.t_indices = config.t_indices()
		self.b_indices = config.b_indices()
		
		# Load wild-type data if not provided
		self.wt_data1 = load_gene_expression_data(1)
		self.wt_data2 = load_gene_expression_data(2)

		# Define colormap and normalization to be used consistently across all plots
		self.norm = mpl.colors.Normalize(vmin=-1.5, vmax=1.5)
		self.cmap = plt.cm.RdBu_r
		
		# Subset the data into cell cycle groups
		self._subset_cell_cycle_genes()
	
	def _subset_cell_cycle_genes(self):
		"""
		Subset the expression data into different cell cycle groups.
		
		Sets the following attributes:
		- dg1_only: Genes with peak expression in DG1 but not in MG1
		- mg1_only: Genes with peak expression in MG1 but not in DG1
		- remaining_cell_cycle: Genes that are cell cycling but not DG1/MG1 specific
		"""

		# Daughter G1 only:
		# Cell cycle gene (on daughter branch)
		# DG1 peak
		# No MG1 peak, optionally cell cycling on the mother branch
		self.dg1_only = self.expression_df[
			(self.expression_df.bin_b != 'Not cell cycle') & 
			(self.expression_df.peak_phase_b == 'DG1') & 
			(self.expression_df.peak_phase_t != 'MG1')]
		
		# Mother G1 only
		# Cell cycle gene (on mother branch)
		# MG1 peak
		# No DG1 peak, optionally cell cycling on the daughter branch
		self.mg1_only = self.expression_df[
			(self.expression_df.bin_t != 'Not cell cycle') & 
			(self.expression_df.peak_phase_t == 'MG1') & 
			(self.expression_df.peak_phase_b != 'DG1')]
		
		# Remaining genes, cycling on either branch, and...
		cell_cycle_in_either = self.expression_df[
			(self.expression_df.bin_t != 'Not cell cycle') |
			(self.expression_df.bin_b != 'Not cell cycle')]
		
		# Not Daughter G1 or Mother G1 specific
		cell_cycle_in_either_minus_dg1_mg1_idx = list(
			set(cell_cycle_in_either.index)
			.difference(set(self.dg1_only.index))
			.difference(set(self.mg1_only.index))
		)
		self.remaining_cell_cycle = self.expression_df.loc[cell_cycle_in_either_minus_dg1_mg1_idx]
		
		# Print summary statistics
		print(f"There are {len(self.dg1_only)} DG1 only genes and {len(self.mg1_only)} MG1 only genes")
		print(f"and {len(self.remaining_cell_cycle)} that are cell cycling but not DG1/MG1 specific")

	
	def plot_wt(self, wt_data, sorted_indices, xticks=True, xlabel=False):
		"""
		Plot the wild-type heatmap for the given sorted indices.
		
		Parameters:
		-----------
		sorted_indices : array-like
			Indices to use for sorting the data
		"""
		extent = [0, int(wt_data.columns[-1]), 0, len(wt_data)]
		wt_data_log2fold = wt_data.copy()
		wt_data_log2fold.loc[:] = np.log2((wt_data.values+1) / 
									  (wt_data.mean(1).values[:, None]+1))
		plt.imshow(wt_data_log2fold.loc[sorted_indices], aspect='auto',
				  cmap=self.cmap, norm=self.norm, extent=extent)

		if not xticks:
			plt.xticks([])
		else:
			plt.xticks(np.arange(0, int(wt_data.columns[-1])+10, 40))

		if xlabel:
			plt.xlabel("Experiment time, min")

		plt.yticks([])

	
	def retrieve_data_to_plot_heatmap(self, sel_rows, sort_key):
		"""
		Retrieve and process data for plotting heatmaps.
		
		Parameters:
		-----------
		sel_rows : pandas.DataFrame
			Selected rows to plot
		sort_key : str
			Column to use for sorting
			
		Returns:
		--------
		tx_logfold : numpy.ndarray
			Log-fold change data for plotting
		indices_sorted : array-like
			Sorted indices
		"""
		indices_sorted = sel_rows.sort_values(sort_key).index
		tx = self.analysis.deconvolved_genes_F.loc[indices_sorted]
		
		indices = np.concatenate([self.t_indices, self.b_indices])
		tx_logfold = np.log2((tx.values+1e-5) / (tx[indices].mean(1).values[:, None]+1e-5))
		return tx_logfold, indices_sorted
	

	def plot_im(self, sel_rows, sort_key, indices, plot_xticks=False):
		"""
		Plot a heatmap image for the given selection and indices.
		
		Parameters:
		-----------
		sel_rows : pandas.DataFrame
			Selected rows to plot
		sort_key : str
			Column to use for sorting
		indices : array-like
			Indices for the specific branch to plot
			
		Returns:
		--------
		dat : numpy.ndarray
			Data used for plotting
		indices_sorted : array-like
			Sorted indices
		"""

		dat, indices_sorted = self.retrieve_data_to_plot_heatmap(sel_rows, sort_key)
		extent = [0, dat.shape[1], 0, dat.shape[0]]

		plt.imshow(dat[:, indices], cmap=self.cmap, aspect='auto', 
				   norm=self.norm)
		if not plot_xticks:
			plt.xticks([])

		plt.yticks([])
		return dat, indices_sorted
	

	def plot_heatmaps(self, row_heights=None, figsize=(11, 13)):
		"""
		Plot heatmaps with configurable row heights.
		
		Parameters:
		-----------
		row_heights : list, optional
			List of relative heights for each row. Default is equal heights.
			Example: [0.1, 0.4, 0.2]
		figsize : tuple, optional
			Figure size in inches (width, height). Default is (11, 6).
			
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The figure containing the heatmaps
		"""
		nrows = 3
		ncols = 5
		
		# Set default row heights if not provided
		if row_heights is None:
			row_heights = [1, 1, 1]  # Equal heights by default
		
		# Create figure
		fig = plt.figure(figsize=figsize)
		
		# Create GridSpec with specified row heights
		gs = gridspec.GridSpec(nrows, ncols, height_ratios=row_heights)
		
		# Define a function to plot each row
		def plot_row(group_name, selection, sort_key, row_idx):

			# Wild-type plot (first column)
			plt.subplot(gs[row_idx, 0])
			dat, indices_sorted = self.retrieve_data_to_plot_heatmap(selection, sort_key)
			self.plot_wt(self.wt_data1, indices_sorted, xticks=(row_idx == 2),
				xlabel=(row_idx==2))
			plt.ylabel(group_name, rotation=0, va='center', ha='right', fontsize=16)
			if row_idx == 0:
				plt.title("Wild-type 1", fontsize=14, y=1.05)

			# Wild-type plot (first column)
			plt.subplot(gs[row_idx, 1])
			dat, indices_sorted = self.retrieve_data_to_plot_heatmap(selection, sort_key)
			self.plot_wt(self.wt_data2, indices_sorted, xticks=(row_idx == 2))
			if row_idx == 0:
				plt.title("Wild-type 2", fontsize=14, y=1.05)
				
			# Recovery branch plot
			plt.subplot(gs[row_idx, 2])
			dat, indices_sorted = self.plot_im(selection, sort_key, self.i_indices,
				plot_xticks=(row_idx==2))
			if row_idx == 2: plt.xlabel("Average single cell time, min")
			if row_idx == 0:
				plt.title("Deconvolved,\nRecovery branch", fontsize=14, y=1.05)
				
			# Mother branch plot 
			plt.subplot(gs[row_idx, 3])
			dat, indices_sorted = self.plot_im(selection, sort_key, self.t_indices,
				plot_xticks=(row_idx==2))
			if row_idx == 0:
				plt.title("Deconvolved,\nMother branch", fontsize=13)
			
			# Daughter branch plot 
			plt.subplot(gs[row_idx, 4])
			dat, indices_sorted = self.plot_im(selection, sort_key, self.b_indices,
				plot_xticks=(row_idx==2))
			if row_idx == 0:
				plt.title("Deconvolved,\nDaughter branch", fontsize=13)
			
			return indices_sorted
		
		# Plot each row
		plot_row(f"Mother G1,\nn={len(self.mg1_only)}", self.mg1_only, 'peak_idx_t', 0)
		plot_row(f"All other\ncell cycle genes,\nn={len(self.remaining_cell_cycle)}", 
			self.remaining_cell_cycle, 'peak_idx_t', 1)
		plot_row(f"Daughter G1,\nn={len(self.dg1_only)}", self.dg1_only, 'peak_idx_b', 2)
		
		plt.tight_layout()
		plt.subplots_adjust(left=0.3, right=0.7)
		plt.suptitle("Cell cycle gene expression", fontsize=23, y=1.05)

		return fig
	
	def create_colorbar_figure(self, height=1, width=0.5):
		"""
		Create a separate figure with just a vertical colorbar.
		
		Parameters:
		-----------
		height : float, optional
			Height of the colorbar figure in inches. Default is 6.
		width : float, optional
			Width of the colorbar figure in inches. Default is 1.
			
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The figure containing only the colorbar
		"""
		# Create a ScalarMappable object with the colormap
		sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
		sm.set_array([])  # Dummy array for the colorbar
		
		# Create a new figure for the colorbar
		fig = plt.figure(figsize=(width, height))
		
		# Add a single axis that fills the figure
		ax = fig.add_axes([0.3, 0.05, 0.3, 0.9])
		
		# Create the colorbar
		cbar = plt.colorbar(sm, cax=ax)
		cbar.set_label('log₂ fold change')
		
		return fig
	
	def plot_all(self, row_heights=[0.1, 0.6, 0.2], figsize=(23, 9), 
				 colorbar_height=1, colorbar_width=0.5,
				 save_directory=None):
		"""
		Create both heatmap and colorbar figures.
		"""
		from src.figure_configs import save_figure_for_paper

		ndg1 = len(self.dg1_only)
		nmg1 = len(self.mg1_only)
		nother = len(self.remaining_cell_cycle)

		total = ndg1+nmg1+nother
		row_heights = [nmg1/total, nother/total, ndg1/total]

		# Plot heatmaps
		heatmap_fig = self.plot_heatmaps(row_heights=row_heights, figsize=figsize)

		if save_directory is not None:
			save_figure_for_paper(f"{save_directory}/heatmap_timecourse.png")
		
		# Create colorbar figure
		colorbar_fig = self.create_colorbar_figure(height=colorbar_height, width=colorbar_width)

		if save_directory is not None:
			save_figure_for_paper(f"{save_directory}/heatmap_timecourse_colorbar.png")

		return heatmap_fig, colorbar_fig
