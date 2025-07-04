import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from src.origins import load_origins
from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
from src.combined_chromatin_model import CombinedChromatinModel
from src.config import load_default_chrom_configs, get_average_timepoints_for_branch
from src.figure_configs import save_figure_for_paper
import matplotlib.gridspec as gridspec
import os

class FigureLoci(object):
	"""Analysis of deconvolved locus.

	Goal:
	- Present effectiveness of chromatin deconvolution.
	- Introduce the need for copy number correction.

	How:
	- Show the locus with interesting gene level and origin cycling chromatin
	- Show the overall occupancy of the plot, compare to a late replicating locus 
	  to show need for copy number correction.
	"""

	def __init__(self, output_directory):

		self.origins = load_origins()
		self.origin = self.origins.loc['oridb_817']
		self.output_directory = output_directory
		self.save_directory = output_directory + '/figure_loci_plots'
		self.figures_directory = output_directory + '/Figures'

		self.copy_corrected_data_directory = f'{self.output_directory}/chromatin_deconvolution/deconvolution_data/'
		self.no_correction_data_directory = f'{self.output_directory}/chromatin_deconvolution_no_copy/deconvolution_data/'


		# Create analyses for each of the efficient and distal loci
		# Create copy and no copy correction analyses (for sanity checking). 
		# Final figure will be a comparison of no copy correction between the efficient and distal regions
		self.genome_analyses = {
			"efficient_with_copy": GenomeDeconvolutionAnalysis(self.copy_corrected_data_directory),
			"efficient_no_copy": GenomeDeconvolutionAnalysis(self.no_correction_data_directory),
			"distal_with_copy": GenomeDeconvolutionAnalysis(self.copy_corrected_data_directory),
			"distal_no_copy": GenomeDeconvolutionAnalysis(self.no_correction_data_directory),
		}
		
		# Create combined chromatin models for raw vs predicted vs deconvolved comparisons
		self.combined_models = {
			"efficient_with_copy": None,
			"efficient_no_copy": None,
			"distal_with_copy": None,
			"distal_no_copy": None,
		}

		# Store the gene region definitions
		self.gene_regions = {
			"efficient_with_copy": {},
			"efficient_no_copy": {},
			"distal_with_copy": {},
			"distal_no_copy": {},
		}
		
		# Load default configurations
		self.config1, self.config2 = load_default_chrom_configs()
		
	def load_regions(self):

		# Window specific to oridb 817, this includes CLB2 and a control gene to the right of 
		# the origin footprint, 10kb window
		origin = self.origin
		chrom = origin.chr
		span = origin.pos-4000, origin.pos+4000

		# Load the origin's locus
		self.genome_analyses['efficient_with_copy'].load_mnase_span(origin.chr, span)
		self.genome_analyses['efficient_with_copy'].clear_cache()
		self.genome_analyses['efficient_no_copy'].load_mnase_span(origin.chr, span)
		self.genome_analyses['efficient_no_copy'].clear_cache()
		print(f"Proximal efficient region chr{origin.chr}: ", span)

		# Load a span on the same chromosome that is late replicating
		distal_span = origin.pos-45000 - 4000, origin.pos-45000 + 4000
		self.genome_analyses['distal_with_copy'].load_mnase_span(origin.chr, distal_span)
		self.genome_analyses['distal_with_copy'].clear_cache()
		self.genome_analyses['distal_no_copy'].load_mnase_span(origin.chr, distal_span)
		print(f"Distal region chr{origin.chr}: ", distal_span)
		
		# Load combined chromatin models for comparisons
		self._load_combined_models(origin.chr, span, distal_span)

	
	def _load_combined_models(self, chrom, efficient_span, distal_span):
		"""Load CombinedChromatinModel objects for raw vs predicted vs deconvolved comparisons."""
		
		# Load efficient region models
		self.combined_models['efficient_with_copy'] = CombinedChromatinModel(self.config1, self.config2)
		self.combined_models['efficient_with_copy'].load_mnase_span(chrom, efficient_span)
		self.combined_models['efficient_with_copy'].setup_deconv_model(copy_correct=True)
		
		self.combined_models['efficient_no_copy'] = CombinedChromatinModel(self.config1, self.config2)
		self.combined_models['efficient_no_copy'].load_mnase_span(chrom, efficient_span)
		self.combined_models['efficient_no_copy'].setup_deconv_model(copy_correct=False)
		
		# Load distal region models
		self.combined_models['distal_with_copy'] = CombinedChromatinModel(self.config1, self.config2)
		self.combined_models['distal_with_copy'].load_mnase_span(chrom, distal_span)
		self.combined_models['distal_with_copy'].setup_deconv_model(copy_correct=True)
		
		self.combined_models['distal_no_copy'] = CombinedChromatinModel(self.config1, self.config2)
		self.combined_models['distal_no_copy'].load_mnase_span(chrom, distal_span)
		self.combined_models['distal_no_copy'].setup_deconv_model(copy_correct=False)

	def _subset_f_metric_bp(self, f_imgs, bp_tuple, bin_start, metric='mean'):
		"""Calculate metrics for a given region."""
		from src.helpers import calc_entropy
		
		bins = (bp_tuple[0]-bin_start)//10, bp_tuple[1]//10, \
			(bp_tuple[2]-bin_start)//10, bp_tuple[3]//10
		bins = np.array(bins).astype(int)
		dat = f_imgs[:, bins[1]:bins[3], bins[0]:bins[2]]
		
		if metric == 'mean':
			return dat.mean((1, 2))
		elif metric == 'entropy':
			# Collapse rows and compute the positional entropy for each timepoint
			row_means = dat.mean(1)
			entropies = np.apply_along_axis(lambda row: calc_entropy(row), axis=1, arr=row_means)
			return entropies
		elif metric == 'none':
			return dat
			
		raise ValueError(f"Unknown metric: {metric}")

	def _compute_metric_for_comparison(self, analysis_key, metric_type, region_params):
		"""
		Compute raw, predicted, and deconvolved data for a given metric and region.
		
		Parameters:
		-----------
		analysis_key : str
			Key for the analysis ('efficient_no_copy', etc.)
		metric_type : str
			Type of metric ('mean' or 'entropy')
		region_params : tuple
			(x_start, y_start, x_end, y_end) defining the region
			
		Returns:
		--------
		dict with keys: 'raw1', 'raw2', 'predicted1', 'predicted2', 'deconvolved',
		'raw_tps1', 'raw_tps2', 'deconv_tps'
		"""
		
		# Get the models and data
		combined_model = self.combined_models[analysis_key]
		genome_analysis = self.genome_analyses[analysis_key]
		
		# Get span start for coordinate conversion
		span_start = genome_analysis.loaded_subset_span[0]
		
		# Extract raw data from both replicates
		rep1_g_imgs = combined_model.chrom1_model.G_imgs
		rep2_g_imgs = combined_model.chrom2_model.G_imgs
		raw_tps1 = combined_model.chrom1_model.config.timepoints
		raw_tps2 = combined_model.chrom2_model.config.timepoints
		
		# Compute raw metrics
		raw1_metric = self._subset_f_metric_bp(rep1_g_imgs, region_params, span_start, metric_type)
		raw2_metric = self._subset_f_metric_bp(rep2_g_imgs, region_params, span_start, metric_type)
		
		# Get deconvolved data
		f_imgs = genome_analysis.loaded_subset_data
		deconv_metric = self._subset_f_metric_bp(f_imgs, region_params, span_start, metric_type)
		
		# Compute predicted data using deconvolution matrices
		f_imgs_shape = f_imgs.shape
		predicted_g1 = combined_model.H1 @ f_imgs.reshape((f_imgs.shape[0], -1))
		predicted_g2 = combined_model.H2 @ f_imgs.reshape((f_imgs.shape[0], -1))
		predicted_g1_imgs = predicted_g1.reshape((predicted_g1.shape[0], *f_imgs_shape[1:]))
		predicted_g2_imgs = predicted_g2.reshape((predicted_g2.shape[0], *f_imgs_shape[1:]))
		
		# Compute predicted metrics
		predicted1_metric = self._subset_f_metric_bp(predicted_g1_imgs, region_params, span_start, metric_type)
		predicted2_metric = self._subset_f_metric_bp(predicted_g2_imgs, region_params, span_start, metric_type)
		
		# Get timepoints for deconvolved data
		i_tps = get_average_timepoints_for_branch(self.config1, self.config2, 'i')
		t_tps = get_average_timepoints_for_branch(self.config1, self.config2, 't')
		b_tps = get_average_timepoints_for_branch(self.config1, self.config2, 'b')
		tb_tps = (t_tps + b_tps) / 2.
		
		# Get indices for deconvolved data
		i_indices = self.config1.i_indices()
		t_indices = self.config1.t_indices()
		b_indices = self.config1.b_indices()
		
		return {
			'raw1': raw1_metric,
			'raw2': raw2_metric,
			'predicted1': predicted1_metric,
			'predicted2': predicted2_metric,
			'deconvolved': deconv_metric,
			'raw_tps1': raw_tps1,
			'raw_tps2': raw_tps2,
			'i_tps': i_tps,
			'tb_tps': tb_tps,
			'i_indices': i_indices,
			't_indices': t_indices,
			'b_indices': b_indices
		}

	def plot_raw_predicted_and_deconvolved_on_axes(self, row_axes, analysis_key, metric_type, region_params, 
												   ylabel="", ylim=None, show_legend=True, 
												   show_titles=True, show_xticks=True):
		"""
		Create a 4-panel comparison plot of raw, predicted, and deconvolved data on provided axes.
		"""

		# Compute all the data
		data = self._compute_metric_for_comparison(analysis_key, metric_type, region_params)

		from src.config import load_default_chrom_configs, retrieve_phase_ticks
		config1, config2 = load_default_chrom_configs()

		color_raw = plt.cm.Greys(0.6)
		color_fit = plt.cm.Reds(0.5)
		color_prom_small = plt.cm.Oranges(0.45)
		color_entropy = plt.cm.Blues(0.75)
		color_footprint = plt.cm.Purples(0.6)
		color_metric = color_prom_small

		if 'entropy' in ylabel.lower():
			color_metric = color_entropy
		elif 'promoter' in ylabel.lower():
			color_metric = color_prom_small
		elif 'footprint' in ylabel.lower():
			color_metric = color_footprint
		else:
			raise ValueError("Unhandled ylabel for plot color")

		# Plot replicate 1
		ax = row_axes[0]
		ax.plot(data['raw_tps1'], data['raw1'], label='Raw', c=color_raw)
		ax.plot(data['raw_tps1'], data['predicted1'], label='Predicted', c=color_fit)
		if ylim is not None:
			ax.set_ylim(ylim)
		if show_titles:
			ax.set_title("Replicate 1")

		ax.set_xlim(data['raw_tps1'][0], data['raw_tps1'][-1])
		if show_legend: ax.legend()
		if ylabel: ax.set_ylabel(ylabel)
		if not show_xticks: ax.set_xticks([])
		ax.set_xticks(np.arange(0,  data['raw_tps1'][-1], 40))
		
		# Plot replicate 2
		ax = row_axes[1]
		ax.plot(data['raw_tps2'], data['raw2'], label='Raw',
			c=color_raw, lw=2)
		ax.plot(data['raw_tps2'], data['predicted2'], label='Predicted', 
			c=color_fit, lw=2)
		ax.set_yticks([])
		if ylim is not None:
			ax.set_ylim(ylim)
		if show_titles:
			ax.set_title("Replicate 2")
		ax.set_xlim(data['raw_tps2'][0], data['raw_tps2'][-1])
		if not show_xticks: ax.set_xticks([])
		else:
			ax.set_xticks(np.arange(0,  data['raw_tps2'][-1], 40))

		# Plot recovery (i branch)
		ax = row_axes[2]
		ax.plot(data['i_tps'], data['deconvolved'][data['i_indices']], 
			c=color_metric, lw=3)
		if ylim is not None:
			ax.set_ylim(ylim)
		ax.set_yticks([])
		ax.set_xlim(data['i_tps'][0], data['i_tps'][-1])
		if show_titles:
			ax.set_title("Recovery branch")
		if not show_xticks: 
			ax.set_xticks([])
		else:
			xticks = retrieve_phase_ticks('i', config1, config2)
			phase_ticks, edge_ticks = xticks

			ax.set_xticks(phase_ticks)
			ax.set_xticklabels(['RG1', 'S', 'G2/M'], fontsize=8)

			ax.set_xticks(edge_ticks, minor=True)

			ax.tick_params(axis='x', which='major', length=0)
			ax.tick_params(axis='x', which='minor', length=10) 

		# Plot mother/daughter (average of t and b branches)
		ax = row_axes[3]
		tb_dat = (data['deconvolved'][data['t_indices']] + data['deconvolved'][data['b_indices']]) / 2.
		ax.plot(data['tb_tps'], tb_dat, c=color_metric, lw=3)
		if ylim is not None:
			ax.set_ylim(ylim)
		ax.set_yticks([])
		ax.set_xlim(data['tb_tps'][0], data['tb_tps'][-1])
		if show_titles:
			ax.set_title("Mother/Daughter")
		if not show_xticks: ax.set_xticks([])
		else:
			xticks = retrieve_phase_ticks('tb', config1, config2)
			phase_ticks, edge_ticks = xticks

			ax.set_xticks(phase_ticks)
			ax.set_xticklabels(['SG1', 'S', 'G2/M'], fontsize=8)

			ax.set_xticks(edge_ticks, minor=True)

			ax.tick_params(axis='x', which='major', length=0)
			ax.tick_params(axis='x', which='minor', length=10) 


	def plot_raw_predicted_and_deconvolved(self, analysis_key, metric_type, region_params, 
										 ylabel="", figsize=(11, 2), ylim=None):
		"""
		Create a 4-panel comparison plot of raw, predicted, and deconvolved data.
		This is a wrapper around plot_raw_predicted_and_deconvolved_on_axes for backward compatibility.
		"""
		
		# Create figure with custom gridspec: 5 columns (including spacer) and n_metrics rows
		fig = plt.figure(figsize=figsize)
		gs = gridspec.GridSpec(1, 6, figure=fig, 
							   width_ratios=[1, 0.1, 1, 0.3, 1, 1],  # Spacer columns at 1 and 3
							   wspace=0.0,  # Small spacing within groups
							   hspace=0.1)  # Spacing between rows
		
		# Create 2D axes array to maintain compatibility with existing code
		row_axes = [
			fig.add_subplot(gs[0]),  # First group, column 1
			fig.add_subplot(gs[2]),  # First group, column 2
			fig.add_subplot(gs[4]),  # Second group, column 1 (skip spacer at index 2)
			fig.add_subplot(gs[5])   # Second group, column 2
		]
		
		
		# Call the new method
		self.plot_raw_predicted_and_deconvolved_on_axes(
			row_axes=row_axes, 
			analysis_key=analysis_key,
			metric_type=metric_type,
			region_params=region_params,
			ylabel=ylabel,
			ylim=ylim
		)

		plt.subplots_adjust(top=0.7)
		
		return fig

	def plot_gene_metrics_grouped(self, gene_name, metrics_to_plot=None, analysis_key=None,
									figsize=None, ylims=None, show_column_titles=True):
		"""
		Create a grouped plot for a single gene showing multiple metrics in separate rows.
		fig = locus_vignette.plot_gene_metrics_grouped('CLB2', metrics)
		"""

		default_metrics = [
			{'region_type': 'promoter', 'metric_type': 'mean', 'label': 'Promoter occupancy'},
			{'region_type': 'gene_body', 'metric_type': 'entropy', 'label': 'Gene body entropy'}
		]

		if metrics_to_plot is None:
			metrics_to_plot = default_metrics
		
		n_metrics = len(metrics_to_plot)

		if figsize is None:
			if n_metrics == 1:
				figsize = (11, 2)
			elif n_metrics == 2:
				figsize = (11, 3.2)
			else:
				raise ValueError("Not handled number of metrics to plot")

		gene_regions = self.gene_regions[analysis_key]
		
		# Create figure with custom gridspec: 5 columns (including spacer) and n_metrics rows
		fig = plt.figure(figsize=figsize)
		gs = gridspec.GridSpec(n_metrics, 6, figure=fig, 
							   width_ratios=[1, 0.1, 1, 0.3, 1, 1],  # Spacer columns at 1 and 3
							   wspace=0.0,  # Small spacing within groups
							   hspace=0.1)  # Spacing between rows
		
		# Create 2D axes array to maintain compatibility with existing code
		axes = []
		for row in range(n_metrics):
			row_axes = [
				fig.add_subplot(gs[row, 0]),  # First group, column 1
				fig.add_subplot(gs[row, 2]),  # First group, column 2
				fig.add_subplot(gs[row, 4]),  # Second group, column 1 (skip spacer at index 2)
				fig.add_subplot(gs[row, 5])   # Second group, column 2
			]
			axes.append(row_axes)
		
		# Convert to numpy array for compatibility (optional)
		import numpy as np
		axes = np.array(axes)
		
		# Handle case where there's only one metric (maintain same structure)
		if n_metrics == 1:
			axes = axes.reshape(1, -1)
		
		# Plot each metric
		for i, metric_spec in enumerate(metrics_to_plot):
			region_type = metric_spec['region_type']
			metric_type = metric_spec['metric_type']
			
			# Get region parameters
			if region_type == 'promoter':
				region_params = gene_regions[gene_name]['promoter_bp_tuple']
			elif region_type == 'gene_body':
				region_params = gene_regions[gene_name]['gene_body_bp_tuple']
			else:
				raise ValueError("region_type must be 'promoter' or 'gene_body'")
			
			label = f"{region_type.replace('_', ' ').title()}"
			if metric_type == 'entropy':
				label += "\nentropy"
			else:
				label += "\noccupancy"
			
			# Determine ylim
			ylim = None
			if ylims and label in ylims:
				ylim = ylims[label]
			elif 'ylim' in metric_spec:
				ylim = metric_spec['ylim']
			else:
				# Set default ylim based on metric type
				ylim = (0, 2.5) if metric_type == 'mean' else (4, 6.0)

			# Plot on the current row
			row_axes = axes[i]
			self.plot_raw_predicted_and_deconvolved_on_axes(
				row_axes=row_axes,
				analysis_key=analysis_key,
				metric_type=metric_type,
				region_params=region_params,
				ylabel=label,
				ylim=ylim,
				show_legend=(i == 0),  # Only show legend on first row
				show_titles=(i == 0 and show_column_titles),  # Only show titles on first row
				show_xticks=(i == len(axes)-1) # Show xticks on last row
			)
		
		# Set overall title
		from src.sgd import get_gene_title_name
		gene_title = get_gene_title_name(gene_name)
		plt.suptitle(f"{gene_title}", fontweight='demi', fontsize=18)
		plt.subplots_adjust(top=(0.79 if len(axes) == 2 else 0.7))
		
		return fig


	def plot_multiple_genes_comparison(self, gene_names, metric_spec, analysis_key='efficient_no_copy',
									 figsize=None, ylim=None, colors=None):
		"""
		Create a comparison plot showing the same metric across multiple genes.
		"""
		
		n_genes = len(gene_names)
		
		# Auto-calculate figsize if not provided
		if figsize is None:
			figsize = (4 * 4, 2 * n_genes)  # 4 columns, height scales with genes
		
		# Create figure
		fig, axes = plt.subplots(n_genes, 4, figsize=figsize, sharex=True)
		
		# Handle single gene case
		if n_genes == 1:
			axes = axes.reshape(1, -1)
		
		# Generate colors if not provided
		if colors is None:
			colors = plt.cm.Set1(np.linspace(0, 1, n_genes))

		gene_regions = self.gene_regions[analysis_key]
		
		# Plot each gene
		for i, gene_name in enumerate(gene_names):
			if gene_name not in self.gene_regions:
				print(f"Warning: Gene {gene_name} not found in stored regions, skipping")
				continue
			
			# Get region parameters
			region_type = metric_spec['region_type']
			if region_type == 'promoter':
				region_params = gene_regions[gene_name]['promoter_bp_tuple']
			elif region_type == 'gene_body':
				region_params = gene_regions[gene_name]['gene_body_bp_tuple']
			else:
				raise ValueError("region_type must be 'promoter' or 'gene_body'")
			
			# Generate label
			metric_type = metric_spec['metric_type']
			label = f"{gene_name}"
			
			# Plot on the current row
			row_axes = axes[i]
			self.plot_raw_predicted_and_deconvolved_on_axes(
				row_axes=row_axes,
				analysis_key=analysis_key,
				metric_type=metric_type,
				region_params=region_params,
				ylabel=label,
				ylim=ylim,
				show_legend=(i == 0),  # Only show legend on first row
				show_titles=(i == 0)   # Only show titles on first row
			)
		
		# Set overall title
		region_label = metric_spec['region_type'].replace('_', ' ')
		metric_label = 'entropy' if metric_spec['metric_type'] == 'entropy' else 'occupancy'
		plt.suptitle(f"{region_label.title()} {metric_label} comparison", fontweight='demi', fontsize=16)
		
		# Adjust layout
		plt.tight_layout()
		
		return fig

	
	def plot_origin_footprint_deconvolution(self, analysis_key='efficient_no_copy', ylim=(0, 3.2)):
		"""Convenience method for plotting origin footprint deconvolution."""
		footprint_bp_tuple = (self.origin.pos-100, 30, self.origin.pos+20, 120)
		fig = self.plot_raw_predicted_and_deconvolved(
			analysis_key=analysis_key,
			metric_type='mean',
			region_params=footprint_bp_tuple,
			ylabel="Footprint occupancy",
			ylim=ylim)
		plt.suptitle(f"{self.get_origin_name()}", fontweight='demi', fontsize=18)

		return fig

	def plot_total_window_comparision(self, keys=['efficient_no_copy','distal_no_copy']):
		fig = self.compare_loaded_branch_means(self.genome_analyses[keys[0]],
											   self.genome_analyses[keys[1]],
											   with_b=[False, False],
											   labels=['Proximal', 'Distal'],
											   plot_branches=['i', 'tb'],
											   branch_names=['Recovery', 'Mother/Daughter'])
		plt.suptitle("Proximal vs distal occupancy", fontsize=16,
			fontweight='demi')
		plt.tight_layout()
		plt.subplots_adjust(wspace=0, top=0.8)
		return fig
	
	def plot_gene_deconvolution(self, gene_name, region_type, analysis_key='efficient_no_copy', 
							  metric_type='mean', plus_one_adjustment=0, region_params=None, ylim=None):
		"""
		Convenience method for plotting gene region deconvolution.
		
		Parameters:
		-----------
		gene_name : str
			Name of the gene (e.g., 'CLB5', 'CLB2')
		region_type : str
			'promoter' or 'gene_body'
		analysis_key : str
			Which analysis to use
		metric_type : str
			'mean' for occupancy or 'entropy' for positional entropy
		plus_one_adjustment : int
			Adjustment to the +1 nucleosome position
		region_params : dict or None
			Custom region parameters. If None, uses default parameters.
		ylim : tuple or None
			Y-axis limits
		"""
		from src.sgd import read_geneset_with_computed_regions
		from src.reference_data import load_plus_ones
		
		# Load gene data
		genes = read_geneset_with_computed_regions()
		genes = genes.join(load_plus_ones())
		
		# Get gene with adjustment
		gene = genes[genes['gene'] == gene_name].iloc[0].copy()
		gene['combined_+1'] += plus_one_adjustment
		
		# Define default parameters if not provided
		if region_params is None:
			if region_type == 'promoter':
				region_params = {'start_offset': -500, 'end_offset': 0, 'y_start': 0, 'y_end': 140}
			elif region_type == 'gene_body':
				region_params = {'start_offset': 0, 'end_offset': 500, 'y_start': 90, 'y_end': 260}
			else:
				raise ValueError("region_type must be 'promoter' or 'gene_body'")
		
		# Convert to bp_tuple
		plus_one = gene['combined_+1']
		bp_tuple = (
			plus_one + region_params['start_offset'],
			region_params['y_start'],
			plus_one + region_params['end_offset'],
			region_params['y_end']
		)
		
		# Set default ylim based on metric type
		if ylim is None:
			ylim = (0, 2.5) if metric_type == 'mean' else (4, 6.0)
		
		ylabel = f"{gene_name} {region_type.replace('_', ' ')}"
		if metric_type == 'entropy':
			ylabel += " entropy"
		else:
			ylabel += " occupancy"
		
		return self.plot_raw_predicted_and_deconvolved(
			analysis_key=analysis_key,
			metric_type=metric_type,
			region_params=bp_tuple,
			ylabel=ylabel,
			ylim=ylim
		)

	def get_origin_name(self):
		origin = self.origin
		ars_name_mapping = {
			'ARS1626.5': 'ARS1635' # Alias term in Belksy set compared to SGD R64
		}
		if origin.ars_name in ars_name_mapping.keys():
			ars_name = ars_name_mapping[origin.ars_name]
		else:
			ars_name = origin.ars_name.ars_name
		return ars_name

	def plot_locus(self, analysis_key, with_copy_correction=False,
		highlight_bins=[]):
		from src.config import load_default_chrom_configs
		from src.figure_configs import save_figure_for_paper

		config1, config2 = load_default_chrom_configs()

		if analysis_key in ['efficient_with_copy', 'efficient_no_copy']:

			ars_name = self.get_origin_name()
			title = "Deconvolved chromatin at efficient origin $\\it{" + ars_name + "}$"

			# with_copy_correction = analysis_key == 'efficient_with_copy'
			# if with_copy_correction: title += ", with copy correction"

		else:

			title = "Distal (~40kb) from efficient origin"

			# with_copy_correction = analysis_key == 'distal_with_copy'
			# if with_copy_correction: title += ", with copy correction"
			
		genome_analysis = self.genome_analyses[analysis_key]
		fig = genome_analysis.plot_loaded_data(config1, figsize=(12, 12), highlight_bins=highlight_bins)

		plt.suptitle(title, fontweight='demi', fontsize=32)
		plt.tight_layout()
		
		return fig


	def plot_copy_number_comparison(self):
		fig = self.compare_loaded_branch_means(self.genome_analysis_with_copy, 
			self.genome_analysis_no_copy)
		plt.suptitle("Total occupancy, efficient origin $\\it{" + self.origin.ars_name + "}$", 
			y=1.15, fontweight='demi', fontsize=16)
		return fig

	def compare_loaded_branch_means(self, analysis1, analysis2, with_b=[True, False],
		labels=['1', '2'], plot_branches=['i', 't', 'b'], branch_names=['Recovery', 'Mother', 'Daughter']):
		"""Load the branch means for the locus of the two runs and plot the comparison."""
		from src.config import load_default_chrom_configs

		config1, config2 = load_default_chrom_configs()

		from src.RealDataReplication import read_n_fr_b

		# B is independent of replicate, but set to 1 (req. argument)
		nfrb1 = read_n_fr_b(analysis1.chrom, analysis1.loaded_subset_span, 1, 'output/draft3_run/')
		b1 = nfrb1[3]

		nfrb2 = read_n_fr_b(analysis2.chrom, analysis2.loaded_subset_span, 1, 'output/draft3_run/')
		b2 = nfrb2[3]

		def plot_branch_mean(branch, colors=['red', 'blue']):

			mean_data1 = analysis1.loaded_subset_data.mean((1, 2))
			mean_data2 = analysis2.loaded_subset_data.mean((1, 2))

			if branch in ['i', 't', 'b']:

				if branch == 'i':
					branch_indices = config1.i_indices()
					tps = config1.get_timepoints_for_branch('i')
				elif branch == 't':
					branch_indices = config1.t_indices()
					tps = config1.get_timepoints_for_branch('t')
				elif branch == 'b':
					branch_indices = config1.b_indices()
					tps = config1.get_timepoints_for_branch('b')
		
				dat1 = mean_data1[branch_indices]
				dat2 = mean_data2[branch_indices]

			elif branch == 'tb':
				tps = config1.get_timepoints_for_branch('t')

				dat1 = (mean_data1[config1.t_indices()]+mean_data1[config2.b_indices()])/2
				dat2 = (mean_data2[config1.t_indices()]+mean_data2[config2.b_indices()])/2

			# Enable or disable the multiplier of b for each plot
			bs = [b1, b2]
			bs[0] = bs[0] if with_b[0] else 1
			bs[1] = bs[1] if with_b[1] else 1

			plt.plot(tps, dat1 * bs[0], label=labels[0], color=colors[0], lw=2.5)
			plt.plot(tps, dat2 * bs[1], label=labels[1], color=colors[1], lw=2.5)

			plt.legend(ncol=2)
			plt.ylim(0.5, 1.5)
			plt.xlim(tps[0], tps[-1])

		num_cols = len(plot_branches)

		fig = plt.figure(figsize=(9, 3))

		eff_color = plt.cm.Reds(0.65)
		distal_color = plt.cm.Blues(0.65)

		for col in range(num_cols):
			plt.subplot(1, num_cols, col+1)

			plot_branch_mean(plot_branches[col], colors=[
				eff_color, distal_color])

			if col == 0:
				plt.ylabel("Normalized occupancy")

			plt.title(branch_names[col])
			if col > 0: plt.yticks([])

		return fig

	# Add these methods to your LocusVignette class
	def define_gene_configurations(self, gene_configs=None):
		"""
		Define gene configurations for region analysis.
		
		Parameters:
		-----------
		gene_configs : dict or None
			Dictionary of gene configurations. If None, uses default configurations.
			Format:
			{
				'gene_name': {
					'plus_one_adjustment': int,
					'promoter_params': {'start_offset': int, 'end_offset': int, 'y_start': int, 'y_end': int},
					'gene_body_params': {'start_offset': int, 'end_offset': int, 'y_start': int, 'y_end': int}
				}
			}
		"""
		if gene_configs is None:
			# Default configurations based on your notebook
			self.gene_configs = {
				'CLB5': {
					'plus_one_adjustment': 260,
					'promoter_params': {'start_offset': 0, 'end_offset': 300, 'y_start': 0, 'y_end': 140},
					'gene_body_params': {'start_offset': -500, 'end_offset': 0, 'y_start': 90, 'y_end': 260}
				},
				'CLB2': {
					'plus_one_adjustment': -40,
					'promoter_params': {'start_offset': -500, 'end_offset': 0, 'y_start': 0, 'y_end': 140},
					'gene_body_params': {'start_offset': 0, 'end_offset': 500, 'y_start': 90, 'y_end': 260}
				},
				'THI22': {
					'plus_one_adjustment': -60,
					'promoter_params': {'start_offset': -500, 'end_offset': 0, 'y_start': 0, 'y_end': 140},
					'gene_body_params': {'start_offset': 0, 'end_offset': 500, 'y_start': 90, 'y_end': 260}
				},
				'SNT309': {
					'plus_one_adjustment': -80,
					'promoter_params': {'start_offset': -100, 'end_offset': 0, 'y_start': 0, 'y_end': 140},
					'gene_body_params': {'start_offset': 0, 'end_offset': 500, 'y_start': 90, 'y_end': 260}
				},
				'PRE2': {
					'plus_one_adjustment': -0,
					'promoter_params': {'start_offset': -280, 'end_offset': 0, 'y_start': 0, 'y_end': 140},
					'gene_body_params': {'start_offset': 0, 'end_offset': 500, 'y_start': 90, 'y_end': 260}
				}
			}
		else:
			self.gene_configs = gene_configs

	def compute_and_store_regions_all_analyses(self):
		"""Store metrics for all regions and analyses"""
		for key in self.genome_analyses.keys():
			self.compute_and_store_regions(key)


	def compute_and_store_regions(self, analysis_key='efficient_no_copy', include_origin=True):
		"""
		Compute and store all gene regions and origin footprint as member variables.
		
		Parameters:
		-----------
		analysis_key : str
			Which analysis to use for span information
		include_origin : bool
			Whether to include origin footprint region
		"""
		from src.sgd import read_nondubious_genes_dataset
		from src.reference_data import load_plus_ones
		
		# Load gene data
		genes = read_nondubious_genes_dataset()
		genes = genes.join(load_plus_ones())
		
		# Get span start for coordinate conversion
		span_start = self.genome_analyses[analysis_key].loaded_subset_span[0]

		gene_regions = self.gene_regions[analysis_key]
		
		# Process each configured gene
		for gene_name, config in self.gene_configs.items():
			try:
				# Get gene with adjustment
				gene = genes[genes['gene'] == gene_name].iloc[0].copy()
				gene['combined_+1'] += config['plus_one_adjustment']
				
				# Compute bp_tuples
				plus_one = gene['combined_+1']
				
				promoter_bp_tuple = (
					plus_one + config['promoter_params']['start_offset'],
					config['promoter_params']['y_start'],
					plus_one + config['promoter_params']['end_offset'],
					config['promoter_params']['y_end']
				)
				
				gene_body_bp_tuple = (
					plus_one + config['gene_body_params']['start_offset'],
					config['gene_body_params']['y_start'],
					plus_one + config['gene_body_params']['end_offset'],
					config['gene_body_params']['y_end']
				)
				
				# Store regions
				gene_regions[gene_name] = {
					'promoter_bp_tuple': promoter_bp_tuple,
					'gene_body_bp_tuple': gene_body_bp_tuple,
					'gene_data': gene,  # Store for reference
					'config': config    # Store config for reference
				}
				
			except IndexError:
				print(f"Warning: Gene {gene_name} not found in dataset")
				continue
		
		# Store origin footprint if requested
		if include_origin:

			# Footprint selection specific to ARS1635
			self.origin_footprint_bp_tuple = (
				self.origin.pos - 100, 30, 
				self.origin.pos + 20, 120
			)
			print("Stored origin footprint region")
		print(f"Stored regions for {len(gene_regions)} genes")

		
	def get_highlight_bins_for_plotting(self, analysis_key, color_scheme=None, regions_to_plot=None,
		include_origin=True):
		"""
		Generate highlight_bins list for use with plot_locus().
		"""
		if not hasattr(self, 'gene_regions'):
			raise ValueError("Must call compute_and_store_regions() first")
		
		# Default color scheme
		if color_scheme is None:
			import matplotlib.pyplot as plt
			color_scheme = {
				'CLB2': plt.cm.Blues(0.57),
				'CLB5': plt.cm.Reds(0.67),
				'THI22': plt.cm.Blues(0.9),
				'SNT309': plt.cm.Blues(0.9),
				'PRE22': plt.cm.Reds(0.9),
				'origin': 'gray'
			}
		
		highlight_bins = []
		
		# Add origin footprint if requested and it exists
		if include_origin and hasattr(self, 'origin_footprint_bp_tuple'):
			origin_color = color_scheme.get('origin', 'gray')
			highlight_bins.append((origin_color, self.origin_footprint_bp_tuple))
		
		# Determine which regions to plot
		if regions_to_plot is None:
			# Default: include all regions for all genes
			regions_to_plot = {gene_name: ['promoter_bp_tuple', 'gene_body_bp_tuple'] 
							  for gene_name in self.gene_regions.keys()}
		
		gene_regions = self.gene_regions[analysis_key]

		# Add specified gene regions
		for gene_name, region_types in regions_to_plot.items():
			if gene_name not in gene_regions:
				print(f"Warning: Gene {gene_name} not found in stored regions, skipping")
				continue
				
			gene_color = color_scheme.get(gene_name, 'black')
			regions = gene_regions[gene_name]
			
			for region_type in region_types:
				if region_type in regions:
					highlight_bins.append((gene_color, regions[region_type]))
				else:
					print(f"Warning: Region type {region_type} not found for gene {gene_name}, skipping")
		
		return highlight_bins

	def compute_gene_metrics_from_stored_regions(self, analysis_key='efficient_no_copy'):
		"""
		Compute metrics for all stored gene regions.
		
		Parameters:
		-----------
		analysis_key : str
			Which analysis to use
			
		Returns:
		--------
		dict : Dictionary of computed metrics for each gene
		"""
		if not hasattr(self, 'gene_regions'):
			raise ValueError("Must call compute_and_store_regions() first")
		
		# Get data
		f_imgs = self.genome_analyses[analysis_key].loaded_subset_data
		span_start = self.genome_analyses[analysis_key].loaded_subset_span[0]
		
		gene_metrics = {}
		
		for gene_name, regions in self.gene_regions.items():
			promoter_bp_tuple = regions['promoter_bp_tuple']
			gene_body_bp_tuple = regions['gene_body_bp_tuple']
			
			metrics = {
				'promoter_occupancy': self._subset_f_metric_bp(f_imgs, promoter_bp_tuple, span_start, 'mean'),
				'gene_body_occupancy': self._subset_f_metric_bp(f_imgs, gene_body_bp_tuple, span_start, 'mean'),
				'gene_body_entropy': self._subset_f_metric_bp(f_imgs, gene_body_bp_tuple, span_start, 'entropy')
			}
			
			gene_metrics[gene_name] = metrics
		
		return gene_metrics

	def add_gene_to_analysis(self, gene_name, plus_one_adjustment=0, 
							promoter_params=None, gene_body_params=None):
		"""
		Add a new gene to the analysis configuration.
		
		Parameters:
		-----------
		gene_name : str
			Name of the gene to add
		plus_one_adjustment : int
			Adjustment to the +1 nucleosome position
		promoter_params : dict
			Promoter region parameters
		gene_body_params : dict
			Gene body region parameters
		"""
		# Default parameters
		if promoter_params is None:
			promoter_params = {'start_offset': -500, 'end_offset': 0, 'y_start': 0, 'y_end': 140}
		if gene_body_params is None:
			gene_body_params = {'start_offset': 0, 'end_offset': 500, 'y_start': 90, 'y_end': 260}
		
		# Initialize gene_configs if it doesn't exist
		if not hasattr(self, 'gene_configs'):
			self.gene_configs = {}
		
		self.gene_configs[gene_name] = {
			'plus_one_adjustment': plus_one_adjustment,
			'promoter_params': promoter_params,
			'gene_body_params': gene_body_params
		}
		
		print(f"Added {gene_name} to gene configurations")

	def plot_locus_with_stored_regions(self, analysis_key="efficient_with_copy", 
									 color_scheme=None, regions_to_plot=None, include_origin=True):
		"""
		Plot locus using stored gene regions for highlighting.
		"""
		if hasattr(self, 'gene_regions') or hasattr(self, 'origin_footprint_bp_tuple'):
			highlight_bins = self.get_highlight_bins_for_plotting(analysis_key, color_scheme, 
				regions_to_plot, include_origin)
		else:
			highlight_bins = []
		
		fig = self.plot_locus(analysis_key=analysis_key, 
					   highlight_bins=highlight_bins)
		return fig

	def plot_stored_gene_deconvolution(self, gene_name, region_type, 
			analysis_key='efficient_no_copy', metric_type='mean', ylim=None):
		"""
		Plot gene deconvolution using stored regions.
		
		Parameters:
		-----------
		gene_name : str
			Name of the gene (must be in stored regions)
		region_type : str
			'promoter' or 'gene_body'
		analysis_key : str
			Which analysis to use
		metric_type : str
			'mean' for occupancy or 'entropy' for positional entropy
		ylim : tuple or None
			Y-axis limits
		"""

		gene_regions = self.gene_regions[analysis_key]
		
		if region_type == 'promoter':
			region_params = gene_regions[gene_name]['promoter_bp_tuple']
		elif region_type == 'gene_body':
			region_params = gene_regions[gene_name]['gene_body_bp_tuple']
		else:
			raise ValueError("region_type must be 'promoter' or 'gene_body'")
		
		# Set default ylim based on metric type
		if ylim is None:
			ylim = (0, 2.5) if metric_type == 'mean' else (4, 6.0)
		
		ylabel = f"{gene_name} {region_type.replace('_', ' ')}"
		if metric_type == 'entropy':
			ylabel += " entropy"
		else:
			ylabel += " occupancy"
		
		return self.plot_raw_predicted_and_deconvolved(
			analysis_key=analysis_key,
			metric_type=metric_type,
			region_params=region_params,
			ylabel=ylabel,
			ylim=ylim
		)

	def plot_distal_gene_metrics(self):
		"""Plot the distal gene metrics"""

		fig1 = self.plot_gene_metrics_grouped('PRE2', analysis_key='distal_with_copy')

		fig2 = self.plot_gene_metrics_grouped('SNT309', analysis_key='distal_with_copy', 
			metrics_to_plot=[
				{'region_type': 'gene_body', 'metric_type': 'entropy', 
				'label': 'Gene body entropy',
					 'ylim': (4, 6)}
			])

		return [fig1, fig2]

	def plot_proximal_gene_metrics(self):
		"""Plot the proximal gene metrics"""

		figs = []

		gb_entropy_metrics_only = [{'region_type': 'gene_body', 'metric_type': 'entropy', 
			'label': 'Gene body entropy'}]
		
		fig1 = self.plot_gene_metrics_grouped('CLB5', analysis_key='efficient_with_copy')
		figs.append(fig1)
		
		fig2 = self.plot_gene_metrics_grouped('THI22', 
			metrics_to_plot=gb_entropy_metrics_only, analysis_key='efficient_with_copy')
		figs.append(fig2)

		fig3 = self.plot_origin_footprint_deconvolution('efficient_with_copy')
		figs.append(fig3)
		
		return figs

	def plot_distal_locus(self):
		subset_regions = {'PRE2': ['gene_body_bp_tuple'],
						  'SNT309': ['gene_body_bp_tuple']}
		fig_distal = self.plot_locus_with_stored_regions(analysis_key='distal_with_copy', 
			regions_to_plot=subset_regions)

	def plot_proximal_locus(self):
		subset_regions = {'CLB5': ['gene_body_bp_tuple', 'promoter_bp_tuple'],
						  'THI22': ['gene_body_bp_tuple']}
		fig_proximal = self.plot_locus_with_stored_regions(analysis_key='efficient_with_copy', 
			regions_to_plot=subset_regions)

	def run_and_save_all(self):
		"""
		Create save directory and save all plots from the notebook analysis.
		"""
		# Create save directory if it doesn't exist
		os.makedirs(self.save_directory, exist_ok=True)
		print(f"Created/verified save directory: {self.save_directory}")
		
		# Save distal locus plot
		self.plot_distal_locus()
		save_path = os.path.join(self.save_directory, 'Distal_Locus_Plot.png')
		save_figure_for_paper(save_path)
		print(f"Saved: Distal_Locus_Plot.png")
		
		# Save proximal locus plot
		self.plot_proximal_locus()
		save_path = os.path.join(self.save_directory, 'Proximal_Locus_Plot.png')
		save_figure_for_paper(save_path)
		print(f"Saved: Proximal_Locus_Plot.png")

		# Save proximal gene metrics
		save_names = ['Proximal_Clb5_metrics.png', 'Proximal_Thi22_Metrics.png', 
			'Origin_Footprint_Deconvolution.png']
		figs_proximal = self.plot_proximal_gene_metrics()
		for i, fig in enumerate(figs_proximal):
			save_path = os.path.join(self.save_directory, save_names[i])
			save_figure_for_paper(save_path, fig=fig)
			print(f"Saved: {save_path}")
		
		# Save distal gene metrics
		fig_distal_metrics = self.plot_distal_gene_metrics()
		save_names = ['Distal_Snt309_metrics.png', 'Distal_Pre22_metrics.png']
		for i, fig in enumerate(fig_distal_metrics):
			save_path = os.path.join(self.save_directory, save_names[i])
			save_figure_for_paper(save_path, fig=fig)
			print(f"Saved: {save_path}")
		
		# Save total window comparison
		fig_comparison = self.plot_total_window_comparision()
		save_path = os.path.join(self.save_directory, 'Total_Window_Comparison.png')
		save_figure_for_paper(save_path)
		print(f"Saved: Total_Window_Comparison.png")
		
		print(f"\nAll plots saved successfully to: {self.save_directory}")

	def layout_panel_proximal(self, canvas_width=1024, canvas_height=540, margins=20, 
						column_padding=30, between_padding=30, debug_mode=True):
		"""
		Create a composite figure panel with all locus plots and metrics.
		Layout: proximal locus on left, three other plots stacked vertically on right.
		"""
		# Import compositor and helper functions
		# Note: You may need to adjust these import paths based on your project structure
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, layout_images_vertically, \
			add_panel_labels_to_images
		
		# Create compositor 
		compositor = FigureCompositor(canvas_width, canvas_height, debug_mode=debug_mode)
		
		# Define file paths
		proximal_locus_path = f'{self.save_directory}/Proximal_Locus_Plot.png'
		clb5_metrics_path = f'{self.save_directory}/Proximal_Clb5_Metrics.png'
		thi22_metrics_path = f'{self.save_directory}/Proximal_Thi22_Metrics.png'
		origin_footprint_path = f'{self.save_directory}/Origin_Footprint_Deconvolution.png'
		
		# Calculate column widths (left column gets ~50% of available width)
		available_width = canvas_width - (2 * margins)
		left_column_width = int((available_width - column_padding) * 0.49)
		right_column_width = available_width - left_column_width - column_padding
		
		# Place proximal locus on the left
		left_column_images = layout_images_vertically(
			compositor,
			[proximal_locus_path],
			between_padding=between_padding,
			margin=margins,
			x_position=margins,
			widths=[left_column_width],
			image_keys=['proximal_locus']
		)
		
		# Calculate starting x position for right column
		right_column_start_x = margins + left_column_width + column_padding
		
		# Layout right column (three plots stacked vertically)
		right_column_images = layout_images_vertically(
			compositor,
			[clb5_metrics_path, thi22_metrics_path, origin_footprint_path],
			between_padding=between_padding,
			margin=margins,  # Use same top margin as left column
			x_position=right_column_start_x,
			widths=[right_column_width] * 3,  # All three images same width
			image_keys=['clb5_metrics', 'thi22_metrics', 'origin_footprint']
		)
		
		# Add panel labels (A-D) to top-left of each image
		add_panel_labels_to_images(
			compositor,
			compositor.placed_images,
			labels='ABCD',
			font_size=36,
			offset=(-10, -12),  # Slightly above and to the left of each image
			font_type='bold',
			color=(0, 0, 0)
		)
		
		# Save the composite figure
		output_path = f'{self.figures_directory}/Figure3_Locus.png'
		compositor.save(output_path)
		print(f"Combined panel saved to: {output_path}")
		
		return compositor



	def layout_panel_full(self, canvas_width=1024, canvas_height=1120, margins=20, 
					column_padding=30, between_padding=30, debug_mode=True):
		"""
		Create a composite figure panel with all locus plots and metrics.
		"""
		# Import compositor and helper functions
		# Note: You may need to adjust these import paths based on your project structure
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, layout_images_vertically, add_panel_labels_to_images
		
		# Create compositor 
		compositor = FigureCompositor(canvas_width, canvas_height, debug_mode=debug_mode)
		
		# Define file paths
		proximal_locus_path = f'{self.save_directory}/Proximal_Locus_Plot.png'
		distal_locus_path = f'{self.save_directory}/Distal_Locus_Plot.png'
		clb5_metrics_path = f'{self.save_directory}/Proximal_Clb5_Metrics.png'
		thi22_metrics_path = f'{self.save_directory}/Proximal_Thi22_Metrics.png'
		origin_footprint_path = f'{self.save_directory}/Origin_Footprint_Deconvolution.png'
		pre2_metrics_path = f'{self.save_directory}/Distal_Pre22_Metrics.png'
		snt309_metrics_path = f'{self.save_directory}/Distal_Snt309_metrics.png'

		# Removing total plots, switching to show results as with copy correction
		# so total window motivation will be moved to the copy correction figure
		# total_comparison_path = f'{self.save_directory}/Total_Window_Comparison.png'
		
		# Layout top row (A and B) horizontally
		top_row_images = layout_images_horizontally(
			compositor,
			[proximal_locus_path, distal_locus_path],
			width_proportions=[1, 1],  # Equal width for both top images
			between_padding=column_padding,
			margin=margins,
			image_keys=['proximal_locus', 'distal_locus']
		)
		
		# Get information about top row for positioning columns
		proximal_info = top_row_images['proximal_locus']
		distal_info = top_row_images['distal_locus']
		
		# Calculate starting y position for columns
		column_start_y = proximal_info['logical_position'][1] + proximal_info['logical_size'][1] + between_padding
		
		# Layout left column (B, C, D) below proximal locus
		left_column_images = layout_images_vertically(
			compositor,
			[clb5_metrics_path, thi22_metrics_path, origin_footprint_path],
			between_padding=between_padding,
			margin=(0, column_start_y),  # No left margin, use calculated y position as top margin
			x_position=proximal_info['logical_position'][0],
			widths=[proximal_info['logical_size'][0]] * 3,  # Same width as proximal locus image
			image_keys=['clb5_metrics', 'thi22_metrics', 'origin_footprint']
		)
		
		# Layout right column (F, G) below distal locus
		right_column_images = layout_images_vertically(
			compositor,
			[pre2_metrics_path, snt309_metrics_path],
			between_padding=between_padding,
			margin=(0, column_start_y),  # No left margin, use calculated y position as top margin
			x_position=distal_info['logical_position'][0],
			widths=[distal_info['logical_size'][0]] * 2,  # Same width as distal locus image
			image_keys=['pre2_metrics', 'snt309']
		)
		
		# Add panel labels (A-G) to top-left of each image
		add_panel_labels_to_images(
			compositor,
			compositor.placed_images,
			labels='AEBCDFGH',
			font_size=36,
			offset=(-10, -12),  # Slightly above and to the left of each image
			font_type='bold',
			color=(0, 0, 0)
		)
		
		# Save the composite figure
		output_path = f'{self.figures_directory}/Figure3_Loci.png'
		compositor.save(output_path)
		print(f"Combined panel saved to: {output_path}")
		
		return compositor


def plot_gene_expression_for_example_genes():
	"""For reference, the raw gene expression for these genes may be useful"""
	from src.sgd import get_orfnames
	gene_names = ['CLB5', 'THI22', 'PRE2', 'SNT309']
	orfnames = get_orfnames(gene_names)

	from src.gene_expression import load_gene_expression_data

	raw_expression_rep1 = load_gene_expression_data(1)
	raw_expression_rep2 = load_gene_expression_data(2)

	plt.figure(figsize=(9, 2))
	plt.subplot(1, 2, 1)
	plt.plot(raw_expression_rep1.loc[orfnames].values.T, label=gene_names)
	plt.legend()
	plt.title("Raw gene expression, replicate 1")

	plt.subplot(1, 2, 2)
	plt.plot(raw_expression_rep2.loc[orfnames].values.T, label=gene_names)
	plt.legend()
	plt.title("Raw gene expression, replicate 2")
