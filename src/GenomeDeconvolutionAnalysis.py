import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.reference_data import load_p1_gene_regions
from src.sgd import get_orfname
from src.deconvolved_chromatin_loader import DeconvolvedChromatinDataLoader


class GenomeDeconvolutionAnalysis:
	"""
	Class to perform analysis on genome-wide deconvolution results.
	
	Functions:
	load_mnase_span - Load the genomic data for a given chromosome and span,
					 e.g., loading a gene's chromatin context.
	"""
	
	def __init__(self, outdir, chromatin_data_dir=None):
		"""
		Initialize the GenomeDeconvolutionAnalysis.
		
		Parameters
		----------
		outdir : str
			Base directory containing deconvolved data files
		"""
		self.outdir = outdir
		
		# Use the output directory and default chromatin data path
		if chromatin_data_dir is None:
			self.data_loader = DeconvolvedChromatinDataLoader(outdir)

		# If specified, set the chromatin data directory data directly
		# for loading non-copy corrected data
		else:
			self.data_loader = DeconvolvedChromatinDataLoader(outdir, chromatin_data_dir=chromatin_data_dir)
		
		# These will be set by load_mnase_span for backward compatibility
		self.chrom = None
		self.loaded_subset_data = None
		self.loaded_subset_span = None
	
	def load_mnase_span(self, chrom, mnase_span, window_size=10000):
		"""
		Load the MNase data for a given span with caching.
		
		Args:
			chrom: Chromosome name/number
			mnase_span: Tuple of (start, end) positions to load
			window_size: Size of the windows (default: 10000)
		
		Returns:
			Tuple of (loaded_data, actual_span_loaded) or (None, None) if no data is loaded
		"""
		# Update window size if different from default
		if window_size != self.data_loader.window_size:
			self.data_loader.window_size = window_size
		
		# Use the data loader to load the span
		loaded_data, loaded_span = self.data_loader.load_mnase_span(chrom, mnase_span)
		
		# Store the results as instance variables for backward compatibility
		if loaded_data is not None:
			self.chrom = chrom
			self.loaded_subset_data = loaded_data
			self.loaded_subset_span = loaded_span
		
		return loaded_data, loaded_span
	
	def clear_cache(self):
		"""Clear the window cache."""
		self.data_loader.clear_cache()
	
	def plot_gene(self, gene_name, config1, analysis=None):
		"""
		Plot chromatin data for a specific gene.
		
		Parameters
		----------
		gene_name : str
			Name of the gene to plot
		config1 : object
			Configuration object
		analysis : object, optional
			Analysis object containing deconvolved gene expression data
		"""
		gene_metric_regions = load_p1_gene_regions()
		orfname = get_orfname(gene_name)
		gene = gene_metric_regions.loc[orfname]
		
		chrom = gene.chr
		mnase_span = gene['combined_+1'] - 1000, gene['combined_+1'] + 1000
		
		chromatin_gene_data_F, loaded_span = self.load_mnase_span(chrom, mnase_span)
		
		if analysis is not None:
			expression_f = analysis.deconvolved_genes_F.loc[orfname].values
			self.gene_expression_data = expression_f
		
		from src.sgd import get_gene_title_name
		title = get_gene_title_name(orfname)
		plotter = self.plot_loaded_data(config1, expression_f, title)
		
		return plotter
	
	def plot_loaded_data(self, config=None, expression_f=None, title=None,
						figsize=(15, 5), highlight_bins=[], 
						branch_type='mean_mother_daughter',
						rna_plotter=None, tpm_plotter=None, 
						plot_index_labels=True,
						tfs=[]):
		"""
		Plot the loaded chromatin data.
		
		Parameters
		----------
		config : object
			Configuration object
		expression_f : array-like, optional
			Expression data to plot alongside chromatin
		title : str, optional
			Title for the plot
		figsize : tuple, optional
			Figure size (width, height)
		highlight_bins : list, optional
			Bins to highlight in the plot
			
		Returns
		-------
		plotter : DeconvolutionChromatinExpressionPlotter
			The plotter object
		"""
		from src.ChromatinRNALocusPlotter import SingleBranchChromatinPlotter

		if config is None:
			from src.config import load_default_chrom_configs
			config, _ = load_default_chrom_configs()
		
		plotter = SingleBranchChromatinPlotter(
			outdir=self.outdir,
			config1=config,
			figsize=figsize,
			branch_type=branch_type,
			title=title,
			rna_plotter=rna_plotter,
			deconvolved_tpm_plotter=tpm_plotter,
			plot_index_labels=plot_index_labels,
			tfs=tfs,
		)
		plotter.set_chrom_span(self.chrom, self.loaded_subset_span)
		plotter.set_chromatin_data(self.loaded_subset_data)
		plotter.highlight_bins = highlight_bins

		if expression_f is not None:
			plotter.set_expression_data(expression_f)
		
		plotter.plot()

		self.plotter = plotter
		return plotter
