import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from src.sgd import read_nondubious_genes_dataset, read_geneset_with_computed_regions
from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
from src.orf_plotter import load_default_orf_plotter
from src.rna_pileup_plotter import RNASeqPileupPlotter
from src.combined_chromatin_model import CombinedChromatinModel
from src.config import load_default_chrom_configs
from src.transcript_boundary_caller import TranscriptBoundaryCaller
from src.transcript_boundary_visualizer import plot_tx_transcript_context
from pipeline.transcription_processor import ExpressionAnalysisProcessor
from src.deconvolved_tpm_plotter import DeconvolvedTPMPlotter
from pipeline.chromatin_metrics_processor import ChromatinMetricsProcessor
from src.utils import mkdir_safe
from pipeline.expression_chromatin_integration import IntegratedChromatinExpressionAnalyzer
from src.figure_configs import save_figure_for_paper


class FigureNongenicTranscripts:
	"""
	A class for analyzing non-genic and antisense transcripts genome-wide.
	
	Performs filtering, characterization, and visualization of non-genic transcripts
	based on expression levels, chromatin metrics, and overlap with existing genes.
	"""
	
	def __init__(self, output_dir):
		"""
		Initialize the analyzer with data processors and configuration parameters.
		
		Parameters:
		-----------
		output_dir : str
			Path to the output directory containing analysis results
		"""
		# Store output directory
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/nongenic_antisense_figures"
		mkdir_safe(self.save_dir)
		self.figures_dir = f"{self.output_dir}/Figures"

		# Configuration parameters (as member variables)
		self.min_ptr = 0
		self.qthreshold = 0.95
		self.locus_span_size = 1000  # bp on each side for locus plotting
		self.wider_span_size = 3600  # bp for transcript calling visualization
	
		self.select_transcript_examples()

		# Initialize data processors and other components
		self._initialize_processors()
		
		# Load all data upon initialization
		self.load_data()

	def select_transcript_examples(self):

		# Example non-genic transcripts::

		#  transcript_name = 'nogene_chr4_130361_130486' # Antisense transcript, ribosomal gene
		#  transcript_name = 'nogene_chr13_618858_619269' # Antisense transcript, 

		# Decent example of both, there is a lot of gene overlap here
		#('3', 'Antisense & Divergent', 'nogene_chr4_1456150_1456694'),  # Antisense transcript, ribosomal

		# Selected transcripts for scatter plot annotation
		self.selected_transcripts = [
			('1', 'Divergent', 'nogene_chr4_443744_444111'), # Divergently transcribed transcript
			('2', 'Divergent & Antisense', 'nogene_chr13_618858_619269'),
		]

	def _initialize_processors(self):
		"""Initialize all required data processors and analysis components."""
		# Initialize processors
		self.expression_processor = ExpressionAnalysisProcessor(self.output_dir)
		self.chromatin_metrics_processor = ChromatinMetricsProcessor(self.output_dir)
		self.integration = IntegratedChromatinExpressionAnalyzer(self.chromatin_metrics_processor,
												   self.expression_processor,
												   self.output_dir)	
		
		# Initialize genome deconvolution analysis
		self.genome_deconv_analysis = GenomeDeconvolutionAnalysis(
			self.output_dir,
			f'{self.output_dir}/chromatin_deconvolution/deconvolution_data/'
		)
		
		# Initialize chromatin configurations and plotters
		config1, config2 = load_default_chrom_configs()
		self.config1 = config1
		self.config2 = config2
		self.orf_plotter = load_default_orf_plotter()
		self.rna_plotter = RNASeqPileupPlotter(self.output_dir)
		self.rna_plotter.ylim = 13
		self.combined_model = CombinedChromatinModel(config1=config1, config2=config2)
	
	def load_data(self):
		"""Load all required data: expression, chromatin metrics, and gene annotations."""
		print("Loading expression data...")
		self.expression_processor.setup_data_loaders()
		self.loaded_deconvolved_expression = self.expression_processor.load_deconvolved_expression()
		self.raw_expression = self.expression_processor.load_raw_expression()

		# Setting a minimum threshold for transcript level
		mean_tx = self.expression_processor.expression_data.mean(1)	
		self.min_expression_level = np.quantile(mean_tx, q=0.5)
		print("Setting a minimum transcription level: The 50th percentile of the average gene expression: ", 
			self.min_expression_level)

		self.tpm_plotter = DeconvolvedTPMPlotter(self.expression_processor.expression_data,
								   self.expression_processor.all_transcripts_set)
		
		print("Loading chromatin metrics...")
		self.chromatin_metrics_processor.setup_data_loaders()
		self.chromatin_metrics_processor.load_and_assign_saved_metrics()
		
		print("Computing expression PTRs...")
		self.expression_processor.compute_expression_ptrs()
		
		print("Loading gene annotations...")
		self.genes = read_nondubious_genes_dataset()
		self.geneset = read_geneset_with_computed_regions()
		
		print("Filtering overlapping transcripts...")
		self.filter_overlapping_transcripts()
		
		print("Data loading complete.")
	
	def filter_overlapping_transcripts(self):
		"""
		Filter non-genic transcripts based on gene overlaps.
		
		Removes transcripts overlapping with genes on same strand.
		Labels transcripts overlapping on opposite strand as antisense.
		"""
		nongenic_set = self.expression_processor.all_transcripts_set
		nongenic_set = nongenic_set[nongenic_set.transcript_class == 'nongenic']
		
		keep_index = []
		filtered_index = []
		antisense_transcripts = {}
		
		for transcript_name, transcript_row in nongenic_set.iterrows():
			span = transcript_row.start, transcript_row.stop
			chrom = transcript_row.chr
			within_span = (span[1] >= self.geneset.left_end) & (span[0] <= self.geneset.right_end)
			same_strand = self.geneset.strand == transcript_row.strand
			
			overlapping_genes = self.geneset[(self.geneset.chr == chrom) & 
				within_span & 
				same_strand]
			antisense_genes = self.geneset[(self.geneset.chr == chrom) & 
				within_span & 
				~(same_strand)]
			
			# Keep track of transcripts that overlap
			if len(overlapping_genes) > 0:
				filtered_index.append(transcript_name)
			else:
				keep_index.append(transcript_name)
				
				# If transcript doesn't overlap same strand, check opposite strand
				if len(antisense_genes) > 0:
					antisense_transcripts[transcript_name] = antisense_genes.index.values
		
		# Create filtered dataset
		self.filtered_nongenic_transcripts = nongenic_set.loc[keep_index].copy()
		self.filtered_nongenic_transcripts['antisense_gene'] = np.nan
		
		# Add antisense gene annotations
		for transcript_name, genes_arr in antisense_transcripts.items():
			self.filtered_nongenic_transcripts.loc[transcript_name, 'antisense_gene'] = ','.join(genes_arr)

		# Add divergent transcript annotations
		self.filtered_nongenic_transcripts = self.identify_divergent_genes_to_nongenic_transcripts(self.filtered_nongenic_transcripts)
		
		self.antisense_transcripts = antisense_transcripts
		
		print(f"Filtered {len(nongenic_set)} nongenic transcripts overlapping with genes on the same strand")
		print(f"  Kept: {len(self.filtered_nongenic_transcripts)}")
		print(f"  Removed (same-strand overlap): {len(filtered_index)}")
		print(f"  Antisense transcripts: {len(antisense_transcripts)}")
		print(f"  Divergent transcripts: {self.filtered_nongenic_transcripts.divergent_orf.count()}")

	def identify_divergent_genes_to_nongenic_transcripts(self, filtered_nongenic_transcripts):
		from src.transcripts_dataset import load_transcripts_sets

		all_genes, non_genes = load_transcripts_sets('output/draft4_run/')
		# Let's characterize the antisense and divergent transcripts.

		# Antisense should be annotated already, let's check divergent transcripts. 
		# What is considered a shared promoter?
		# Criteria:
		#             given a transcript, there exists an upstream gene that is less than: 
		#             500 bp away

		nongenic_transcripts_w_divergent = filtered_nongenic_transcripts.copy()

		# if transcript is left to right (+), 
		# divergent is right to left (-)
		search_window = -600, 100

		# Loop through nongenic transcripts, annotate divergent genes if criteria matches
		for transcript_name, transcript in nongenic_transcripts_w_divergent.iterrows():
			strand = transcript.strand
			if strand == '+':
				TSS = transcript.promoter_start
				transcript_search_span = (TSS+search_window[0], 
										  TSS+search_window[1])
			else:
				TSS = transcript.promoter_end
				transcript_search_span = (TSS-search_window[1], 
										  TSS-search_window[0])

			found_genes = all_genes[(all_genes.chr == transcript.chr) & 
								   (all_genes.TSS > transcript_search_span[0]) & 
								   (all_genes.TSS < transcript_search_span[1]) &
								   (all_genes.strand != strand)]
			
			if len(found_genes) > 0:
				nongenic_transcripts_w_divergent.loc[transcript_name, 'divergent_orf'] = \
					found_genes.index[0]

		return nongenic_transcripts_w_divergent

	
	def apply_expression_filters(self, min_expression_level=None, min_ptr=None):
		"""
		Apply expression-based filters to the non-genic transcripts.
		
		Parameters:
		-----------
		min_expression_level : float, optional
			Minimum 90th percentile expression level (default: self.min_expression_level)
		min_ptr : float, optional
			Minimum PTR value (default: self.min_ptr)
			
		Returns:
		--------
		pd.DataFrame
			Filtered transcript data with expression and chromatin metrics
		"""
		if min_expression_level is None:
			min_expression_level = self.min_expression_level
		if min_ptr is None:
			min_ptr = self.min_ptr

		expression_level_filtered_ptr_data = self._retrieve_filtered_data(min_expression_level, min_ptr)
		self.expression_level_filtered_ptr_data = expression_level_filtered_ptr_data

		print(f"Filtered {len(self.filtered_nongenic_transcripts)} nongenic transcripts with criteria:")
		print(f"Minimum expression level: {self.min_expression_level}")
		print(f"Minimum PTR: {self.min_ptr}")
		print(f"  Kept: {len(self.expression_level_filtered_ptr_data)}")
		print(f"  Removed: {(len(self.filtered_nongenic_transcripts))-len(expression_level_filtered_ptr_data)}")
		print(f"  Antisense transcripts: {self.expression_level_filtered_ptr_data.antisense_gene.count()}")
		print(f"  Divergent transcripts: {self.expression_level_filtered_ptr_data.divergent_orf.count()}")

		both = self.expression_level_filtered_ptr_data[~(self.expression_level_filtered_ptr_data.antisense_gene.isna())
											& ~(self.expression_level_filtered_ptr_data.divergent_orf.isna())]
		print(f"  Antisense & Divergent: {len(both)}")

		return expression_level_filtered_ptr_data
	
	def _retrieve_filtered_data(self, min_expression_level, min_ptr):
		"""Internal method to retrieve and filter transcript data."""
		plot_data = self.expression_processor.all_transcripts_ptrs
		
		# Calculate 90th percentile expression
		high_quantile_expression = self.expression_processor.expression_data[
			self.config1.t_indices()].quantile(axis=1, q=0.9)
		high_quantile_expression = pd.DataFrame(high_quantile_expression)
		
		# Join data
		plot_data = plot_data.join(high_quantile_expression).join(
			self.filtered_nongenic_transcripts[['antisense_gene', 'divergent_orf']], how='right')
		plot_data.columns = ['expression_ptr', 'transcript_class', 
							 'percentile_90', 'antisense_gene', 'divergent_orf']
		
		# Apply filters
		plot_data = plot_data[(plot_data.percentile_90 > min_expression_level) & 
							  (plot_data.expression_ptr > min_ptr)]
		
		# Join with chromatin metrics
		plot_data = self._join_with_chromatin(plot_data, 'promoter_occupancy')
		plot_data = self._join_with_chromatin(plot_data, 'nucleosome_entropy')
		plot_data = self._join_with_chromatin(plot_data, 'nucleosome_occupancy')
		
		return plot_data
	
	def _join_with_chromatin(self, plot_data, chromatin_key):
		"""Join plot data with chromatin metrics."""
		chromatin_data = self.chromatin_metrics_processor.deconvolved_chromatin_ptrs[chromatin_key]
		return plot_data.join(chromatin_data, how='left')
	
	def create_scatter_analysis(self, qthreshold=None):
		"""
		Create the main 3-panel scatter plot analysis.
		
		Parameters:
		-----------
		qthreshold : float, optional
			Quantile threshold for highlighting high-PTR transcripts (default: self.qthreshold)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The generated figure object
		"""
		if qthreshold is None:
			qthreshold = self.qthreshold
			
		# Get filtered data
		joined_data = self.apply_expression_filters()
		
		# Create figure
		fig = plt.figure(figsize=(4, 10))
		
		# Plot configurations
		plot_configs = [
			{
				'x_column': 'promoter_occupancy_ptr',
				'title': 'Promoter occupancy',
				'xlabel': 'Promoter occupancy PTR',
				'colormap': plt.cm.Oranges,
				'subplot_pos': 1,
				'xlim': (0.99, 1.5),
			},
			{
				'x_column': 'nucleosome_entropy_ptr', 
				'title': 'Nucleosome entropy',
				'xlabel': 'Entropy PTR',
				'colormap': plt.cm.Purples,
				'subplot_pos': 2,
				'xlim': (0.99, 1.36),
			},
			{
				'x_column': 'nucleosome_occupancy_ptr',
				'title': 'Nucleosome occupancy', 
				'xlabel': 'Nucleosome occupancy PTR',
				'colormap': plt.cm.Blues,
				'subplot_pos': 3,
				'xlim': (0.99, 1.7),
			}
		]
		
		# Calculate threshold info
		tx_ptr_threshold = np.quantile(self.expression_processor.all_transcripts_ptrs.ptr, q=0.95)
		self.tx_ptr_threshold = tx_ptr_threshold

		num_thresh = len(joined_data[joined_data.expression_ptr >= tx_ptr_threshold])
		print(f"{num_thresh} transcripts with >95 percentile PTR (all transcription)")

		# Create all subplots
		for i, config in enumerate(plot_configs):
			self._create_scatter_subplot(joined_data, tx_ptr_threshold=tx_ptr_threshold, **config)
		
		# Final formatting
		plt.suptitle(f"Non-genic transcripts,\n"
					 f"n={len(joined_data)} ({num_thresh} cycling)", 
					 fontweight='demi', fontsize=18)
		plt.tight_layout()
		plt.subplots_adjust(hspace=0.5)

		save_figure_for_paper(f"{self.save_dir}/nongenic_ptrs.png")
		
		return fig
	
	def _create_scatter_subplot(self, joined_data, x_column, title, xlabel, xlim, colormap, 
			subplot_pos, tx_ptr_threshold):
		"""Create individual scatter plot subplot."""

		nrows, ncols = 3, 1
		plt.subplot(nrows, ncols, subplot_pos)


		# Examine how many genes will be collected when a threshold is applied
		# to the chromatin as well.

		chromatin_values = joined_data[x_column].values
		#chromatin_ptr_threshold = np.quantile(chromatin_values, q=tx_ptr_threshold)

		# print("Value threshold for chromatin measure:", x_column, chromatin_ptr_threshold)

		self.plot_category_ptr_scatter_data(joined_data, x_column)

		plt.axhline(tx_ptr_threshold, c='#232323',  ls='dotted', lw=1, alpha=1.0)

		for label, category, transcript_name in self.selected_transcripts:
			if transcript_name in joined_data.index:
				x, y = joined_data.loc[transcript_name][x_column], \
					joined_data.loc[transcript_name]['expression_ptr']
				
				if x > xlim[1]:
					x = xlim[1]-0.01
					marker = '>'
					facecolor = 'black'
					lw = 1
				else:
					marker = 'D'
					facecolor = 'none'
					lw = 0

				plt.scatter(x, y, marker=marker, s=20, lw=lw, edgecolor='black', 
							facecolor=facecolor)
				plt.text(x, y+0.28, label, fontsize=12, ha='center',
					path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
		
		# Formatting
		plt.legend(ncol=1)
		plt.ylim(0.85, 7)
		plt.xlim(*xlim)
		plt.title(title, fontsize=14)
		plt.xlabel(xlabel)
		plt.ylabel("Expression PTR")


	def plot_category_ptr_scatter_data(self, joined_data, x_column):

		antisense_select = ~joined_data.antisense_gene.isna()
		divergent_select = ~joined_data.divergent_orf.isna()
		both_select = antisense_select & divergent_select
		neither_select = ~antisense_select & ~divergent_select & ~both_select

		neither_transcripts = joined_data[neither_select]
		both_transcripts = joined_data[both_select]
		divergent_only_transcripts = joined_data[divergent_select & ~both_select]
		antisense_only_transcripts = joined_data[antisense_select & ~both_select]

		def _plot_scatter(data, marker, color, label):
			num_threshold = len(data[data.expression_ptr > self.tx_ptr_threshold])
			label = f"{label}, n={len(data)} ({num_threshold})"
			plt.scatter(data[x_column],
					   data.expression_ptr,
					   marker=marker, edgecolor=color, lw=1, facecolor='none',
					   alpha=0.5,
					   s=20, label=label)

		_plot_scatter(divergent_only_transcripts, marker='o', color='blue', label="Divergent only")
		_plot_scatter(antisense_only_transcripts, marker='o', color='red', label="Antisense only")
		_plot_scatter(both_transcripts, marker='o', color='purple', label="Both")
		_plot_scatter(neither_transcripts, marker='o', color='#aaa', label="Neither")

		plt.legend()
	
	def plot_transcript_locus(self, transcript_name, title=None, span_size=None):
		"""
		Plot the chromatin and expression data for a specific transcript locus.
		
		Parameters:
		-----------
		transcript_name : str
			Name of the transcript to plot
		span_size : int, optional
			Size of region to plot on each side of transcript center (default: self.locus_span_size)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The generated figure object
		"""
		if span_size is None:
			span_size = self.locus_span_size
			
		# Parse transcript name to get coordinates
		chrom, span = parse_transcript_name(transcript_name)
		transcript_row = self.filtered_nongenic_transcripts.loc[transcript_name]

		if transcript_row.strand == '+':
			center = span[0]
		else:
			center = span[1]
		
		# Define span
		plotting_span = center - span_size, center + span_size

		# Round the plotting span to the nearest 100
		from src.utils import nearest_span
		plotting_span = nearest_span(plotting_span, 100)
		
		# Load and plot data
		loaded_data = self.genome_deconv_analysis.load_mnase_span(chrom, plotting_span)
		plotter = self.genome_deconv_analysis.plot_loaded_data(
			figsize=(7, 11), 
			title=title,
			highlight_bins=[],
			rna_plotter=self.rna_plotter,
			tpm_plotter=self.tpm_plotter,
			plot_index_labels=False
		)

		for ax in [plotter.annotation_axis, plotter.rna_pileup_axis]:
			ax.axvspan(span[0], span[1], -20, 20, color='#ddd', zorder=0, alpha=0.2)

		for ax in plotter.chromatin_axes:
			ax.axvline(span[0], color='#777', ls='dotted', zorder=0, alpha=0.5)
			ax.axvline(span[1], color='#777',  ls='dotted', zorder=0, alpha=0.5)

		return plotter
	
	def plot_transcript_calling(self, transcript_name, wider_span_size=None):
		"""
		Visualize the transcript boundary calling process for a specific transcript.
		
		Parameters:
		-----------
		transcript_name : str
			Name of the transcript to analyze
		wider_span_size : int, optional
			Size of wider region for transcript calling (default: self.wider_span_size)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The generated figure object
		"""
		if wider_span_size is None:
			wider_span_size = self.wider_span_size
			
		# Parse transcript name
		transcript_split = transcript_name.split('_')
		chrom = int(transcript_split[1].replace('chr', ''))
		center = int(transcript_split[2])
		
		# Define wider span
		wider_span = center - wider_span_size, center + wider_span_size
		
		# Initialize transcript boundary caller
		caller = TranscriptBoundaryCaller(output_directory=self.output_dir)
		caller.set_chrom_span(chrom, wider_span)
		caller.call_boundaries_both_strands()
		
		# Create the plot
		fig = plot_tx_transcript_context(
			self.output_dir, chrom, wider_span, 
			self.orf_plotter, self.rna_plotter, 
			self.combined_model, caller, 
			figsize=(9, 5)
		)
		
		return fig
	
	def get_summary_stats(self):
		"""
		Get summary statistics about the filtered transcripts.
		
		Returns:
		--------
		dict
			Dictionary containing various statistics
		"""
		filtered_data = self.apply_expression_filters()
		
		stats = {
			'total_nongenic_transcripts': len(self.filtered_nongenic_transcripts),
			'high_expression_transcripts': len(filtered_data),
			'antisense_transcripts': len(self.antisense_transcripts),
			'antisense_high_expression': len(filtered_data[~filtered_data.antisense_gene.isna()]),
			'non_antisense_high_expression': len(filtered_data[filtered_data.antisense_gene.isna()]),
			'config': {
				'min_expression_level': self.min_expression_level,
				'min_ptr': self.min_ptr,
				'qthreshold': self.qthreshold,
				'locus_span_size': self.locus_span_size,
				'wider_span_size': self.wider_span_size
			}
		}
		
		return stats

	def plot_selected_phase_space_plots(self):
		"""Plot the phase space plots to see how the non-genic transcripts look in this visualization.

		It's possible that the deconvolved transcripts could use a higher kappa or gamma value. As there
		is an unusual sinusoidal wave on the daughter-branch.

		Keep these plots available, in case we want to include them in the figure panel
		"""

		for label, cat, transcript_name in self.selected_transcripts:
			title = (f"{label} - {cat}")
			self.integration.plot_all_metrics_all_replicates_gene(transcript_name,
				title=title)
			save_figure_for_paper(f"{self.save_dir}/phase_space_{label}_{transcript_name}.png")

	
	def print_summary(self):
		"""Print a formatted summary of the analysis."""
		stats = self.get_summary_stats()
		
		print("=== Non-genic Transcript Analysis Summary ===")
		print(f"Total non-genic transcripts (after filtering): {stats['total_nongenic_transcripts']}")
		print(f"High expression transcripts: {stats['high_expression_transcripts']}")
		print(f"  - Antisense: {stats['antisense_high_expression']}")
		print(f"  - Non-antisense: {stats['non_antisense_high_expression']}")
		print(f"Total antisense transcripts (all expression levels): {stats['antisense_transcripts']}")
		print()
		print("Configuration:")
		for key, value in stats['config'].items():
			print(f"  {key}: {value}")


	def create_plots(self):
		from src.figure_configs import save_figure_for_paper

		# Create main analysis plot
		fig = self.create_scatter_analysis()

		for label, category, transcript_name in self.selected_transcripts:
			chrom, span = parse_transcript_name(transcript_name)
			title = f"{label}: {category}"
			fig2 = self.plot_transcript_locus(transcript_name, title=title)
			save_figure_for_paper(f"{self.save_dir}/locus_{label}_{transcript_name}.png")

		self.plot_selected_phase_space_plots()

	def layout_panel(self):
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 640, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/nongenic_ptrs.png',
			f'{self.save_dir}/locus_1_nogene_chr4_443744_444111.png',
			f'{self.save_dir}/locus_2_nogene_chr13_618858_619269.png',
		]

		# Layout images horizontally with custom width proportions
		# Adjust these proportions based on your image content needs
		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			width_proportions=[0.78, 1.12, 1.12],
			between_padding=30,
			margin=(40, 40),
			image_keys=['nongenic_ptrs', 'locus_1', 'locus_2']  # Custom keys
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=26,
			offset=(-15, -15)  # Adjust offset as needed
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental6_Nongenic_transcription.png')

def parse_transcript_name(transcript_name):
	transcript_split = transcript_name.split('_')
	chrom = int(transcript_split[1].replace('chr', ''))
	start = int(transcript_split[2])
	end = int(transcript_split[3])
	return chrom, (start, end)
