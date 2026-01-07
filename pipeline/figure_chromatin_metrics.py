import os

import matplotlib.pyplot as plt
from pipeline.chromatin_metrics_processor import ChromatinMetricsProcessor
from pipeline.transcription_processor import ExpressionAnalysisProcessor

from src.geneset import cyclin_genes
from pipeline.figure_composer import FigureCompositor
from pipeline.figure_composer_helpers import layout_images_vertically, add_panel_labels_to_images
from src.figure_configs import save_figure_for_paper


class FigureChromatinMetrics:
	"""
	A class to generate and layout chromatin metrics figures.
	
	This class handles the creation of individual chromatin metrics plots
	and arranges them in a single vertical panel layout.
	"""
	
	def __init__(self, output_dir):
		"""
		Initialize the figure generator.
		
		Parameters:
		-----------
		output_dir : str
			The output directory where figures will be saved
		"""
		self.output_dir = output_dir
		self.figures_dir = os.path.join(output_dir, 'chromatin_metrics', 'figures')
		self.panel_figures_dir = os.path.join(output_dir, 'Figures')
		self.chromatin_processor = None
		
		# Ensure figures directory exists
		os.makedirs(self.figures_dir, exist_ok=True)
	
	def setup_processor(self):
		"""
		Set up the ChromatinMetricsProcessor with data loading.
		"""

		self.expression_processor = ExpressionAnalysisProcessor(self.output_dir)
		self.tx_summary = self.expression_processor.run_full_analysis()

		self.chromatin_processor = ChromatinMetricsProcessor(self.output_dir)
		self.chromatin_processor.setup_data_loaders()
		self.chromatin_processor.load_and_assign_saved_metrics()
		
		# Set selected genes to cyclin genes
		self.chromatin_processor.selected_genes = ['CLB5', 'CLB1', 'CLB4', 'CLN1']

		# Setup TPM plotter for locus plots
		self.load_deconvolved_expression_plotter()

	def setup_integration(self):
		from pipeline.expression_chromatin_integration import IntegratedChromatinExpressionAnalyzer

		self.integration = IntegratedChromatinExpressionAnalyzer(self.chromatin_processor,
									 self.expression_processor,
									 self.output_dir)

		# Select coordinated genes by thresholding each measure to the top 5%
		self.integration.apply_threshold()		
		self.integration.compute_and_assign_linkages_data()

		self.integration.apply_threshold('raw_rep1')
		self.integration.apply_threshold('raw_rep2')
		self.integration.apply_threshold()

	def create_integration_plots(self):

		plot_datasets = ['deconvolved']#['raw_rep1', 'raw_rep2', 'deconvolved']
		for dataset in plot_datasets:
			fig = self.integration.plot_ptr_correlations(dataset)
			save_figure_for_paper(f"{self.figures_dir}/expression_chromatin_{dataset}_scatter.png")

		# # Plot cell cycle genes, with trajectory area values
		# self.integration.plot_all_trajectory_area_cell_cycle_genes()
		# save_figure_for_paper(f"{self.figures_dir}/cell_cycle_trajectory_values.png")

		# self.create_top_bottom_metric_trajectory_plots()
		# self.create_raw_deconvolved_trajectory_plots()

		# # Plot trajectories for each gene group
		# self.create_gene_group_trajectories_plots()

	def create_top_bottom_metric_trajectory_plots(self):

		metrics = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		# Metric top and bottom exampels
		for metric in metrics:
			self.integration.plot_chromatin_example_trajectories(metric)
			save_figure_for_paper(f"{self.figures_dir}/top_bottom_trajectories_{metric}.png")

	def create_raw_deconvolved_trajectory_plots(self):

		# Plot the raw vs deconvolved trajectory plots
		# for sample genes
		genes_to_plot = ['MCM7', 'CLB1', 'HHT1', 'HTA1']
		for gene in genes_to_plot:
			fig = self.integration.plot_all_metrics_all_replicates_gene(gene)
			save_figure_for_paper(f"{self.figures_dir}/trajectories_{gene}.png")


	def create_gene_group_trajectories_plots(self):

		from src.geneset import cyclin_genes, mcm_genes, histone_genes

		groups_to_plot = [
			histone_genes('H2A'),
			histone_genes('H2B'),
			histone_genes('H3'),
			histone_genes('H4'),
			mcm_genes(), 
			cyclin_genes('G1'), 
			cyclin_genes('B-S'), 
			cyclin_genes('B-G2'),
			cyclin_genes('B-M')
		]

		group_titles = [
			'Histones H2A', 
			'Histones H2B', 
			'Histones H3', 
			'Histones H4', 
			'MCM1 and MCM2-7 complex',
			'Early, G1-type cyclins',
			'S-phase, B-type cyclins',
			'G2-phase, B-type cyclins',
			'M-phase, B-type cyclins',
		]

		# Precompute limits for ALL histone genes (H2A + H2B + H3 + H4)
		all_histone_genes = (histone_genes('H2A') + histone_genes('H2B') + 
		                     histone_genes('H3') + histone_genes('H4'))

		histone_xlims, histone_ylim = self.integration.compute_trajectory_limits(
			all_histone_genes)

		# Precompute limits for ALL histone genes (H2A + H2B + H3 + H4)
		all_cyclin_genes = (cyclin_genes('G1')+cyclin_genes('B-S')+cyclin_genes('B-G2')+
			cyclin_genes('B-M'))
		cyclin_xlims, cyclin_ylim = self.integration.compute_trajectory_limits(
			all_cyclin_genes)

		for i, gene_group in enumerate(groups_to_plot):

			title = group_titles[i]

			if 'MCM1' in title:
				title = '$\\it{MCM1}$ and $\\it{MCM}2$-$7$ complex'

			if "Histone" in title:
				fig = self.integration.plot_gene_group_trajectories_all_metrics(gene_group,
					title=title, override_xlim=histone_xlims, override_ylim=histone_ylim)

			elif 'cyclins' in title:
				fig = self.integration.plot_gene_group_trajectories_all_metrics(gene_group,
					title=title, override_xlim=cyclin_xlims, override_ylim=cyclin_ylim)

			else:
				fig = self.integration.plot_gene_group_trajectories_all_metrics(gene_group,
					title=title, auto_lims=True)

			save_path = f"{self.figures_dir}/trajectories_group_{group_titles[i].replace(' ', '_')}.png"
			print('Wrote to: ', save_path)
			save_figure_for_paper(save_path)
			plt.close(fig)


	def create_locus_plots(self):

		from src.sgd import get_gene_title_name
		from src.transcripts_dataset import load_transcripts_sets
		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis

		genes, _ = load_transcripts_sets(output_dir='output/draft4_run/')

		genome_deconv_analysis = GenomeDeconvolutionAnalysis(
			'output/draft4_run/')

		genes_to_plot_map = {
			'CLB1': {
				'title': get_gene_title_name('CLB1', include_system=False),
				'save_name': 'locus_CLB1',
				'span_offset': (-1000, 2000),
				'figsize': (9, 11)
			},
			'HTA1': {
				'title': f"{get_gene_title_name('HTB1', include_system=False)} and {get_gene_title_name('HTA1', include_system=False)}",
				'span_offset': (-1600, 1000),
				'save_name': 'locus_HTA1_HTB1',
				'figsize': (7, 11)
			},
			'MCM1': {
				'title': f"{get_gene_title_name('MCM1', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM1',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'MCM2': {
				'title': f"{get_gene_title_name('MCM2', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM2',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'MCM3': {
				'title': f"{get_gene_title_name('MCM3', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM3',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'MCM4': {
				'title': f"{get_gene_title_name('MCM4', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM4',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'MCM5': {
				'title': f"{get_gene_title_name('MCM5', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM5',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'MCM6': {
				'title': f"{get_gene_title_name('MCM6', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM6',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'MCM7': {
				'title': f"{get_gene_title_name('MCM7', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM7',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'YOX1': {
				'title': f"{get_gene_title_name('YOX1', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_YOX1',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
			'YHP1': {
				'title': f"{get_gene_title_name('YHP1', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_YHP1',
				'figsize': (7, 11),
				'tf_binding': ['Mcm1']
			},
		}

		for gene_key, gene_dict in genes_to_plot_map.items():
			gene = genes[genes['gene'] == gene_key].iloc[0]
			span = gene.TSS+gene_dict['span_offset'][0], gene.TSS+gene_dict['span_offset'][1]
			loaded_data = genome_deconv_analysis.load_mnase_span(gene.chr, span)

			tfs = []
			if 'tf_binding' in gene_dict:
				tfs = gene_dict['tf_binding']

			genome_deconv_analysis.plot_loaded_data(figsize=gene_dict['figsize'], 
				title=gene_dict['title'],
				plot_index_labels=False, tfs=tfs,
				tpm_plotter=self.tpm_plotter)
			save_figure_for_paper(f'{self.figures_dir}/{gene_dict["save_name"]}.png')


	def load_deconvolved_expression_plotter(self):
		"""Expression plotter requires the loading of the deconvolved TPM data"""

		from pipeline.transcription_processor import ExpressionAnalysisProcessor
		from src.deconvolved_tpm_plotter import DeconvolvedTPMPlotter

		self.tpm_plotter = DeconvolvedTPMPlotter(self.expression_processor.expression_data,
								   self.expression_processor.all_transcripts_set)

	def create_metrics_plots(self):
		"""
		Create the individual chromatin metrics plots.
		
		This method replicates the functionality from the notebook:
		- Sets up the processor
		- Loads the data
		- Creates and saves the plots
		"""
		if self.chromatin_processor is None:
			self.setup_processor()
		
		# Create the plots (this will save them to the figures directory)
		self.chromatin_processor.create_plots()

		# Save the chromatin ptr plots with cell cycling expression labeling
		self.chromatin_processor.plot_combined_ptr_change_w_expression(
			self.expression_processor.top_cycling_genes.index)
		save_figure_for_paper(f"{self.figures_dir}/combined_ptrs_w_top_expression.png")

		# Plot expression PTRs histogram
		fig, ax, threshold = self.expression_processor.plot_ptr_histogram()
		save_figure_for_paper(f"{self.figures_dir}/expression_ptrs.png")

		print(f"Individual plots saved to: {self.figures_dir}")
	
	def layout_supplemental_panel(self):
		"""
		Create a vertical layout panel with all three chromatin metrics figures.
		
		This arranges the three generated plots:
		- promoter_occupancy_ptr.png
		- nucleosome_entropy_ptr.png  
		- nucleosome_occupancy_ptr.png
		
		In a single vertical panel with appropriate labels.
		"""
		# Create compositor
		compositor = FigureCompositor(1024, 1040, debug_mode=True)
		
		# Define image paths for the three chromatin metrics plots
		image_paths = [
			f'{self.figures_dir}/promoter_occupancy_ptr.png',
			f'{self.figures_dir}/nucleosome_entropy_ptr.png',
			f'{self.figures_dir}/nucleosome_occupancy_ptr.png',
		]
		
		# Verify all images exist
		missing_images = [path for path in image_paths if not os.path.exists(path)]
		if missing_images:
			raise FileNotFoundError(f"Missing image files: {missing_images}")
		
		# Layout images vertically with equal proportions
		placed_images = layout_images_vertically(
			compositor,
			image_paths,
			height_proportions=[0.33, 0.33, 0.34],  # Equal proportions for three plots
			between_padding=20,
			margin=(100, 30),
			image_keys=['PromoterOccupancy', 'NucleosomeEntropy', 'NucleosomeOccupancy']
		)
		
		# Add panel labels (A, B, C)
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=36,
			offset=(-10, 30)
		)
		
		# Save the composed figure
		output_path = f'{self.panel_figures_dir}/Supplemental6_Chromatin_Metrics.png'
		compositor.save(output_path)
		
		print(f"Panel layout saved to: {output_path}")
		
		return output_path
	
	def generate_figures_and_panel(self):
		"""
		Complete workflow: create individual plots and layout panel.
		
		Returns:
		--------
		str
			Path to the final composed figure
		"""
		print("Creating individual chromatin metrics plots...")
		self.create_metrics_plots()
		self.create_integration_plots()
		self.create_locus_plots()
		print("Complete figure generation finished!")

		self.layout_panels()

	def layout_panels():
		print("Creating panel layout...")
		self.layout_trajectories_panel()
		# self.layout_chromatin_transcription_ptrs()
		self.layout_genesets_panel()
		self.layout_mcm_panel()
		self.layout_supplemental_panel()

	def layout_trajectories_panel(self, margin=(30, 30), between_padding=20, 
							  panel_padding=40, add_labels=True, font_size=36):
		"""Layout the figure panel for """
		import os

		from pipeline.figure_composer_helpers import layout_images_vertically

		compositor = FigureCompositor(1024, 510, debug_mode=True)
		image_dir = self.figures_dir
		panel_save_path = os.path.join(self.panel_figures_dir, 'Figure5_Chromatin_Metrics.png')
		
		# Define image paths
		image_paths = {
			'traj_diagrams': './diagrams/Transcription_Chromatin_Diagrams.png',
			'clb1_traj': os.path.join(image_dir, 'trajectories_CLB1.png'),
			'mcm7_traj': os.path.join(image_dir, 'trajectories_MCM7.png')
		}
		
		# Verify all images exist
		missing_images = [path for path in image_paths.values() if not os.path.exists(path)]
		if missing_images:
			raise FileNotFoundError(f"Missing image files: {missing_images}")
		
		all_placed_images = {}

		# Calculate proportions - gene examples on left, diagram on right
		left_prop = 0.305  # Gene examples get ~31% of width
		left_width = (compositor.logical_width - panel_padding - (2 * margin[0])) * left_prop

		# 1. Place two gene example images vertically on the left
		vertical_paths = [image_paths['clb1_traj'], image_paths['mcm7_traj']]
		vertical_keys = ['CLB1Trajectories', 'MCM7Trajectories']

		vertical_images = layout_images_vertically(
			compositor,
			vertical_paths,
			x_position=margin[0],
			between_padding=between_padding,
			margin=(margin[0], margin[1]),
			widths=[left_width, left_width],
			image_keys=vertical_keys
		)
		all_placed_images.update(vertical_images)
		
		# 2. Place the diagram on the right side
		right_col_x = margin[0] + left_width + panel_padding-5
		right_width = (compositor.logical_width - panel_padding - (2 * margin[0]) - left_width)

		right_img = compositor.place_image(
			image_paths['traj_diagrams'],
			x=right_col_x,
			y=margin[1]-10,
			width=right_width,
			name='TrajDiagrams'
		)
		all_placed_images['TrajDiagrams'] = right_img
		
		# 3. Add panel labels if requested
		if add_labels:
			# Order images for labeling: gene examples first, then diagram
			ordered_keys = ['CLB1Trajectories', 'MCM7Trajectories', 'TrajDiagrams']
			labels = ['a', 'b', 'c']

			for key, label in zip(ordered_keys, labels):

				if label == 'c':
					y_offset = 28
				else:
					y_offset = 20

				if key in all_placed_images:
					compositor.add_panel_label_to_image(
						key,
						label,
						offset=(-10, y_offset),
						font_size=font_size,
						font_type='bold'
					)

		# Add the second diagram label (as in original)
		compositor.add_panel_label_to_image(
			'TrajDiagrams',
			'd',
			offset=(-10, 268),
			font_size=font_size,
			font_type='bold'
		)
		
		# Save the composed figure
		compositor.save(panel_save_path)
		print(f"Panel layout saved to: {panel_save_path}")
		
		return panel_save_path

	# todo: deprecated for 4d analysis with clusters.
	#
	# def layout_chromatin_transcription_ptrs(self, margin=(30, 30), between_padding=20,
	# 						 left_width_percent=41, add_labels=True, font_size=24):
	# 	"""
	# 	Layout metrics panel 2 with the following arrangement:
	# 	- Left column (35% width): ptrs_vs_ptr_trajectory.png, cell_cycle_trajectory_values.png
	# 	- Right column (65% width): top_bottom_trajectories_promoter_occupancy.png,
	# 								top_bottom_trajectories_nucleosome_entropy.png,
	# 								top_bottom_trajectories_nucleosome_occupancy.png
	# 	"""
	# 	import os
	# 	from pipeline.figure_composer_helpers import layout_images_vertically, layout_images_horizontally,\
	# 		place_image_below

	# 	image_dir = self.figures_dir
	# 	panel_save_path = os.path.join(self.panel_figures_dir, 'Supplemental6_Chromatin_Transcription.png')

	# 	compositor = FigureCompositor(1024, 980, debug_mode=True)
		
	# 	# Define image paths
	# 	image_paths = {
	# 		'raw_vs_ptr':  os.path.join(image_dir, 'raw_vs_deconvolved_all_metrics_ptrs.png'),
	# 		'ptrs_vs_ptr': os.path.join(image_dir, 'ptrs_vs_ptr_trajectory.png'),
	# 		'cell_cycle_values': os.path.join(image_dir, 'cell_cycle_trajectory_values.png'),
	# 		'promoter_occupancy': os.path.join(image_dir, 'top_bottom_trajectories_promoter_occupancy.png'),
	# 		'nucleosome_entropy': os.path.join(image_dir, 'top_bottom_trajectories_nucleosome_entropy.png'),
	# 		'nucleosome_occupancy': os.path.join(image_dir, 'top_bottom_trajectories_nucleosome_occupancy.png')
	# 	}

	# 	placed_images = layout_images_horizontally(
	# 		compositor,
	# 		[image_paths['raw_vs_ptr'], image_paths['ptrs_vs_ptr']],
	# 		width_proportions=[1., 1.],
	# 		between_padding=20,
	# 		margin=30,
	# 		offsets=[(0, 0), (0, -5)],
	# 		image_keys=['raw_vs_ptr', 'ptrs_vs_ptr']
	# 	)

	# 	place_image_below(compositor, image_paths['cell_cycle_values'], 'raw_vs_ptr',
	# 		vertical_padding=40, width=230, new_key='cell_cycle_trajectories')

	# 	raw_vs_ptr_img = placed_images['raw_vs_ptr']
	# 	y_offset = raw_vs_ptr_img['logical_position'][1]+raw_vs_ptr_img['logical_size'][1]+40
	# 	trajectories_width = 730

	# 	placed_images = layout_images_vertically(
	# 		compositor,
	# 		[image_paths['promoter_occupancy'], image_paths['nucleosome_entropy'],
	# 		 image_paths['nucleosome_occupancy']],
	# 		 widths=[trajectories_width, trajectories_width, trajectories_width],
	# 		 between_padding=30,
	# 		 margin=(0, y_offset),
	# 		 x_position=270,
	# 		 image_keys=['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
	# 	)

	# 	add_panel_labels_to_images(
	# 		compositor, 
	# 		compositor.placed_images,
	# 		font_size=36,
	# 		offset=(-16, 10)
	# 	)

	# 	# Save the composed figure
	# 	compositor.save(panel_save_path)
	# 	print(f"Panel layout saved to: {panel_save_path}")
		
	# 	return panel_save_path

	def layout_genesets_panel(self, margin=(30, 30), between_padding=30, 
			traj_vertical_padding=15, panel_padding=30, add_labels=True, font_size=28):
		"""Layout the figure panel for genesets analysis"""
		import os
		
		from pipeline.figure_composer_helpers import layout_images_vertically

		compositor = FigureCompositor(1024, 520, debug_mode=True)
		image_dir = self.figures_dir
		panel_save_path = os.path.join(self.panel_figures_dir, 'Figure6_Genesets.png')
		
		# Define image paths
		image_paths = {
			'locus_clb1': os.path.join(image_dir, 'locus_CLB1.png'),
			'locus_hta1_htb1': os.path.join(image_dir, 'locus_HTA1_HTB1.png'),

			# Cyclins
			'traj_g1': os.path.join(image_dir, 'trajectories_group_Early,_G1-type_cyclins.png'),
			'traj_s': os.path.join(image_dir, 'trajectories_group_S-phase,_B-type_cyclins.png'),
			'traj_g2': os.path.join(image_dir, 'trajectories_group_G2-phase,_B-type_cyclins.png'),
			'traj_m': os.path.join(image_dir, 'trajectories_group_M-phase,_B-type_cyclins.png'),

			# Histones
			'traj_h2a': os.path.join(image_dir, 'trajectories_group_Histones_H2A.png'),
			'traj_h2b': os.path.join(image_dir, 'trajectories_group_Histones_H2B.png'),
			'traj_h3': os.path.join(image_dir, 'trajectories_group_Histones_H3.png'),
			'traj_h4': os.path.join(image_dir, 'trajectories_group_Histones_H4.png')
		}
		
		# Verify all images exist
		missing_images = [path for path in image_paths.values() if not os.path.exists(path)]
		if missing_images:
			raise FileNotFoundError(f"Missing image files: {missing_images}")
		
		all_placed_images = {}
		
		# Calculate fixed height for locus plots
		locus_height = compositor.logical_height - (2 * margin[1])
		
		# STEP 1: Place locus plots from right to left
		
		# Place CLB1 locus (rightmost)
		clb1_img = compositor.place_image(
			image_paths['locus_clb1'],
			x=compositor.logical_width - margin[0],  # Start from right edge
			y=margin[1],
			height=locus_height,
			name='LocusCLB1',
			anchor='top_right'  # Anchor to right edge
		)
		all_placed_images['LocusCLB1'] = clb1_img
		clb1_width = clb1_img['logical_size'][0]
		
		# STEP 2: Calculate widths for the middle sections
		# Now we have: Histones | HTA1_HTB1 | Cyclins | CLB1

		# Divide remaining space: histones column, HTA1_HTB1, cyclins column
		section_padding = 30
		trajectory_column_width = 135
		
		# STEP 3: Place sections from left to right

		# Place histones column (leftmost - A)
		histone_paths = [image_paths['traj_h2a'], image_paths['traj_h2b'], 
						 image_paths['traj_h3'], image_paths['traj_h4']]
		histone_keys = ['TrajH2A', 'TrajH2B', 'TrajH3', 'TrajH4']

		histones_placed = layout_images_vertically(
			compositor=compositor,
			image_paths_arr=histone_paths,
			between_padding=traj_vertical_padding,
			margin=(0, margin[1]),
			x_position=margin[0],
			image_keys=histone_keys,
			widths=[trajectory_column_width] * len(histone_paths),
			preserve_aspect_ratio=True
		)
		all_placed_images.update(histones_placed)

		# Place HTA1_HTB1 locus (second from left - B)
		hta1_htb1_x = margin[0] + trajectory_column_width + section_padding
		hta1_htb1_img = compositor.place_image(
			image_paths['locus_hta1_htb1'],
			x=hta1_htb1_x,
			y=margin[1],
			height=locus_height,
			name='LocusHTA1HTB1',
			anchor='top_left'
		)
		hta1_htb1_width = hta1_htb1_img['logical_size'][0]
		all_placed_images['LocusHTA1HTB1'] = hta1_htb1_img

		# Place cyclins column (third from left - C)
		cyclins_x = 490
		cyclin_paths = [image_paths['traj_g1'], image_paths['traj_s'], 
						image_paths['traj_g2'], image_paths['traj_m']]
		cyclin_keys = ['TrajG1', 'TrajS', 'TrajG2', 'TrajM']

		cyclins_placed = layout_images_vertically(
			compositor=compositor,
			image_paths_arr=cyclin_paths,
			between_padding=traj_vertical_padding,
			margin=(0, margin[1]),
			x_position=cyclins_x,
			image_keys=cyclin_keys,
			widths=[trajectory_column_width] * len(cyclin_paths),
			preserve_aspect_ratio=True
		)
		all_placed_images.update(cyclins_placed)

		# Place CLB1 locus (rightmost - D)
		clb1_x = compositor.logical_width - margin[0] - clb1_width
		clb1_img = compositor.place_image(
			image_paths['locus_clb1'],
			x=clb1_x,
			y=margin[1],
			height=locus_height,
			name='LocusCLB1',
			anchor='top_left'
		)
		all_placed_images['LocusCLB1'] = clb1_img
		
		# STEP 4: Add panel labels if requested
		if add_labels:
			# Labels for the four main sections in order: A=histones, B=cyclins, C=HTA1_HTB1, D=CLB1
			label_assignments = [
				('TrajH2A', 'a'),        # Histones column (leftmost)
				('LocusHTA1HTB1', 'b'),  # HTA1_HTB1 locus (second from left)
				('TrajG1', 'c'),         # Cyclins column (third from left)
				('LocusCLB1', 'd')       # CLB1 locus (rightmost)
			]
			
			for key, label in label_assignments:
				if key in all_placed_images:
					compositor.add_panel_label_to_image(
						key,
						label,
						offset=(-10, 8),
						font_size=font_size,
						font_type='bold'
					)
		
		# Save the composed figure
		compositor.save(panel_save_path)
		print(f"Panel layout saved to: {panel_save_path}")
		
		return panel_save_path

	def layout_mcm_panel(self, margin=(30, 30), between_padding=20, 
						 add_labels=True, font_size=36):
		"""
		Layout MCM-related figures horizontally in a single panel.
		
		Arranges three figures side by side:
		- locus_MCM7.png
		- locus_MCM5.png 
		- trajectories_group_MCM2-7_complex.png
		
		Parameters:
		-----------
		margin : tuple, default (30, 30)
			Horizontal and vertical margins around the panel
		between_padding : int, default 20
			Padding between images
		add_labels : bool, default True
			Whether to add panel labels (A, B, C)
		font_size : int, default 36
			Font size for panel labels
			
		Returns:
		--------
		str
			Path to the saved panel figure
		"""
		import os
		from pipeline.figure_composer_helpers import layout_images_horizontally
		
		# Set up compositor - adjust width as needed for three horizontal images
		compositor = FigureCompositor(1024, 590, debug_mode=True)
		image_dir = self.figures_dir
		panel_save_path = os.path.join(self.panel_figures_dir, 'Supplemental7_MCM_Panel.png')
		
		# Define image paths
		image_paths = [
			os.path.join(image_dir, 'trajectories_group_MCM1_and_MCM2-7_complex.png'),
			os.path.join(image_dir, 'locus_MCM2.png'),
			os.path.join(image_dir, 'locus_MCM7.png'),
		]
		
		# Define image keys for referencing
		image_keys = ['MCMTrajectories', 'MCM2Locus', 'MCM7Locus']
		
		# Verify all images exist
		missing_images = [path for path in image_paths if not os.path.exists(path)]
		if missing_images:
			raise FileNotFoundError(f"Missing image files: {missing_images}")
		
		# Layout images horizontally with equal proportions
		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			width_proportions=[0.905, 1.0, 1.0],  # Equal widths for all three images
			between_padding=between_padding,
			margin=margin,
			image_keys=image_keys
		)
		
		# Add panel labels if requested
		if add_labels:
			labels = ['a', 'b', 'c']
			for key, label in zip(image_keys, labels):
				if key in placed_images:
					compositor.add_panel_label_to_image(
						key,
						label,
						offset=(-10, 21),
						font_size=font_size,
						font_type='bold'
					)
		
		# Save the composed figure
		compositor.save(panel_save_path)
		print(f"MCM panel layout saved to: {panel_save_path}")
		
		return panel_save_path

