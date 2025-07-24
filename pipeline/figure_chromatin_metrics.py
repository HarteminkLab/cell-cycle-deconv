import os
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

		plot_datasets = ['raw_rep1', 'raw_rep2', 'deconvolved']
		for dataset in plot_datasets:
			fig = self.integration.plot_ptr_correlations(dataset)
			save_figure_for_paper(f"{self.figures_dir}/expression_chromatin_{dataset}_scatter.png")

		# Heatmap of counts showing change from raw data to deconvolved, likely
		# a supplemental or omitted entirely
		metrics = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		figsizes = [(4, 3.25), (4, 3.5), (4, 4)]
		for i, metric in enumerate(metrics):
			fig = self.integration.create_gene_dataset_intersection_heatmap(metric,
				figsize=figsizes[i])
			save_figure_for_paper(f"{self.figures_dir}/gene_inclusion_map_{metric}.png")

		# Plot PTR vs PTR with trajectory highlights
		fig = self.integration.plot_ptr_correlations(color_by='normalized_trajectory_area')
		save_figure_for_paper(f"{self.figures_dir}/ptrs_vs_ptr_trajectory.png")

		# Plot PTR vs PTR with density
		fig = self.integration.plot_ptr_correlations(color_by='density')
		save_figure_for_paper(f"{self.figures_dir}/ptrs_vs_ptr.png")

		# Plot cell cycle genes, with trajectory area values
		self.integration.plot_all_trajectory_area_cell_cycle_genes()
		save_figure_for_paper(f"{self.figures_dir}/cell_cycle_trajectory_values.png")

		# Metric top and bottom exampels
		for metric in metrics:
			self.integration.plot_chromatin_example_trajectories(metric)
			save_figure_for_paper(f"{self.figures_dir}/top_bottom_trajectories_{metric}.png")

		# Sample genes to plot
		genes_to_plot = ['MCM7', 'CLB1', 'HHT1', 'HTA1']
		for gene in genes_to_plot:
			fig = self.integration.plot_all_metrics_all_replicates_gene(gene)
			save_figure_for_paper(f"{self.figures_dir}/trajectories_{gene}.png")

		from src.geneset import cyclin_genes, mcm_genes, histone_genes

		groups_to_plot = [
			histone_genes('H2A'),
			histone_genes('H2B'),
			histone_genes('H3'),
			histone_genes('H4'),
			mcm_genes(), 
			cyclin_genes('G1'), 
			cyclin_genes('B-S'), 
			cyclin_genes('B-M')]

		group_titles = [
			'Histones H2A', 
			'Histones H2B', 
			'Histones H3', 
			'Histones H4', 
			'MCM2-7 complex', 
			'G1-type cyclins', 
			'B-type cyclins (S)',
			'B-type cyclins (M)'
		]

		for i, gene_group in enumerate(groups_to_plot):
			fig = self.integration.plot_gene_group_trajectories_all_metrics(gene_group,
				title=group_titles[i])
			save_figure_for_paper(f"{self.figures_dir}/trajectories_group_{group_titles[i].replace(' ', '_')}.png")


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
				'figsize': (7, 11)
			},
			'MCM7': {
				'title': f"{get_gene_title_name('MCM7', include_system=False)}",
				'span_offset': (-1500, 1500),
				'save_name': 'locus_MCM7',
				'figsize': (7, 11)
			},
		}

		for gene_key, gene_dict in genes_to_plot_map.items():
			gene = genes[genes['gene'] == gene_key].iloc[0]
			span = gene.TSS+gene_dict['span_offset'][0], gene.TSS+gene_dict['span_offset'][1]
			loaded_data = genome_deconv_analysis.load_mnase_span(gene.chr, span)
			genome_deconv_analysis.plot_loaded_data(figsize=gene_dict['figsize'], title=gene_dict['title'])
			save_figure_for_paper(f'{self.figures_dir}/{gene_dict["save_name"]}.png')


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
			offset=(-10, -12)
		)
		
		# Save the composed figure
		output_path = f'{self.panel_figures_dir}/Supplemental_Chromatin_Metrics.png'
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
		
		print("Creating panel layout...")
		self.layout_metrics_panel1()
		self.layout_metrics_panel2()
		self.layout_genesets_panel()
		self.layout_mcm_panel()
		self.layout_supplemental_panel()

		print("Complete figure generation finished!")


	def layout_metrics_panel1(self, margin=(30, 30), between_padding=20, 
							  panel_padding=40, add_labels=True, font_size=36):
		"""Layout the figure panel for """
		import os

		from pipeline.figure_composer_helpers import layout_images_horizontally

		compositor = FigureCompositor(1024, 900, debug_mode=True)
		image_dir = self.figures_dir
		panel_save_path = os.path.join(self.panel_figures_dir, 'Figure5_Chromatin_Metrics.png')
		
		# Define image paths
		image_paths = {
			'raw_vs_deconv': os.path.join(image_dir, 'raw_vs_deconvolved_all_metrics_ptrs.png'),
			'clb1_traj': os.path.join(image_dir, 'trajectories_CLB1.png'),
			'mcm7_traj': os.path.join(image_dir, 'trajectories_MCM7.png')
		}
		
		# Verify all images exist
		missing_images = [path for path in image_paths.values() if not os.path.exists(path)]
		if missing_images:
			raise FileNotFoundError(f"Missing image files: {missing_images}")
		
		all_placed_images = {}
		
		# 1. Place the first figure full width at the top
		top_img = compositor.place_image(
			image_paths['raw_vs_deconv'],
			x=margin[0],
			y=margin[1],
			width=compositor.logical_width - (2 * margin[0]),
			name='RawVsDeconv'
		)
		all_placed_images['RawVsDeconv'] = top_img
		
		# 2. Place two images side by side below the first
		# Calculate y position for the second row
		second_row_y = top_img['logical_position'][1] + top_img['logical_size'][1] + panel_padding
		
		# Layout two images horizontally
		side_by_side_paths = [image_paths['clb1_traj'], image_paths['mcm7_traj']]
		side_by_side_keys = ['CLB1Trajectories', 'MCM7Trajectories']
		
		side_by_side_images = layout_images_horizontally(
			compositor,
			side_by_side_paths,
			width_proportions=[1, 1],  # Equal widths
			between_padding=between_padding,
			margin=(margin[0], second_row_y),
			image_keys=side_by_side_keys
		)
		all_placed_images.update(side_by_side_images)
		
		# 4. Add panel labels if requested
		if add_labels:
			# Order images for labeling: top, left, right, bottom
			ordered_keys = ['RawVsDeconv', 'CLB1Trajectories', 'MCM7Trajectories']
			labels = ['A', 'B', 'C']
			
			for key, label in zip(ordered_keys, labels):
				if key in all_placed_images:
					compositor.add_panel_label_to_image(
						key,
						label,
						offset=(-10, -12),
						font_size=font_size,
						font_type='bold'
					)
		
		# Save the composed figure
		compositor.save(panel_save_path)
		print(f"Panel layout saved to: {panel_save_path}")
		
		return panel_save_path


	def layout_metrics_panel2(self, margin=(30, 30), between_padding=20,
							 left_width_percent=49, add_labels=True, font_size=24):
		"""
		Layout metrics panel 2 with the following arrangement:
		- Left column (35% width): ptrs_vs_ptr_trajectory.png, cell_cycle_trajectory_values.png
		- Right column (65% width): top_bottom_trajectories_promoter_occupancy.png,
									top_bottom_trajectories_nucleosome_entropy.png,
									top_bottom_trajectories_nucleosome_occupancy.png
		"""
		import os

		image_dir = self.figures_dir
		panel_save_path = os.path.join(self.panel_figures_dir, 'Figure6_Chromatin_Transcription.png')

		compositor = FigureCompositor(1024, 480, debug_mode=True)
		
		# Define image paths
		image_paths = {
			'ptrs_vs_ptr': os.path.join(image_dir, 'ptrs_vs_ptr_trajectory.png'),
			'cell_cycle_values': os.path.join(image_dir, 'cell_cycle_trajectory_values.png'),
			'promoter_occupancy': os.path.join(image_dir, 'top_bottom_trajectories_promoter_occupancy.png'),
			'nucleosome_entropy': os.path.join(image_dir, 'top_bottom_trajectories_nucleosome_entropy.png'),
			'nucleosome_occupancy': os.path.join(image_dir, 'top_bottom_trajectories_nucleosome_occupancy.png')
		}
		
		# Verify all images exist
		missing_images = [path for path in image_paths.values() if not os.path.exists(path)]
		if missing_images:
			raise FileNotFoundError(f"Missing image files: {missing_images}")
		
		# Calculate column widths
		total_width = compositor.logical_width - (2 * margin[0])
		left_width = int(total_width * left_width_percent / 100)
		right_width = total_width - left_width - between_padding
		
		# Calculate column x positions
		left_x = margin[0]
		right_x = left_x + left_width + between_padding
		
		all_placed_images = {}
		
		# Left column - vertical layout (first two figures)
		left_column_paths = [
			image_paths['ptrs_vs_ptr'],
			image_paths['cell_cycle_values']
		]
		left_column_keys = ['PTRsVsPTR', 'CellCycleValues']
		
		left_images = layout_images_vertically(
			compositor,
			left_column_paths,
			between_padding=between_padding,
			margin=(left_x, margin[1]),
			widths=[left_width, left_width],
			image_keys=left_column_keys
		)
		all_placed_images.update(left_images)
		
		# Right column - vertical layout (last three figures)
		right_column_paths = [
			image_paths['promoter_occupancy'],
			image_paths['nucleosome_entropy'],
			image_paths['nucleosome_occupancy']
		]
		right_column_keys = [
			'PromoterOccupancy',
			'NucleosomeEntropy',
			'NucleosomeOccupancy'
		]
		
		right_images = layout_images_vertically(
			compositor,
			right_column_paths,
			between_padding=between_padding,
			margin=(right_x, margin[1]),
			widths=[right_width] * 3,
			image_keys=right_column_keys
		)
		all_placed_images.update(right_images)
		
		# Add panel labels if requested
		if add_labels:
			# Order images for labeling: left column first, then right column
			ordered_keys = [
				'PTRsVsPTR', 'CellCycleValues',  # Left column
				'PromoterOccupancy', 'NucleosomeEntropy', 'NucleosomeOccupancy'  # Right column
			]
			labels = ['A', 'B', 'C', 'D', 'E']
			
			for key, label in zip(ordered_keys, labels):
				if key in all_placed_images:
					compositor.add_panel_label_to_image(
						key,
						label,
						offset=(-10, -12),
						font_size=font_size,
						font_type='bold'
					)

		# Save the composed figure
		compositor.save(panel_save_path)
		print(f"Panel layout saved to: {panel_save_path}")
		
		return panel_save_path

	def layout_genesets_panel(self, margin=(30, 30), between_padding=30, traj_vertical_padding=15, 
							  panel_padding=30, add_labels=True, font_size=28):
		"""Layout the figure panel for genesets analysis"""
		import os
		
		from pipeline.figure_composer_helpers import layout_images_vertically

		compositor = FigureCompositor(1024, 520, debug_mode=True)
		image_dir = self.figures_dir
		panel_save_path = os.path.join(self.panel_figures_dir, 'Figure_Genesets_Panel.png')
		
		# Define image paths
		image_paths = {
			'locus_clb1': os.path.join(image_dir, 'locus_CLB1.png'),
			'locus_hta1_htb1': os.path.join(image_dir, 'locus_HTA1_HTB1.png'),
			'traj_g1': os.path.join(image_dir, 'trajectories_group_G1-type_cyclins.png'),
			'traj_s': os.path.join(image_dir, 'trajectories_group_B-type_cyclins_(S).png'),
			'traj_m': os.path.join(image_dir, 'trajectories_group_B-type_cyclins_(M).png'),
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
		
		# 1. Place locus images side by side on the left (60% width)
		# Calculate dimensions for each locus image
		locus_height = compositor.logical_height - (2 * margin[1])
		
		# Place CLB1 locus image
		clb1_img = compositor.place_image(
			image_paths['locus_clb1'],
			x=margin[0],
			y=margin[1],
			height=locus_height,
			name='LocusCLB1'
		)
		all_placed_images['LocusCLB1'] = clb1_img
		clb1_width = clb1_img['logical_size'][0]	

		# Place HTA1_HTB1 locus image
		hta1_htb1_img = compositor.place_image(
			image_paths['locus_hta1_htb1'],
			x=margin[0] + clb1_width + between_padding,
			y=margin[1],
			height=locus_height,
			name='LocusHTA1HTB1'
		)
		all_placed_images['LocusHTA1HTB1'] = hta1_htb1_img

		# 2. NOW calculate trajectory section dimensions based on actual locus widths
		hta1_width = hta1_htb1_img['logical_size'][0]
		actual_locus_section_width = clb1_width + between_padding + hta1_width
		traj_start_x = margin[0] + actual_locus_section_width + panel_padding

		# Calculate remaining width for trajectory columns
		remaining_width = compositor.logical_width - margin[0] - actual_locus_section_width - \
			panel_padding - margin[0]
		trajectory_column_width = (remaining_width - between_padding) // 2
		
		# 2. Place cyclins column (G1, S, M) - first column of trajectory section
		cyclin_paths = [image_paths['traj_g1'], image_paths['traj_s'], image_paths['traj_m']]
		cyclin_keys = ['TrajG1', 'TrajS', 'TrajM']

		cyclins_placed = layout_images_vertically(
			compositor=compositor,
			image_paths_arr=cyclin_paths,
			between_padding=traj_vertical_padding,
			margin=(0, margin[1]),  # No horizontal margin since we're setting x_position
			x_position=traj_start_x,
			image_keys=cyclin_keys,
			widths=[trajectory_column_width] * len(cyclin_paths),
			preserve_aspect_ratio=True
		)
		all_placed_images.update(cyclins_placed)

		# 3. Place histones column (H2A, H2B, H3, H4) - second column of trajectory section
		histone_paths = [image_paths['traj_h2a'], image_paths['traj_h2b'], 
						 image_paths['traj_h3'], image_paths['traj_h4']]
		histone_keys = ['TrajH2A', 'TrajH2B', 'TrajH3', 'TrajH4']

		trajectory_column_padding = 30
		histone_start_x = traj_start_x + trajectory_column_width + trajectory_column_padding

		histones_placed = layout_images_vertically(
			compositor=compositor,
			image_paths_arr=histone_paths,
			between_padding=traj_vertical_padding,
			margin=(0, margin[1]),  # No horizontal margin since we're setting x_position
			x_position=histone_start_x,
			image_keys=histone_keys,
			widths=[trajectory_column_width] * len(histone_paths),
			preserve_aspect_ratio=True
		)
		all_placed_images.update(histones_placed)
		
		# 4. Add panel labels if requested
		if add_labels:
			# Labels for the four main sections: A=CLB1, B=HTA1_HTB1, C=cyclins column, D=histones column
			label_assignments = [
				('LocusCLB1', 'A'),
				('LocusHTA1HTB1', 'B'),
				('TrajG1', 'C'),  # Label the cyclins column with the top image
				('TrajH2A', 'D')  # Label the histones column with the top image
			]
			
			for key, label in label_assignments:
				if key in all_placed_images:
					compositor.add_panel_label_to_image(
						key,
						label,
						offset=(-10, -12),
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
		- locus_MCM1.png 
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
		compositor = FigureCompositor(1024, 620, debug_mode=True)
		image_dir = self.figures_dir
		panel_save_path = os.path.join(self.panel_figures_dir, 'Figure_MCM_Panel.png')
		
		# Define image paths
		image_paths = [
			os.path.join(image_dir, 'locus_MCM7.png'),
			os.path.join(image_dir, 'locus_MCM1.png'),
			os.path.join(image_dir, 'trajectories_group_MCM2-7_complex.png')
		]
		
		# Define image keys for referencing
		image_keys = ['MCM7Locus', 'MCM1Locus', 'MCMTrajectories']
		
		# Verify all images exist
		missing_images = [path for path in image_paths if not os.path.exists(path)]
		if missing_images:
			raise FileNotFoundError(f"Missing image files: {missing_images}")
		
		# Layout images horizontally with equal proportions
		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			width_proportions=[1.01, 1.01, 0.96],  # Equal widths for all three images
			between_padding=between_padding,
			margin=margin,
			image_keys=image_keys
		)
		
		# Add panel labels if requested
		if add_labels:
			labels = ['A', 'B', 'C']
			for key, label in zip(image_keys, labels):
				if key in placed_images:
					compositor.add_panel_label_to_image(
						key,
						label,
						offset=(-10, -12),
						font_size=font_size,
						font_type='bold'
					)
		
		# Save the composed figure
		compositor.save(panel_save_path)
		print(f"MCM panel layout saved to: {panel_save_path}")
		
		return panel_save_path


