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

		self.ptr_threshold = 1.25
		self.integration.identify_coordinated_genes('raw_rep1', 
		    ptr_threshold=self.ptr_threshold)
		self.integration.identify_coordinated_genes('raw_rep2', 
		    ptr_threshold=self.ptr_threshold)
		self.integration.identify_coordinated_genes(ptr_threshold=self.ptr_threshold)

	def create_integration_plots(self):

		plot_datasets = ['raw_rep1', 'raw_rep2', 'deconvolved']
		for dataset in plot_datasets:
			fig = self.integration.plot_ptr_correlations(dataset)
			save_figure_for_paper(f"{self.figures_dir}/expression_chromatin_{dataset}_scatter.png")

		metrics = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		figsizes = [(4, 3.25), (4, 3.5), (4, 4)]

		for i, metric in enumerate(metrics):
			fig = self.integration.create_gene_dataset_intersection_heatmap(metric,
			    figsize=figsizes[i])
			save_figure_for_paper(f"{self.figures_dir}/gene_inclusion_map_{metric}.png")

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
		
		print("Creating panel layout...")
		self.layout_supplemental_panel()
		
		print("Complete figure generation finished!")

