import pandas as pd
from src.sgd import get_chromosome_length
from src.orf_plotter import load_default_orf_plotter
from src.timer import Timer
from src.read_bam import read_rna_bam, get_rna_seq_filepaths_df
from src.transcript_boundary_caller import AntisenseTranscriptCaller
from src.transcript_boundary_visualizer import AntisenseTranscriptVisualizer
from src.figure_configs import save_figure_for_paper
from src.utils import mkdir_safe, print_memory_usage
import gc

# todo: rename file to transcript calling runner (not-antisense)

class TranscriptCallerRunner:
	"""
	Orchestrator class for running the complete antisense transcript calling pipeline.
	
	This class handles:
	1. Setup output and save directory
	2. Load pileup data for specified chromosome
	3. Find transcript boundaries
	4. Classify gene associations
	5. Save distribution plots
	6. Save called transcripts to disk
	"""
	
	def __init__(self, output_directory, chromosome):
		"""
		Initialize the antisense transcript calling pipeline.
		
		Parameters:
		-----------
		output_directory : str
			Base output directory for results
		chromosome : int
			Chromosome number to process
		min_length : int, optional
			Minimum transcript length (default: 80)
		max_gap : int, optional
			Maximum gap to merge regions (default: 120)
		smooth : bool, optional
			Whether to smooth pileup data (default: True)
		"""
		self.output_directory = output_directory
		self.chromosome = chromosome
		
		# Initialize components
		self.timer = Timer()
		self.orf_plotter = load_default_orf_plotter()
		self.visualizer = None
		self.results_df = None
		
		# Setup directories
		self.save_dir = f'{output_directory}/transcripts_calling'
		mkdir_safe(self.save_dir)

		from src.transcript_boundary_caller import TranscriptBoundaryCaller

		# Create caller for entire chromosome
		self.caller = TranscriptBoundaryCaller(output_directory=output_directory)
		span = 0, get_chromosome_length(chrom)
		self.caller.set_chrom_span(chrom, span)
		
	
	def detect_transcripts(self):
		"""
		Detect transcript boundaries and annotate gene associations.
		"""
		if self.caller is None:
			raise ValueError("Must load pileup data first. Call load_pileup_data()")
			
		print(f"Detecting transcript boundaries for chromosome {self.chromosome}")
		
		# Detect transcript boundaries
		self.results_df = self.caller.detect_transcript_boundaries(strand='both')
		
		# Annotate transcript-gene overlap
		self.results_df = self.caller.annotate_transcript_gene_overlap()
		
		print(f"Detected {len(self.results_df)} transcripts for chromosome {self.chromosome}")
	
	def save_results(self):
		"""
		Save transcript calling results and distribution plot to disk.
		"""
		if self.results_df is None:
			raise ValueError("Must detect transcripts first. Call detect_transcripts()")
			
		# Save transcript results to CSV
		output_file = f"{self.save_dir}/called_transcripts_chr{self.chromosome}.csv"
		self.results_df.to_csv(output_file, index=False)
		print(f"Saved transcript results to: {output_file}")
		
		# Initialize visualizer and create distribution plot
		self.visualizer = AntisenseTranscriptVisualizer(
			transcript_caller=self.caller,
			orf_plotter=self.orf_plotter,
		)
		
		# Plot and save threshold distribution
		fig = self.visualizer.plot_threshold_distribution()
		plot_file = f"{self.save_dir}/distribution_chr{self.chromosome}.png"
		save_figure_for_paper(plot_file)
		print(f"Saved distribution plot to: {plot_file}")
	
	def run(self, on_cluster):
		"""
		Run the complete transcript calling pipeline.
		
		Returns:
		--------
		pd.DataFrame
			Results dataframe with detected transcripts
		"""
		print(f"Starting transcript calling pipeline for chromosome {self.chromosome}")
		print(f"Output directory: {self.save_dir}")
		
		# Step 1 & 2: Setup and chromosome selection (done in __init__)
		
		# Step 3: Load pileup data
		self.load_pileup_data(on_cluster)
		
		# Step 4 & 5: Find transcript boundaries and classify gene associations
		self.detect_transcripts()
		
		# Step 6 & 7: Save distribution plot and transcript results
		self.save_results()
		
		print(f"Pipeline completed successfully for chromosome {self.chromosome}")
		return self.results_df
