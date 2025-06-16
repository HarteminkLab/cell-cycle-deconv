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


class AntisenseTranscriptRunner:
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
	
	def __init__(self, output_directory, chromosome, min_length=80, max_gap=120, smooth=True):
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
		self.min_length = min_length
		self.max_gap = max_gap
		self.smooth = smooth
		
		# Initialize components
		self.timer = Timer()
		self.orf_plotter = load_default_orf_plotter()
		self.caller = None
		self.visualizer = None
		self.results_df = None
		
		# Setup directories
		self.save_dir = f'{output_directory}/antisense_calling'
		mkdir_safe(self.save_dir)
		
	def _load_all_chrom_reads(self, rna_filepaths_df, replicate, chrom):
		"""
		Load all RNA-seq reads for a specific chromosome and replicate.
		
		Parameters:
		-----------
		rna_filepaths_df : pd.DataFrame
			DataFrame with RNA-seq file paths indexed by replicate and time
		replicate : int
			Replicate number (1 or 2)
		chrom : int
			Chromosome number
			
		Returns:
		--------
		pd.DataFrame
			Combined reads for all timepoints for the specified chromosome
		"""
		all_chrom_reads_arr = []
		replicate_filepaths = rna_filepaths_df.loc[replicate]
		
		for i, (timepoint, row) in enumerate(replicate_filepaths.iterrows()):
			time_rna_reads = read_rna_bam(row.full_path, chroms=[chrom])
			single_chrom_reads = time_rna_reads[time_rna_reads['chr'] == chrom].copy()
			single_chrom_reads['sample'] = timepoint
			all_chrom_reads_arr.append(single_chrom_reads)
			
			self.timer.print_time(f"Replicate {replicate}: {i}/{len(replicate_filepaths)}")
			
			# Conserve memory
			del time_rna_reads
			collected = gc.collect()
			print(f"Freed {collected} objects after timepoint {timepoint}")

			print_memory_usage()
			
		return pd.concat(all_chrom_reads_arr)
	
	def load_pileup_data(self, on_cluster):
		"""
		Load RNA-seq pileup data for both replicates of the specified chromosome.
		"""
		print(f"Loading pileup data for chromosome {self.chromosome}")
		
		# Get RNA-seq file paths
		rna_filepaths_df = get_rna_seq_filepaths_df(on_cluster).set_index(['replicate', 'time'])
		
		# Load reads for both replicates
		print("Loading replicate 1...")
		all_chrom_reads_rep1 = self._load_all_chrom_reads(rna_filepaths_df, 1, self.chromosome)
		
		print("Loading replicate 2...")
		all_chrom_reads_rep2 = self._load_all_chrom_reads(rna_filepaths_df, 2, self.chromosome)
		
		# Get chromosome length
		chrom_len = get_chromosome_length(self.chromosome)
		
		# Initialize caller
		self.caller = AntisenseTranscriptCaller(
			chromosome=self.chromosome,
			min_length=self.min_length,
			max_gap=self.max_gap,
			timer=self.timer
		)
		
		# Load pileup data into caller
		self.caller.load_pileup_from_bam(
			all_chrom_reads_rep1, 
			all_chrom_reads_rep2,
			chrom_len, 
			smooth=self.smooth
		)
		
		print(f"Completed pileup loading for chromosome {self.chromosome}")
	
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
		Run the complete antisense transcript calling pipeline.
		
		Returns:
		--------
		pd.DataFrame
			Results dataframe with detected transcripts
		"""
		print(f"Starting antisense transcript calling pipeline for chromosome {self.chromosome}")
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
