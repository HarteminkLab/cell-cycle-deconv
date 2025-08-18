import pandas as pd
from src.sgd import get_chromosome_length
from src.orf_plotter import load_default_orf_plotter
from src.timer import Timer
from src.read_bam import read_rna_bam, get_rna_seq_filepaths_df
from src.transcript_boundary_caller import TranscriptBoundaryCaller
from src.figure_configs import save_figure_for_paper
from src.utils import mkdir_safe


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
		self.save_dir = f"{output_directory}/transcripts_calling"

		# Create caller for entire chromosome
		self.caller = TranscriptBoundaryCaller(output_directory=output_directory)
		span = 0, get_chromosome_length(chromosome)
		self.span = span
		self.caller.set_chrom_span(chromosome, span)
		
	
	def detect_transcripts(self):
		"""
		Detect transcript boundaries and annotate gene associations.
		"""
		print(f"Detecting transcript boundaries for chromosome {self.chromosome}")
		
		# Detect transcript boundaries
		self.results_df = self.caller.call_boundaries_both_strands()
		
		# Annotate transcript-gene overlap
		from src.sgd import read_sgd_genes
		genes_df = read_sgd_genes(remove_chr_roman=True)
		self.results_df = self.caller.assign_genes_to_transcripts(genes_df)
		
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
	
	def run(self):
		"""
		Run the complete transcript calling pipeline.
		
		Returns:
		--------
		pd.DataFrame
			Results dataframe with detected transcripts
		"""
		print(f"Starting transcript calling pipeline for chromosome {self.chromosome}")
		
		# Step 1: Find transcript boundaries and classify gene associations
		self.detect_transcripts()
		
		print(f"Transcript calling completed for chromosome {self.chromosome}")
		mkdir_safe(self.save_dir)
		self.save_results()

		return self.results_df


def remove_duplicate_transcripts(df, index_name='orf_name'):
	"""
	Remove duplicate transcripts by keeping the one with the closest position.
	
	For '+' strand: keeps transcript with rna_start closest to start
	For '-' strand: keeps transcript with rna_stop closest to stop
	
	Parameters:
	-----------
	df : pandas.DataFrame
		DataFrame with columns: rna_start, rna_stop, strand, start, stop
		Index should contain the gene identifiers
	
	Returns:
	--------
	pandas.DataFrame
		Deduplicated DataFrame with same structure as input
	"""

	# Create a copy to avoid modifying original
	df = df.copy()
	df.index.name = index_name
	df_work = df.reset_index()
	
	# Add distance column
	df_work['_distance'] = df_work.apply(calculate_distance, axis=1)
	
	# Group by index (gene name) and keep row with minimum distance
	# In case of ties, keep the first occurrence
	idx_to_keep = df_work.groupby(index_name)['_distance'].idxmin()
	
	# Filter original dataframe to keep only selected rows
	result = df_work.loc[idx_to_keep]
	
	return result.set_index(index_name)[df.columns]


# Calculate distance based on strand
def calculate_distance(row, start_keys=['rna_start', 'start'], stop_keys=['rna_stop', 'stop'], 
		compute_absolute=True):

	if row['strand'] == '+':
		if compute_absolute:
			return abs(row[start_keys[0]] - row[start_keys[1]])
		else:
			return (row[start_keys[0]] - row[start_keys[1]])

	else:  # strand == '-'
		if compute_absolute:
			return abs(row[stop_keys[0]] - row[stop_keys[1]])
		else:
			return (row[stop_keys[0]] - row[stop_keys[1]])
