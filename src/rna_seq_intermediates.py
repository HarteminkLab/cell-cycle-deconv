import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
from src.sgd import get_chromosome_length
from src.read_bam import read_rna_bam, get_rna_seq_filepaths_df
from src.utils import mkdir_safe, print_memory_usage
from src.global_config import GlobalConstants
import gc


class RNASeqIntermediateManager:
	"""
	Class for saving and loading RNA-seq intermediate files per chromosome.
	
	Saves two types of files per chromosome:
	1. Raw RNA-seq reads DataFrames (filtered by chromosome)
	2. Processed pileup DataFrames (watson and crick strands)
	
	Files are saved in HDF5 format for efficient storage and retrieval.
	"""
	
	def __init__(self, output_directory: str, chromosome: int):
		"""
		Initialize the RNA-seq intermediate file manager.
		
		Parameters:
		-----------
		output_directory : str
			Base output directory for intermediate files
		chromosome : int
			Chromosome number to process
		"""
		self.output_directory = output_directory
		self.chromosome = chromosome
		
		# Setup intermediate directory structure
		self.intermediate_dir = f'{output_directory}/rna_seq_intermediate'
		mkdir_safe(self.intermediate_dir)
		
		# Create subdirectories for different data types
		self.reads_dir = os.path.join(self.intermediate_dir, 'reads')
		self.pileups_dir = os.path.join(self.intermediate_dir, 'pileups')
		mkdir_safe(self.reads_dir)
		mkdir_safe(self.pileups_dir)
		
		print(f"Initialized RNASeqIntermediateManager for chromosome {chromosome}")
		print(f"Intermediate directory: {self.intermediate_dir}")
	
	def save_chromosome_data(self, on_cluster: bool) -> None:
		"""
		Load RNA-seq data for the chromosome and save intermediate files.
		
		Parameters:
		-----------
		on_cluster : bool
			Whether running on cluster (affects file paths)
		"""
		print(f"Starting data processing and saving for chromosome {self.chromosome}")
		
		# Get RNA-seq file paths
		rna_filepaths_df = get_rna_seq_filepaths_df(on_cluster)
		
		# Process both replicates
		for replicate in [1, 2]:
			print(f"\nProcessing replicate {replicate}...")
			
			# Load raw reads
			all_chrom_reads = self._load_all_chrom_reads(rna_filepaths_df, replicate)
			
			# Save raw reads
			self._save_raw_reads(all_chrom_reads, replicate)
			
			# Compute and save pileups
			self._compute_and_save_pileups(all_chrom_reads, replicate)
			
			# Clean up memory
			del all_chrom_reads
			gc.collect()
			print_memory_usage()
		
		print(f"\nCompleted processing chromosome {self.chromosome}")
	
	def _load_all_chrom_reads(self, rna_filepaths_df: pd.DataFrame, replicate: int) -> pd.DataFrame:
		"""
		Load all RNA-seq reads for a specific chromosome and replicate.
		
		Parameters:
		-----------
		rna_filepaths_df : pd.DataFrame
			DataFrame with RNA-seq file paths indexed by replicate and time
		replicate : int
			Replicate number (1 or 2)
			
		Returns:
		--------
		pd.DataFrame
			Combined reads for all timepoints for the specified chromosome
		"""
		all_chrom_reads_arr = []
		replicate_filepaths = rna_filepaths_df.loc[replicate]
		
		print(f"  Loading reads from {len(replicate_filepaths)} timepoints...")
		
		for i, (timepoint, row) in enumerate(replicate_filepaths.iterrows()):
			time_rna_reads = read_rna_bam(row.full_path, chroms=[self.chromosome])
			single_chrom_reads = time_rna_reads[time_rna_reads['chr'] == self.chromosome].copy()
			single_chrom_reads['sample'] = timepoint
			all_chrom_reads_arr.append(single_chrom_reads)
			
			print(f"    Processed timepoint {i+1}/{len(replicate_filepaths)}: {timepoint}")
			
			# Clean up memory
			del time_rna_reads
			gc.collect()
		
		return pd.concat(all_chrom_reads_arr, ignore_index=True)
	
	def _save_raw_reads(self, all_chrom_reads: pd.DataFrame, replicate: int) -> None:
		"""
		Save raw chromosome reads to HDF5 file.
		
		Parameters:
		-----------
		all_chrom_reads : pd.DataFrame
			DataFrame with columns: chr, start, stop, strand, sample
		replicate : int
			Replicate number (1 or 2)
		"""
		filename = f'rna_seq_rep{replicate}_chr{self.chromosome}.h5'
		filepath = os.path.join(self.reads_dir, filename)
		
		# Save to HDF5 with compression
		all_chrom_reads.to_hdf(filepath, key='reads', mode='w', complevel=9, complib='zlib')
		
		print(f"  Saved raw reads to: {filename} ({len(all_chrom_reads):,} reads)")
	
	def _compute_and_save_pileups(self, all_chrom_reads: pd.DataFrame, replicate: int) -> None:
		"""
		Compute pileups and save to HDF5 files.
		
		Parameters:
		-----------
		all_chrom_reads : pd.DataFrame
			Chromosome reads data
		replicate : int
			Replicate number (1 or 2)
		"""
		print(f"  Computing pileups for replicate {replicate}...")
		
		# Get chromosome span
		chromosome_length = get_chromosome_length(self.chromosome)
		span = (0, chromosome_length)
		
		# Compute pileups (without smoothing)
		watson_pileups_df, crick_pileups_df, _ = self._compute_pileups_advanced(
			all_chrom_reads, span, replicate, smooth=False
		)
		
		# Save watson pileups
		watson_filename = f'watson_rna_seq_pileup_rep{replicate}_chr{self.chromosome}.h5'
		watson_filepath = os.path.join(self.pileups_dir, watson_filename)
		watson_pileups_df.to_hdf(watson_filepath, key='pileup', mode='w', complevel=9, complib='zlib')
		
		# Save crick pileups
		crick_filename = f'crick_rna_seq_pileup_rep{replicate}_chr{self.chromosome}.h5'
		crick_filepath = os.path.join(self.pileups_dir, crick_filename)
		crick_pileups_df.to_hdf(crick_filepath, key='pileup', mode='w', complevel=9, complib='zlib')
		
		print(f"  Saved pileups to: {watson_filename}, {crick_filename}")
		print(f"    Shape: {watson_pileups_df.shape} (timepoints x positions)")
	
	def _compute_pileups_advanced(self, all_chrom_reads: pd.DataFrame, span: Tuple[int, int], 
								replicate: int, smooth: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
		"""
		Optimized pileup computation using advanced numpy techniques.
		Adapted from AntisenseTranscriptCaller._compute_pileups_advanced.
		
		Parameters:
		-----------
		all_chrom_reads : pd.DataFrame
			Reads with columns: chr, start, stop, strand, sample
		span : tuple
			(start, end) coordinates for chromosome
		replicate : int
			Replicate number (1 or 2)
		smooth : bool
			Whether to apply smoothing (default: False)
			
		Returns:
		--------
		tuple
			(watson_pileups_df, crick_pileups_df, bin_boundaries)
		"""
		# Filter and sort reads for cache efficiency
		within_span_reads = all_chrom_reads[
			(all_chrom_reads['start'] < span[1]) & 
			(all_chrom_reads['stop'] > span[0])
		].copy()
		within_span_reads = within_span_reads.sort_values('start')
		
		# Get timepoints
		timepoints = (GlobalConstants.EXPRESSION_WT1_TIMEPOINTS if replicate == 1 
					 else GlobalConstants.EXPRESSION_WT2_TIMEPOINTS)
		
		# Create timepoint and strand mappings for vectorized operations
		timepoint_map = {tp: i for i, tp in enumerate(timepoints)}
		within_span_reads['timepoint_idx'] = within_span_reads['sample'].map(timepoint_map)
		within_span_reads['strand_idx'] = (within_span_reads['strand'] == '+').astype(int)
		
		# Pre-allocate result arrays
		n_positions = span[1] - span[0]
		n_timepoints = len(timepoints)
		result_shape = (2, n_timepoints, n_positions)  # [strand, timepoint, position]
		pileups = np.zeros(result_shape, dtype=np.int32)
		
		# Vectorized histogram computation
		bins = np.arange(span[0], span[1] + 1)  # Integer bin edges
		
		for strand_idx in [0, 1]:  # 0=crick(-), 1=watson(+)
			strand_reads = within_span_reads[within_span_reads['strand_idx'] == strand_idx]
			
			for tp_idx in range(n_timepoints):
				tp_reads = strand_reads[strand_reads['timepoint_idx'] == tp_idx]
				if len(tp_reads) > 0:
					counts, _ = np.histogram(tp_reads['start'].values, bins=bins)
					pileups[strand_idx, tp_idx, :] = counts
		
		# Convert to DataFrames with timepoints as index
		watson_pileups_df = pd.DataFrame(pileups[1, :, :], index=timepoints)
		crick_pileups_df = pd.DataFrame(pileups[0, :, :], index=timepoints)
		bin_boundaries = bins
		
		return watson_pileups_df, crick_pileups_df, bin_boundaries
	
	# =====================================================================
	# LOADING METHODS
	# =====================================================================
	
	def load_raw_reads(self, replicate: int) -> pd.DataFrame:
		"""
		Load raw chromosome reads from HDF5 file.
		
		Parameters:
		-----------
		replicate : int
			Replicate number (1 or 2)
			
		Returns:
		--------
		pd.DataFrame
			Raw reads with columns: chr, start, stop, strand, sample
		"""
		filename = f'rna_seq_rep{replicate}_chr{self.chromosome}.h5'
		filepath = os.path.join(self.reads_dir, filename)
		
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"Raw reads file not found: {filepath}")
		
		reads_df = pd.read_hdf(filepath, key='reads')
		print(f"Loaded raw reads from {filename}: {len(reads_df):,} reads")
		
		return reads_df
	
	def load_pileups(self, replicate: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Load watson and crick pileups from HDF5 files.
		
		Parameters:
		-----------
		replicate : int
			Replicate number (1 or 2)
			
		Returns:
		--------
		tuple
			(watson_pileups_df, crick_pileups_df)
		"""
		watson_filename = f'watson_rna_seq_pileup_rep{replicate}_chr{self.chromosome}.h5'
		crick_filename = f'crick_rna_seq_pileup_rep{replicate}_chr{self.chromosome}.h5'
		
		watson_filepath = os.path.join(self.pileups_dir, watson_filename)
		crick_filepath = os.path.join(self.pileups_dir, crick_filename)
		
		if not os.path.exists(watson_filepath):
			raise FileNotFoundError(f"Watson pileup file not found: {watson_filepath}")
		if not os.path.exists(crick_filepath):
			raise FileNotFoundError(f"Crick pileup file not found: {crick_filepath}")
		
		watson_pileups_df = pd.read_hdf(watson_filepath, key='pileup')
		crick_pileups_df = pd.read_hdf(crick_filepath, key='pileup')
		
		print(f"Loaded pileups from {watson_filename}, {crick_filename}")
		print(f"Shape: {watson_pileups_df.shape} (timepoints x positions)")
		
		return watson_pileups_df, crick_pileups_df
	
	def load_both_replicates_pileups(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		"""
		Load pileups for both replicates.
		
		Returns:
		--------
		tuple
			(watson_rep1, crick_rep1, watson_rep2, crick_rep2)
		"""
		watson_rep1, crick_rep1 = self.load_pileups(replicate=1)
		watson_rep2, crick_rep2 = self.load_pileups(replicate=2)
		
		return watson_rep1, crick_rep1, watson_rep2, crick_rep2
	
	def files_exist(self) -> bool:
		"""
		Check if all intermediate files exist for this chromosome.
		
		Returns:
		--------
		bool
			True if all 6 files exist, False otherwise
		"""
		# Check reads files
		reads_files = [
			f'rna_seq_rep1_chr{self.chromosome}.h5',
			f'rna_seq_rep2_chr{self.chromosome}.h5'
		]
		
		for filename in reads_files:
			filepath = os.path.join(self.reads_dir, filename)
			if not os.path.exists(filepath):
				return False
		
		# Check pileup files
		pileup_files = [
			f'watson_rna_seq_pileup_rep1_chr{self.chromosome}.h5',
			f'crick_rna_seq_pileup_rep1_chr{self.chromosome}.h5',
			f'watson_rna_seq_pileup_rep2_chr{self.chromosome}.h5',
			f'crick_rna_seq_pileup_rep2_chr{self.chromosome}.h5'
		]
		
		for filename in pileup_files:
			filepath = os.path.join(self.pileups_dir, filename)
			if not os.path.exists(filepath):
				return False
		
		return True
	
	def get_file_info(self) -> dict:
		"""
		Get information about saved files.
		
		Returns:
		--------
		dict
			File sizes and existence status
		"""
		info = {
			'chromosome': self.chromosome,
			'intermediate_dir': self.intermediate_dir,
			'reads_dir': self.reads_dir,
			'pileups_dir': self.pileups_dir,
			'files': {}
		}
		
		# Check reads files
		reads_pattern = 'rna_seq_rep{}_chr{}.h5'
		for replicate in [1, 2]:
			filename = reads_pattern.format(replicate, self.chromosome)
			filepath = os.path.join(self.reads_dir, filename)
			
			if os.path.exists(filepath):
				size_mb = os.path.getsize(filepath) / (1024 * 1024)
				info['files'][f'reads/{filename}'] = f"{size_mb:.2f} MB"
			else:
				info['files'][f'reads/{filename}'] = "Not found"
		
		# Check pileup files
		pileup_patterns = [
			'watson_rna_seq_pileup_rep{}_chr{}.h5',
			'crick_rna_seq_pileup_rep{}_chr{}.h5'
		]
		
		for replicate in [1, 2]:
			for pattern in pileup_patterns:
				filename = pattern.format(replicate, self.chromosome)
				filepath = os.path.join(self.pileups_dir, filename)
				
				if os.path.exists(filepath):
					size_mb = os.path.getsize(filepath) / (1024 * 1024)
					info['files'][f'pileups/{filename}'] = f"{size_mb:.2f} MB"
				else:
					info['files'][f'pileups/{filename}'] = "Not found"
		
		return info