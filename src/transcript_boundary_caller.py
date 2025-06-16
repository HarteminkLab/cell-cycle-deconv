import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Optional, Union
from scipy.stats.distributions import norm
import warnings


class AntisenseTranscriptCaller:
	"""
	Class for detecting antisense transcript boundaries from stranded RNA-seq data.
	
	Uses a two-threshold approach with quantile-based thresholds for robust detection:
	1. Primary threshold (default: 75th percentile) identifies transcript cores
	2. Extension threshold (default: 50th percentile) extends to natural boundaries
	"""
	
	def __init__(self, 
				 chromosome: int,
				 primary_threshold_quantile: float = 0.25,
				 extension_threshold_quantile: float = 0.1,
				 min_length: int = 100,
				 max_gap: int = 50,
				 smoothing_sigma: float = 10,
				 smoothing_window: int = 60,
				 timer=None):
		"""
		Initialize the antisense transcript caller.
		
		Parameters:
		-----------
		chromosome : int
			Chromosome number to analyze
		primary_threshold_quantile : float
			Quantile for primary threshold (default: 0.75)
		extension_threshold_quantile : float  
			Quantile for extension threshold (default: 0.50)
		min_length : int
			Minimum transcript length in bp (default: 100)
		max_gap : int
			Maximum gap to merge nearby regions (default: 50)
		smoothing_sigma : float
			Gaussian smoothing sigma parameter (default: 2.0)
		smoothing_window : int
			Smoothing window size (default: 60)
		timer : Timer object
			Optional timer for performance monitoring
		"""
		
		self.chromosome = chromosome
		self.primary_threshold_quantile = primary_threshold_quantile
		self.extension_threshold_quantile = extension_threshold_quantile
		self.min_length = min_length
		self.max_gap = max_gap
		self.smoothing_sigma = smoothing_sigma
		self.smoothing_window = smoothing_window
		
		# Timer for performance monitoring
		self.timer = timer
		self._log_time = self._create_timer_function()
		
		# Data storage
		self.watson_pileups_df = None
		self.crick_pileups_df = None
		self.bin_boundaries = None
		self.chromosome_span = None
		
		# Results storage
		self.watson_transcripts = None
		self.crick_transcripts = None
		self.results_df = None
		
	def _create_timer_function(self):
		"""Create timer function based on whether timer is available."""
		if self.timer is not None:
			return self.timer.print_time
		else:
			return lambda msg: print(f"[AntisenseTranscriptCaller] {msg}")
	
	def load_pileup_from_bam(self, 
						   all_chrom_reads_rep1: pd.DataFrame,
						   all_chrom_reads_rep2: pd.DataFrame,
						   chromosome_length: int,
						   smooth: bool = True):
		"""
		Load and compute pileup data from BAM-derived reads DataFrames.
		
		Parameters:
		-----------
		all_chrom_reads_rep1 : pd.DataFrame
			Reads from replicate 1 with columns: chr, start, stop, strand, sample
		all_chrom_reads_rep2 : pd.DataFrame  
			Reads from replicate 2 with columns: chr, start, stop, strand, sample
		chromosome_length : int
			Length of chromosome in bp
		smooth : bool
			Whether to apply Gaussian smoothing (default: True)
		"""

		self.timer.start()
		
		self._log_time(f"Starting pileup computation for chromosome {self.chromosome}")
		
		# Set chromosome span
		self.chromosome_span = (0, chromosome_length)
		
		# Compute pileups for both replicates
		watson_rep1, crick_rep1, bins = self._compute_pileups_advanced(
			all_chrom_reads_rep1, self.chromosome_span, replicate=1, smooth=smooth)
		
		watson_rep2, crick_rep2, _ = self._compute_pileups_advanced(
			all_chrom_reads_rep2, self.chromosome_span, replicate=2, smooth=smooth)
		
		# Combine replicates (average)
		self.watson_pileups_df = (watson_rep1 + watson_rep2) / 2
		self.crick_pileups_df = (crick_rep1 + crick_rep2) / 2
		self.bin_boundaries = bins
		
		self._log_time("Completed pileup computation and replicate averaging")
		
	def load_pileup_from_arrays(self,
							  watson_pileups: np.ndarray,
							  crick_pileups: np.ndarray,
							  chromosome_span: Tuple[int, int],
							  smooth: bool = True):
		"""
		Load pre-computed pileup arrays.
		
		Parameters:
		-----------
		watson_pileups : np.ndarray
			Watson strand pileup values (timepoints x positions)
		crick_pileups : np.ndarray
			Crick strand pileup values (timepoints x positions)  
		chromosome_span : Tuple[int, int]
			Start and end coordinates of chromosome
		smooth : bool
			Whether to apply Gaussian smoothing (default: True)
		"""
		
		self._log_time("Loading pre-computed pileup arrays")
		
		self.chromosome_span = chromosome_span
		self.bin_boundaries = np.arange(chromosome_span[0], chromosome_span[1] + 1)
		
		# Convert to DataFrames
		self.watson_pileups_df = pd.DataFrame(watson_pileups)
		self.crick_pileups_df = pd.DataFrame(crick_pileups)
		
		if smooth:
			self._apply_smoothing()
			
		self._log_time("Completed pileup array loading")
		
	def detect_transcript_boundaries(self, strand: str = 'both') -> pd.DataFrame:
		"""
		Detect transcript boundaries using two-threshold approach.
		
		Parameters:
		-----------
		strand : str
			Which strand(s) to analyze: 'watson', 'crick', or 'both' (default: 'both')
			Custom (primary, extension) thresholds instead of quantile-based
			
		Returns:
		--------
		pd.DataFrame
			Results with columns: chromosome, strand, start, end, length
		"""
		
		if self.watson_pileups_df is None or self.crick_pileups_df is None:
			raise ValueError("No pileup data loaded. Call load_pileup_from_bam() or load_pileup_from_arrays() first.")
		
		self._log_time("Starting transcript boundary detection")
		
		# Compute average pileups across timepoints
		watson_avg = self.watson_pileups_df.mean(axis=0).values
		crick_avg = self.crick_pileups_df.mean(axis=0).values
		
		# Convert to log2 scale
		watson_log = np.log2(watson_avg + 1)
		crick_log = np.log2(crick_avg + 1)
		
		# Determine thresholds
		eps_zero_cutoff = 0.1
		combined_data = np.concatenate([watson_log, crick_log])
		primary_threshold = np.quantile(combined_data[combined_data > eps_zero_cutoff], self.primary_threshold_quantile)
		extension_threshold = np.quantile(combined_data[combined_data > eps_zero_cutoff], self.extension_threshold_quantile)
		self._log_time(f"Calculated quantile thresholds: primary={primary_threshold:.3f} (Q{self.primary_threshold_quantile}), extension={extension_threshold:.3f} (Q{self.extension_threshold_quantile})")
		
		# Detect boundaries for each strand
		results = []
		
		if strand in ['watson', 'both']:
			self.watson_transcripts = self._find_transcript_boundaries_optimized(
				watson_log, primary_threshold, extension_threshold)
			self._log_time(f"Found {len(self.watson_transcripts)} Watson transcripts")
			
			# Convert to genomic coordinates and add to results
			for start_idx, end_idx in self.watson_transcripts:
				results.append({
					'chromosome': self.chromosome,
					'strand': '+',
					'start': self.chromosome_span[0] + start_idx,
					'end': self.chromosome_span[0] + end_idx,
					'length': end_idx - start_idx + 1
				})
		
		if strand in ['crick', 'both']:
			self.crick_transcripts = self._find_transcript_boundaries_optimized(
				crick_log, primary_threshold, extension_threshold)
			self._log_time(f"Found {len(self.crick_transcripts)} Crick transcripts")
			
			# Convert to genomic coordinates and add to results
			for start_idx, end_idx in self.crick_transcripts:
				results.append({
					'chromosome': self.chromosome,
					'strand': '-',
					'start': self.chromosome_span[0] + start_idx,
					'end': self.chromosome_span[0] + end_idx,
					'length': end_idx - start_idx + 1
				})
		
		# Create results DataFrame
		self.results_df = pd.DataFrame(results)
		
		self._log_time(f"Completed boundary detection: {len(results)} total transcripts")
		self.primary_threshold = primary_threshold
		self.extension_threshold = extension_threshold
		
		return self.results_df

	def annotate_transcript_gene_overlap(self, buffer: int = 100,
											column_name: str = 'overlapping_gene') -> pd.DataFrame:
		"""
		Annotate transcript boundaries with overlapping gene information.
		For multiple overlapping genes, selects the 5' most gene.
		"""
		from src.sgd import read_sgd_genes
		genes_df = read_sgd_genes(remove_chr_roman=True)
		
		if self.results_df is None:
			raise ValueError("No transcript results available. Run detect_transcript_boundaries() first.")
		
		self._log_time(f"Starting transcript gene annotation with {buffer}bp buffer (5' most priority)")
		
		# Filter genes to current chromosome
		chromosome_genes = genes_df[genes_df['chr'] == self.chromosome].copy()
		
		if len(chromosome_genes) == 0:
			self._log_time(f"No genes found on chromosome {self.chromosome}")
			self.results_df[column_name] = None
			return self.results_df
		
		# Initialize annotation column
		gene_annotations = []
		
		# Annotate each transcript
		for _, transcript in self.results_df.iterrows():
			transcript_strand = transcript['strand']
			transcript_start = transcript['start']
			transcript_end = transcript['end']
			
			# Filter genes to same strand
			same_strand_genes = chromosome_genes[chromosome_genes['strand'] == transcript_strand]
			
			# Find ALL overlapping genes
			overlapping_genes = []
			
			for gene_orf_name, gene in same_strand_genes.iterrows():
				gene_start_buffered = gene['start'] - buffer
				gene_end_buffered = gene['stop'] + buffer
				
				# Check for overlap
				if (transcript_start <= gene_end_buffered and 
					transcript_end >= gene_start_buffered):
					overlapping_genes.append((gene_orf_name, gene['start']))
			
			# Select the 5' most gene among overlapping genes
			if overlapping_genes:
				if transcript_strand == '+':
					# Watson: 5' most = leftmost = minimum start
					selected_gene = min(overlapping_genes, key=lambda x: x[1])[0]
				else:
					# Crick: 5' most = rightmost = maximum start  
					selected_gene = max(overlapping_genes, key=lambda x: x[1])[0]
			else:
				selected_gene = None
			
			gene_annotations.append(selected_gene)
		
		# Add annotation column to results
		self.results_df[column_name] = gene_annotations
		
		# Enhanced logging
		total_transcripts = len(self.results_df)
		overlapping_count = sum(1 for x in gene_annotations if x is not None)
		
		# Count multiple overlaps for reporting
		multiple_overlap_count = 0
		for _, transcript in self.results_df.iterrows():
			same_strand_genes = chromosome_genes[chromosome_genes['strand'] == transcript['strand']]
			overlaps = []
			for _, gene in same_strand_genes.iterrows():
				gene_start_buffered = gene['start'] - buffer
				gene_end_buffered = gene['stop'] + buffer
				if (transcript['start'] <= gene_end_buffered and 
					transcript['end'] >= gene_start_buffered):
					overlaps.append(gene.name)
			if len(overlaps) > 1:
				multiple_overlap_count += 1
		
		self._log_time(f"Annotation complete: {overlapping_count} transcripts overlap with genes, "
					   f"{multiple_overlap_count} had multiple overlaps (5' most selected)")
		
		return self.results_df

	def get_non_overlapping_transcripts(self) -> pd.DataFrame:
		"""
		Get transcripts that don't overlap with any genes.
		
		Returns:
		--------
		pd.DataFrame
			Subset of results_df where overlapping_gene column is None
		"""
		if self.results_df is None:
			raise ValueError("No transcript results available. Run detect_transcript_boundaries() first.")
		
		if 'overlapping_gene' not in self.results_df.columns:
			raise ValueError("Gene overlap annotation not found. Run annotate_transcript_gene_overlap() first.")
		
		return self.results_df[self.results_df['overlapping_gene'].isna()].copy()

	def get_gene_overlapping_transcripts(self) -> pd.DataFrame:
		"""
		Get transcripts that overlap with genes.
		
		Returns:
		--------
		pd.DataFrame
			Subset of results_df where overlapping_gene column is not None
		"""
		if self.results_df is None:
			raise ValueError("No transcript results available. Run detect_transcript_boundaries() first.")
		
		if 'overlapping_gene' not in self.results_df.columns:
			raise ValueError("Gene overlap annotation not found. Run annotate_transcript_gene_overlap() first.")
		
		return self.results_df[self.results_df['overlapping_gene'].notna()].copy()
		
	def save_results(self, filepath: str):
		"""Save results to CSV file."""
		if self.results_df is None:
			raise ValueError("No results to save. Run detect_transcript_boundaries() first.")
			
		self.results_df.to_csv(filepath, index=False)
		self._log_time(f"Results saved to {filepath}")
		
	def get_transcript_summary(self) -> dict:
		"""Get summary statistics of detected transcripts."""
		if self.results_df is None:
			return {}
			
		summary = {
			'total_transcripts': len(self.results_df),
			'watson_transcripts': len(self.results_df[self.results_df.strand == '+']),
			'crick_transcripts': len(self.results_df[self.results_df.strand == '-']),
			'mean_length': self.results_df.length.mean(),
			'median_length': self.results_df.length.median(),
			'length_range': (self.results_df.length.min(), self.results_df.length.max())
		}
		
		return summary
	
	# =====================================================================
	# INTERNAL METHODS (optimized versions from notebook)
	# =====================================================================
	
	def _compute_pileups_advanced(self, all_chrom_reads, span, replicate, smooth=False):
		"""
		Optimized pileup computation using advanced numpy techniques.
		Adapted from notebook's compute_pileups_advanced function.
		"""
		from src.global_config import GlobalConstants
		
		# Filter and sort reads for cache efficiency
		within_span_reads = all_chrom_reads[(all_chrom_reads['start'] < span[1]) & 
											(all_chrom_reads['stop'] > span[0])].copy()
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
		
		# Apply smoothing if requested
		if smooth:
			for strand_idx in [0, 1]:
				for tp_idx in range(n_timepoints):
					pileups[strand_idx, tp_idx, :] = self._smooth_rna_curve(
						pileups[strand_idx, tp_idx, :].astype(np.float64)
					)
		
		# Convert to DataFrames
		watson_pileups_df = pd.DataFrame(pileups[1, :, :], index=timepoints)
		crick_pileups_df = pd.DataFrame(pileups[0, :, :], index=timepoints)
		bin_boundaries = bins
		
		return watson_pileups_df, crick_pileups_df, bin_boundaries
	
	def _apply_smoothing(self):
		"""Apply Gaussian smoothing to loaded pileup data."""
		self._log_time("Applying Gaussian smoothing")
		
		# Apply smoothing to each timepoint
		for idx in self.watson_pileups_df.index:
			self.watson_pileups_df.loc[idx] = self._smooth_rna_curve(self.watson_pileups_df.loc[idx].values)
			self.crick_pileups_df.loc[idx] = self._smooth_rna_curve(self.crick_pileups_df.loc[idx].values)

	def _get_smoothing_kernel(self, plot=False):
		n = self.smoothing_window
		xs = np.linspace(-n//2, n//2+1, n)
		kernel = norm.pdf(xs, 0, self.smoothing_sigma)
		kernel = kernel/kernel.max()

		if plot:
			import matplotlib.pyplot as plt
			plt.plot(xs, kernel)

		return kernel
	
	def _smooth_rna_curve(self, input_pileup):
		"""
		Apply Gaussian smoothing to RNA pileup curve.
		Adapted from notebook's smooth_rna_curve function.
		"""
		# Create Gaussian kernel
		kernel = self._get_smoothing_kernel()
		
		# Apply convolution
		smoothed_pileup = np.convolve(input_pileup, kernel, mode='same')
		return smoothed_pileup
	
	def _find_transcript_boundaries_optimized(self, pileup_values, primary_threshold, extension_threshold):
		"""
		Optimized transcript boundary identification.
		Adapted from notebook's find_transcript_boundaries function.
		"""
		
		# Step 1: Find core regions above primary threshold (vectorized)
		core_mask = pileup_values >= primary_threshold
		core_regions = self._get_regions_from_mask_optimized(core_mask)
		
		if not core_regions:
			return []
		
		# Step 2: Vectorized extension using extension threshold
		extension_mask = pileup_values >= extension_threshold
		extended_regions = self._extend_regions_vectorized(core_regions, extension_mask)
		
		# Step 3: Merge overlapping or nearby regions (optimized)
		merged_regions = self._merge_nearby_regions_optimized(extended_regions, self.max_gap)
		
		# Step 4: Filter by minimum length (vectorized)
		final_regions = self._filter_by_length_vectorized(merged_regions, self.min_length)
		
		return final_regions
	
	def _get_regions_from_mask_optimized(self, mask):
		"""Optimized conversion of boolean mask to regions using vectorized operations."""
		if not np.any(mask):
			return []
		
		# Vectorized approach: find all transitions at once
		padded_mask = np.concatenate(([False], mask, [False]))
		diff = np.diff(padded_mask.astype(np.int8))
		
		# Find start and end positions in one operation
		starts = np.where(diff == 1)[0]
		ends = np.where(diff == -1)[0] - 1
		
		return list(zip(starts, ends))
	
	def _extend_regions_vectorized(self, regions, extension_mask):
		"""Vectorized region extension."""
		if not regions:
			return []
		
		extended_regions = []
		mask_length = len(extension_mask)
		
		# Convert to numpy for faster processing
		regions_array = np.array(regions)
		
		for start, end in regions_array:
			# Vectorized left extension
			if start > 0:
				# Find the leftmost position where extension_mask becomes False
				left_chunk = extension_mask[:start][::-1]  # Reverse for cumsum
				if len(left_chunk) > 0:
					# Find first False in reversed array
					false_positions = np.where(~left_chunk)[0]
					if len(false_positions) > 0:
						extended_start = start - false_positions[0]
					else:
						extended_start = 0
				else:
					extended_start = start
			else:
				extended_start = start
				
			# Vectorized right extension
			if end < mask_length - 1:
				# Find the rightmost position where extension_mask becomes False
				right_chunk = extension_mask[end + 1:]
				if len(right_chunk) > 0:
					false_positions = np.where(~right_chunk)[0]
					if len(false_positions) > 0:
						extended_end = end + false_positions[0]
					else:
						extended_end = mask_length - 1
				else:
					extended_end = end
			else:
				extended_end = end
				
			extended_regions.append((extended_start, extended_end))
		
		return extended_regions
	
	def _merge_nearby_regions_optimized(self, regions, max_gap):
		"""Optimized merging using numpy operations."""
		if not regions:
			return []
		
		if len(regions) == 1:
			return regions
		
		# Convert to numpy array for vectorized operations
		regions_array = np.array(sorted(regions))
		starts = regions_array[:, 0]
		ends = regions_array[:, 1]
		
		# Vectorized gap calculation
		gaps = starts[1:] - ends[:-1] - 1
		
		# Find regions that should be merged (gap <= max_gap)
		merge_mask = gaps <= max_gap
		
		# Build merged regions efficiently
		merged = []
		current_start = starts[0]
		current_end = ends[0]
		
		for i, should_merge in enumerate(merge_mask):
			if should_merge:
				# Extend current region
				current_end = max(current_end, ends[i + 1])
			else:
				# Finish current region and start new one
				merged.append((current_start, current_end))
				current_start = starts[i + 1]
				current_end = ends[i + 1]
		
		# Add the last region
		merged.append((current_start, current_end))
		
		return merged
	
	def _filter_by_length_vectorized(self, regions, min_length):
		"""Vectorized length filtering."""
		if not regions:
			return []
		
		# Convert to numpy for vectorized length calculation
		regions_array = np.array(regions)
		lengths = regions_array[:, 1] - regions_array[:, 0] + 1
		
		# Vectorized filtering
		valid_mask = lengths >= min_length
		valid_regions = regions_array[valid_mask]
		
		return [tuple(region) for region in valid_regions]


def call_chromosome_TSSes(caller_results_df, chromosome_num):
    """
    Identify RNA-seq called TSSes for a single chromosome.
    
    Parameters:
    caller_results_df: DataFrame with transcript calls from AntisenseTranscriptCaller
    chromosome_num: int, chromosome number to process
    
    Returns:
    DataFrame with gene_id as index and rna_TSS column
    """
    # Filter for chromosome and overlapping genes
    genes_with_called_transcripts = caller_results_df[
        (caller_results_df.chromosome == chromosome_num) & 
        (~caller_results_df.overlapping_gene.isna())
    ]
    
    # Split by strand
    watson_genes = genes_with_called_transcripts[genes_with_called_transcripts.strand == '+']
    crick_genes = genes_with_called_transcripts[genes_with_called_transcripts.strand == '-']
    
    # For Watson (+): TSS is minimum start (5' end)
    watson_TSSes = watson_genes.groupby('overlapping_gene')[['start']].min()\
        .rename(columns={'start': 'rna_TSS'})
    
    # For Crick (-): TSS is maximum end (5' end) - CORRECTED
    crick_TSSes = crick_genes.groupby('overlapping_gene')[['end']].max()\
        .rename(columns={'end': 'rna_TSS'})
    
    return pd.concat([watson_TSSes, crick_TSSes])

def compare_all_chromosomes_TSS(combined_rna_TSSes, reference_tss_column='TSS'):
    """
    Compare RNA-seq called TSSes across all chromosomes with existing annotations.
    
    Parameters:
    caller_results_dict: dict, {chromosome: AntisenseTranscriptCaller.results_df}
    reference_tss_column: str, 'TSS' or 'park_TSS'
    
    Returns:
    DataFrame with all TSS comparisons and creates histogram
    """
    from src.sgd import read_nondubious_genes_dataset
    
    # Get SGD gene data
    genes = read_nondubious_genes_dataset()
    
    # Join with RNA-seq TSSes
    tss_comparison = genes.join(combined_rna_TSSes)[[reference_tss_column, 'rna_TSS']]
    #tss_comparison = tss_comparison.dropna()  # Remove genes without RNA calls
    
    # Calculate differences
    tss_comparison['tss_difference'] = tss_comparison['rna_TSS']-\
        tss_comparison[reference_tss_column]
    
    # Negate crick strand, such that differences are 5' oriented
    crick_selection = genes.strand == '-'
    tss_comparison[crick_selection] = -tss_comparison[crick_selection]
    
    # Create histogram
    plt.figure(figsize=(4, 2))
    plt.hist(tss_comparison['tss_difference'], bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel(f'Called RNA-seq TSS - Park TSS, bp (strand corrected)')
    plt.ylabel('Count')
    plt.title(f'Change in TSS calls using stranded RNA-seq', fontweight='demi',
             fontsize=12)
    plt.xlim(-1000, 1000)
    
    # Add summary statistics
    median_diff = tss_comparison['tss_difference'].median()
    mean_diff = tss_comparison['tss_difference'].mean()
    
    print(f"Total genes with RNA-seq TSS calls: {len(tss_comparison)}")
    print(f"Median difference: {median_diff:.1f}bp")
    print(f"Mean difference: {mean_diff:.1f}bp")
    print(f"Std deviation: {tss_comparison['tss_difference'].std():.1f}bp")
    
    return tss_comparison
