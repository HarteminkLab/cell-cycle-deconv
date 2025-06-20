import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class TranscriptBoundaryCaller:
	"""
	Standalone class for detecting transcript boundaries from pileup data.
	
	Uses a two-threshold approach:
	1. Primary threshold identifies transcript cores
	2. Extension threshold extends to natural boundaries
	
	Now supports genomic coordinate input/output.
	"""
	
	def __init__(self, 
				 min_length: int = 100,
				 max_gap: int = 200,
				 refine_boundaries: bool = True,
				 refinement_cutoff: float = 0.1,
				 max_avg_diff: Optional[float] = None,
				 max_fold_diff: Optional[float] = 2.0,
				 output_directory: str = None):
		"""
		Initialize the transcript boundary caller.
		
		Parameters:
		-----------
		min_length : int
			Minimum transcript length in bp (default: 100)
		max_gap : int
			Maximum gap to merge nearby regions in bp (default: 50)
		refine_boundaries : bool
			Whether to refine boundaries using adaptive cutoffs (default: True)
		refinement_cutoff : float
			Proportion of transcript mean for boundary refinement (default: 0.1)
		max_avg_diff : float, optional
			Maximum absolute difference in average values to allow merging
			If None, no average difference constraint is applied
		max_fold_diff : float, optional
			Maximum fold difference in average values to allow merging (default: 2.0)
			Set to None to disable fold difference constraint
			Example: 2.0 means regions with 2x difference won't be merged
		"""
		self.output_directory = output_directory
		self.min_length = min_length
		self.max_gap = max_gap
		self.refine_boundaries = refine_boundaries
		self.refinement_cutoff = refinement_cutoff
		self.max_avg_diff = max_avg_diff
		self.max_fold_diff = max_fold_diff

	def set_chrom_span(self, chrom, span):
		"""
		Set the chromosome, genomic span, and replicate, and load the corresponding data.
		"""

		from src.rna_seq_intermediates import RNASeqIntermediateManager
		from src.pileup_helpers import smooth_rna_curve

		self.chromosome = chrom
		self.span = span
		self.bp_positions = range(span[0], span[1])
		
		# Initialize manager and load data
		manager = RNASeqIntermediateManager(
			output_directory=self.output_directory, 
			chromosome=chrom
		)
		watson_r1, crick_r1, watson_r2, crick_r2 = manager.load_both_replicates_pileups()
		watson_r1 = watson_r1[self.bp_positions]
		watson_r2 = watson_r2[self.bp_positions]
		crick_r1 = crick_r1[self.bp_positions]
		crick_r2 = crick_r2[self.bp_positions]

		# Combine the replicates and take the mean to compute transcripts
		# on the entire experiment
		watson_data = (watson_r1.mean(0)+watson_r2.mean(0))/2.
		crick_data = (crick_r1.mean(0)+crick_r2.mean(0))/2.

		# Log transform for the boundary calling
		watson_data = np.log2(watson_data+1)
		crick_data = np.log2(crick_data+1)

		smoothed_mean_watson = smooth_rna_curve(watson_data)
		smoothed_mean_crick = smooth_rna_curve(crick_data)

		self.watson_data_raw = watson_data
		self.crick_data_raw = crick_data

		# Select the genomic span
		self.watson_data = smoothed_mean_watson
		self.crick_data = smoothed_mean_crick

	def call_boundaries(self, min_threshold=0.1, 
			min_extension_threshold=0.05):

		span = self.span

		# Call the watson and crick transcripts from the input data
		self.called_watson_transcripts = self._call_boundaries(self.watson_data, min_threshold,
								 min_extension_threshold,
                                 start_pos=span[0], end_pos=span[1]-1)
		self.called_crick_transcripts = self._call_boundaries(self.crick_data, min_threshold,
								 min_extension_threshold, start_pos=span[0], 
								 end_pos=span[1]-1)

	
	def _call_boundaries(self, 
					   pileup_vector: np.ndarray,
					   primary_threshold: float,
					   extension_threshold: float,
					   start_pos: Optional[int] = None,
					   end_pos: Optional[int] = None) -> pd.DataFrame:
		"""
		Detect transcript boundaries from pileup data.
		
		Parameters:
		-----------
		pileup_vector : np.ndarray
			1D array of log2-transformed and smoothed pileup values
		primary_threshold : float
			Threshold for identifying transcript cores
		extension_threshold : float
			Threshold for extending transcript boundaries
		start_pos : int, optional
			Genomic start position (1-based) corresponding to first element of pileup_vector
			If None, uses 1-based indexing starting from 1
		end_pos : int, optional
			Genomic end position (1-based) corresponding to last element of pileup_vector
			If provided, used for validation against start_pos and vector length
		chromosome : str, optional
			Chromosome/contig name to include in output
			
		Returns:
		--------
		pd.DataFrame
			Results with columns: start, end, average_value, [chromosome]
			Coordinates are in genomic space (1-based)
		"""
		
		# Validate inputs and set up coordinate system
		vector_length = len(pileup_vector)
		
		if start_pos is None:
			start_pos = 1  # Default to 1-based genomic coordinates

		if end_pos is not None:
			expected_length = end_pos - start_pos + 1
			if expected_length != vector_length:
				raise ValueError(
					f"Span length mismatch: end_pos - start_pos + 1 = {expected_length}, "
					f"but pileup_vector length = {vector_length}"
				)
		else:
			end_pos = start_pos + vector_length - 1
		
		# Store coordinate conversion info
		self._genomic_start = start_pos
		self._genomic_offset = start_pos  # For converting array indices to genomic positions
		
		# Step 1: Find core regions above primary threshold
		core_mask = pileup_vector >= primary_threshold
		core_regions = self._get_regions_from_mask_optimized(core_mask)
		
		# Step 2: Extend regions using extension threshold
		extension_mask = pileup_vector >= extension_threshold
		extended_regions = self._extend_regions_vectorized(core_regions, extension_mask)
		
		# Step 3: Merge overlapping or nearby regions
		merged_regions = self._merge_nearby_regions_optimized(extended_regions, self.max_gap, pileup_vector)
		
		# Step 4: Filter by minimum length
		length_filtered_regions = self._filter_by_length_vectorized(merged_regions, self.min_length)
		
		# Step 5: Refine boundaries with adaptive cutoffs (optional)
		if self.refine_boundaries:
			final_regions = self._refine_boundaries(length_filtered_regions, pileup_vector, self.refinement_cutoff)
		else:
			final_regions = length_filtered_regions
		
		# Step 6: Convert to genomic coordinates and create DataFrame
		results = []
		for start_idx, end_idx in final_regions:
			# Convert array indices to genomic coordinates
			genomic_start = self._array_to_genomic(start_idx)
			genomic_end = self._array_to_genomic(end_idx)
			
			# Calculate average value
			region_values = pileup_vector[start_idx:end_idx+1]
			avg_value = np.mean(region_values)
			
			result = {
				'start': genomic_start,
				'end': genomic_end,
				'average_value': avg_value
			}
				
			results.append(result)
			
		return pd.DataFrame(results)
	
	def _array_to_genomic(self, array_index: int) -> int:
		"""Convert 0-based array index to 1-based genomic coordinate."""
		return self._genomic_offset + array_index
	
	def _genomic_to_array(self, genomic_pos: int) -> int:
		"""Convert 1-based genomic coordinate to 0-based array index."""
		return genomic_pos - self._genomic_offset
	
	def _refine_boundaries(self, regions, pileup_vector, cutoff_proportion):
		"""
		Refine transcript boundaries by finding where signal drops to a proportion of transcript mean.
		
		Parameters:
		-----------
		regions : list of tuples
			Initial (start, end) regions to refine (in array indices)
		pileup_vector : np.ndarray
			The pileup data
		cutoff_proportion : float
			Proportion of transcript mean to use as cutoff (e.g., 0.1 = 10% of mean)
			
		Returns:
		--------
		list of tuples
			Refined (start, end) regions (in array indices)
		"""
		if not regions:
			return []
		
		refined_regions = []
		vector_length = len(pileup_vector)
		
		for start, end in regions:
			# Calculate adaptive cutoff based on transcript mean
			transcript_values = pileup_vector[start:end+1]
			transcript_mean = np.mean(transcript_values)
			adaptive_cutoff = transcript_mean * cutoff_proportion
			
			# Refine start boundary (scan right from start to find where signal rises above cutoff)
			refined_start = start
			for pos in range(start, min(end + 1, vector_length)):
				if pileup_vector[pos] >= adaptive_cutoff:
					refined_start = pos
					break
			
			# Refine end boundary (scan left from end to find where signal falls below cutoff)
			refined_end = end
			for pos in range(end, max(refined_start - 1, -1), -1):
				if pileup_vector[pos] >= adaptive_cutoff:
					refined_end = pos
					break
			
			# Only keep if refined region is still valid
			if refined_end >= refined_start and (refined_end - refined_start + 1) >= self.min_length:
				refined_regions.append((refined_start, refined_end))
		
		return refined_regions
	
	# =====================================================================
	# INTERNAL OPTIMIZATION METHODS
	# =====================================================================
	
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
	
	def _merge_nearby_regions_optimized(self, regions, max_gap, pileup_vector):
		"""
		Optimized merging using numpy operations with average value constraints.
		
		Parameters:
		-----------
		regions : list of tuples
			List of (start, end) regions to merge
		max_gap : int
			Maximum gap distance to allow merging
		pileup_vector : np.ndarray
			Pileup data for calculating average values
			
		Returns:
		--------
		list of tuples
			Merged regions
		"""
		if not regions:
			return []
		
		if len(regions) == 1:
			return regions
		
		# Convert to numpy array for vectorized operations
		regions_array = np.array(sorted(regions))
		starts = regions_array[:, 0]
		ends = regions_array[:, 1]
		
		# Calculate average values for each region
		avg_values = np.array([
			np.mean(pileup_vector[start:end+1]) 
			for start, end in regions_array
		])
		
		# Vectorized gap calculation
		gaps = starts[1:] - ends[:-1] - 1
		
		# Find regions that should be merged based on gap constraint
		gap_merge_mask = gaps <= max_gap
		
		# Apply average value constraints if specified
		if self.max_avg_diff is not None or self.max_fold_diff is not None:
			avg_merge_mask = np.ones_like(gap_merge_mask, dtype=bool)
			
			for i in range(len(gap_merge_mask)):
				current_avg = avg_values[i]
				next_avg = avg_values[i + 1]
				
				# Check absolute difference constraint
				if self.max_avg_diff is not None:
					abs_diff = abs(current_avg - next_avg)
					if abs_diff > self.max_avg_diff:
						avg_merge_mask[i] = False
						continue
				
				# Check fold difference constraint
				if self.max_fold_diff is not None:
					# Avoid division by zero and handle negative values
					if current_avg > 0 and next_avg > 0:
						fold_diff = max(current_avg / next_avg, next_avg / current_avg)
						if fold_diff > self.max_fold_diff:
							avg_merge_mask[i] = False
							continue
					elif current_avg <= 0 and next_avg <= 0:
						# Both are zero or negative, allow merging
						pass
					else:
						# One is positive, one is zero/negative - don't merge
						avg_merge_mask[i] = False
						continue
			
			# Combine gap and average constraints
			merge_mask = gap_merge_mask & avg_merge_mask
		else:
			merge_mask = gap_merge_mask
		
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

	def plot_called_transcripts(self):

		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=(13, 4))

		span = self.span

		x_positions = np.arange(span[0], span[1])

		plt.fill_between(x_positions, self.watson_data, 0, color=plt.cm.Blues(0.5))
		plt.fill_between(x_positions, -self.crick_data, 0, color=plt.cm.Reds(0.5))

		def plot_called_row(row, flip):
			y = -20 if flip else 20
			mean_y = -row.average_value if flip else row.average_value
			plt.plot([row.start, row.end], 
				[mean_y, mean_y], color='black', ls='dotted', lw=0.75)
			plt.fill_between([row.start, row.end],  
				[y, y], 0, lw=0,
					 color='#eee', zorder=0, alpha=0.45)
			plt.plot([row.start, row.start], [0, y], c='black', lw=0.25)
			plt.plot([row.end, row.end], [0, y], c='black', lw=0.25)
				
		for i, row in self.called_watson_transcripts.iterrows():
			plot_called_row(row, False)
			
		for i, row in self.called_watson_transcripts.iterrows():
			plot_called_row(row, True)
			
			plt.fill_between([row.start, row.end],  
				[-20, -20], 0, lw=0,
					 color='#ddd', alpha=0.45, zorder=0)
			
		plt.ylim(-3, 3)
		plt.xlim(span[0], span[1])
		plt.suptitle("Called transcript boundaries", fontweight='demi')

		return fig
