import numpy as np
import pandas as pd
from src.config import load_default_chrom_configs
from src.utils import print_fl

class ChromatinMetricsProcessor:
	"""
	Process chromatin data to compute metrics for both raw and deconvolved data.
	
	This class coordinates between ChromatinModel (for raw data) and 
	GenomeDeconvolutionAnalysis (for deconvolved data) to compute consistent
	metrics across all transcripts.
	"""
	
	def __init__(self, output_dir: str):
		"""
		Initialize the ChromatinMetricsProcessor.
		
		Parameters
		----------
		output_dir : str
			Directory where output CSV files will be saved
		"""
		self.output_dir = output_dir

		self.config1, self.config2 = load_default_chrom_configs()
		
		# Data sources (to be initialized in setup)
		from src.transcripts_dataset import load_transcripts_sets
		self.all_transcripts_set = load_transcripts_sets(output_dir=output_dir,
			combined=True)
		self.deconv_analyzer = None
		
		# Processing parameters
		self.fragment_lengths = {
			'small': (0, 100),
			'nucleosome': (130, 200),
			'all': (0, 260)
		}
		self.metric_types = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']

	def setup_data_loaders(self):
		"""
		Initialize the data loading classes.
		
		Sets up:
		- self.chromatin_model: ChromatinModel instance for raw data
		- self.deconv_analyzer: GenomeDeconvolutionAnalysis instance for deconvolved data
		"""
		from src.raw_chromatin_loader import RawChromatinDataLoader
		from src.deconvolved_chromatin_loader import DeconvolvedChromatinDataLoader

		self.deconvolved_chromatin_loader = DeconvolvedChromatinDataLoader(self.output_dir)
		self.raw_replicate1_chromatin_loader = RawChromatinDataLoader(replicate=1)
		self.raw_replicate2_chromatin_loader = RawChromatinDataLoader(replicate=2)

	def load_mnase_span(self, chrom, mnase_span):
		self.deconvolved_chromatin_loader.load_mnase_span(chrom, mnase_span)
		self.raw_replicate1_chromatin_loader.load_mnase_span(chrom, mnase_span)
		self.raw_replicate2_chromatin_loader.load_mnase_span(chrom, mnase_span)

	def subset_loaded_data(self, genomic_region, fragment_lengths):

		self.deconvolved_subset_data = self.deconvolved_chromatin_loader.subset_loaded_data(genomic_region=genomic_region,
		fragment_lengths=fragment_lengths)
		self.raw_replicate1_subset_data = self.raw_replicate1_chromatin_loader.subset_loaded_data(genomic_region=genomic_region,
		fragment_lengths=fragment_lengths)
		self.raw_replicate2_subset_data = self.raw_replicate2_chromatin_loader.subset_loaded_data(genomic_region=genomic_region,
		fragment_lengths=fragment_lengths)


	def compute_gene_metrics(self, gene, data_loader):
		"""
		Compute chromatin metrics for a single gene using the specified data loader.
		
		Parameters
		----------
		gene : pd.Series
			Gene information including chr, TSS, promoter/gene_body boundaries
		data_loader : ChromatinDataLoader
			Data loader instance (deconvolved, raw_rep1, or raw_rep2)
			
		Returns
		-------
		dict
			Dictionary with metric names as keys and computed values
		"""

		# Extract gene boundaries
		chrom = gene['chr']
		promoter_span = gene['promoter_start'], gene['promoter_end']
		gene_body_span = gene['gene_body_start'], gene['gene_body_end']

		promoter_start, promoter_end = min(promoter_span), max(promoter_span)
		gene_body_start, gene_body_end = min(gene_body_span), max(gene_body_span)
		
		# Determine span to load (include some padding to ensure we get all needed data)
		span_start = min(promoter_start, gene_body_start) - 2000
		span_end = max(promoter_end, gene_body_end) + 2000
		
		# Load the data for this gene's region
		data_loader.load_mnase_span(chrom, (span_start, span_end))
		
		# Initialize metrics dictionary
		metrics = {}

		# 1. Promoter Occupancy (all fragment lengths)
		promoter_data = data_loader.subset_loaded_data(
			genomic_region=(promoter_start, promoter_end),
			fragment_lengths=self.fragment_lengths['small']
		)

		# Sum across all timepoints, fragment lengths, and positions
		promoter_occupancy = np.mean(promoter_data, axis=1).mean(axis=1)
		metrics['promoter_occupancy'] = promoter_occupancy
		
		# 2. Gene Body Nucleosome Entropy (nucleosomal fragments only)
		gene_body_nuc_data = data_loader.subset_loaded_data(
			genomic_region=(gene_body_start, gene_body_end),
			fragment_lengths=self.fragment_lengths['nucleosome']
		)
		# Calculate entropy for each timepoint
		nucleosome_entropy = self._compute_entropy(gene_body_nuc_data)
		metrics['nucleosome_entropy'] = nucleosome_entropy
		
		# 3. Gene Body Nucleosome Occupancy (nucleosomal fragments only)
		# Sum across all timepoints, fragment lengths, and positions
		nucleosome_occupancy = np.mean(gene_body_nuc_data, axis=1).mean(axis=1)
		metrics['nucleosome_occupancy'] = nucleosome_occupancy
		
		return metrics

	def _compute_entropy(self, data):
		"""
		Compute entropy of chromatin data.
		
		Parameters
		----------
		data : np.ndarray
			Chromatin data with shape (timepoints, fragment_lengths, positions)
			
		Returns
		-------
		float
			Mean entropy across timepoints
		"""
		# Based on the code snippet you provided earlier
		from src.helpers import calc_entropy
		
		# Reshape to (timepoints, flattened_fragments_x_positions)
		reshaped = data.reshape(data.shape[0], -1)
		
		# Calculate entropy for each timepoint
		entropies = []
		for t in range(reshaped.shape[0]):
			# Add small value to avoid log(0)
			entropy = calc_entropy(reshaped[t] + 0.001)
			entropies.append(entropy)
		
		# Return mean entropy across timepoints
		return np.array(entropies)

	def compute_metrics_for_data(self, data_loader, debug=False,
		compute_raw_ptrs=False):
		"""
		Compute chromatin metrics for all transcripts using a single data loader.
		
		Parameters
		----------
		data_loader : ChromatinDataLoader
			One of: deconvolved_chromatin_loader, raw_replicate1_chromatin_loader, 
			or raw_replicate2_chromatin_loader
		
		Returns
		-------
		dict
			Dictionary with metric names as keys and DataFrames as values:
			{
				'promoter_occupancy': DataFrame,
				'nucleosome_entropy': DataFrame, 
				'nucleosome_occupancy': DataFrame
			}
			Each DataFrame has orf_names as rows and timepoint indices as columns.
		"""
		from src.timer import Timer
		
		# Set default transcript set
		transcript_set = self.all_transcripts_set
		
		timer = Timer()
		print_fl(f"Starting metric computation for {len(transcript_set)} transcripts...")
		
		# Determine the number of timepoints by testing with a sample gene
		n_timepoints = None
		for orf_name, gene in transcript_set.iterrows():
			try:
				sample_metrics = self.compute_gene_metrics(gene, data_loader)
				n_timepoints = len(sample_metrics['promoter_occupancy'])
				print_fl(f"Detected {n_timepoints} timepoints from sample gene {orf_name}")
				break
			except Exception as e:
				print_fl(f"Warning: Could not use {orf_name} as sample gene: {str(e)}")
				continue
		
		if n_timepoints is None:
			raise RuntimeError("Could not determine number of timepoints from any sample gene")
		
		# Initialize empty dataframes for each metric
		orf_names = transcript_set.index.tolist()
		timepoint_columns = list(range(n_timepoints))
		
		results = {}
		for metric_name in self.metric_types:
			results[metric_name] = pd.DataFrame(
				index=orf_names, 
				columns=timepoint_columns,
				dtype=float
			)
		
		# Process each gene
		successful_genes = 0
		failed_genes = 0
		
		for i, (orf_name, gene) in enumerate(transcript_set.iterrows()):
			try:
				# Compute metrics for this gene
				metrics = self.compute_gene_metrics(gene, data_loader)
				
				# Add each metric to its corresponding dataframe
				for metric_name in self.metric_types:
					results[metric_name].loc[orf_name] = metrics[metric_name]
				
				successful_genes += 1
				
			except Exception as e:
				# Skip this gene, leave NaN values
				print_fl(f"Warning: Failed to compute metrics for {orf_name}: {str(e)}")
				failed_genes += 1
			
			# Progress tracking every 200 genes
			if (i + 1) % 200 == 0:
				time_str = timer.get_time()
				print_fl(f"Processed {i + 1}/{len(transcript_set)} genes. "
					  f"Time elapsed: {time_str}. "
					  f"Success: {successful_genes}, Failed: {failed_genes}")

				if debug:
					break

		# Compute the peak to trough values
		peak_to_trough_results = self.compute_peak_to_trough_values(results,
			compute_raw_ptrs=compute_raw_ptrs)

		# Final summary
		time_str = timer.get_time()
		print_fl(f"Completed! Processed {len(transcript_set)} genes in {time_str}. "
			  f"Success: {successful_genes}, Failed: {failed_genes}")
		
		return results, peak_to_trough_results


	def compute_peak_to_trough_values(self, chromatin_metrics_dic, compute_raw_ptrs=False):
		"""Compute peak to trough values for a given run's dictionary of
		metric values"""
		from src.peak_to_trough import compute_ptr_tb, compute_quantile_ptr
		config = self.raw_replicate1_chromatin_loader.config

		peak_to_trough_values = {}
		ptr_lo, ptr_hi = 0.1, 0.9

		for key in chromatin_metrics_dic.keys():
			metric_values = chromatin_metrics_dic[key]

			# Deconvolved ptr values (mother and daughter branches)
			if not compute_raw_ptrs:
				ptr_values = np.apply_along_axis(lambda row: 
					compute_ptr_tb(config, row, lo=ptr_lo, hi=ptr_hi), axis=1, arr=metric_values.values)

			# Raw data ptrs values (no config indexing)
			else:
				ptr_values = np.apply_along_axis(lambda row: 
					compute_quantile_ptr(row, ptr_lo, ptr_hi), axis=1, arr=metric_values.values)

			ptrs_df = pd.DataFrame(ptr_values, columns=[key+"_ptr"],
				index=metric_values.index)
			peak_to_trough_values[key] = ptrs_df

		return peak_to_trough_values

	def compute_chromatin_metrics_all_data(self, debug=False):
		"""Compute chromatin metrics and PTR values for each of the
		data loaders"""

		(self.deconvolved_chromatin_metrics,
		 self.deconvolved_chromatin_ptrs) = \
			self.compute_metrics_for_data(self.deconvolved_chromatin_loader, 
			debug=debug, compute_raw_ptrs=False)

		(self.raw_rep1_metrics,
		 self.raw_rep1_ptrs) = self.compute_metrics_for_data(
			self.raw_replicate1_chromatin_loader, 
			debug=debug, compute_raw_ptrs=True)

		(self.raw_rep2_metrics,
		 self.raw_rep2_ptrs) = self.compute_metrics_for_data(
			self.raw_replicate2_chromatin_loader, 
			debug=debug, compute_raw_ptrs=True)

	def save_all_results_disk(self, save_directory):

		# Save all metrics
		self.save_results_to_csv(self.deconvolved_chromatin_metrics, save_directory,
			"deconvolved", "metrics")
		self.save_results_to_csv(self.raw_rep1_metrics, save_directory,
			"raw_rep1", "metrics")  
		self.save_results_to_csv(self.raw_rep2_metrics, save_directory,
			"raw_rep2", "metrics")

		# Save all PTRs  
		self.save_results_to_csv(self.deconvolved_chromatin_ptrs, save_directory,
			"deconvolved", "ptrs")
		self.save_results_to_csv(self.raw_rep1_ptrs, save_directory,
			"raw_rep1", "ptrs")
		self.save_results_to_csv(self.raw_rep2_ptrs, save_directory,
			"raw_rep2", "ptrs")


	def save_results_to_csv(self, results_dict, save_dir, data_source_name, data_type="metrics"):
	    """
	    Save metrics or PTR DataFrames to CSV files.
	    
	    Parameters
	    ----------
	    results_dict : dict
	        Dictionary with metric names as keys and DataFrames as values.
	        Can be metrics dict or PTR dict.
	    save_dir : str
	        Directory where CSV files will be saved
	    data_source_name : str
	        Identifier for the data source (e.g., 'deconvolved', 'raw_rep1', 'raw_rep2')
	        Used as prefix in filenames
	    data_type : str, optional
	        Type of data being saved ('metrics' or 'ptrs'), used in filename
	    """
	    import os
	    from pathlib import Path
	    
	    # Create save directory if it doesn't exist
	    save_path = Path(save_dir)
	    save_path.mkdir(parents=True, exist_ok=True)
	    
	    print_fl(f"Saving {data_type} to {save_dir} with prefix '{data_source_name}'...")
	    
	    saved_files = []
	    for metric_name, dataframe in results_dict.items():
	        # Create filename: data_source_datatype_metric_name.csv
	        if data_type == "metrics":
	            filename = f"{data_source_name}_{metric_name}.csv"
	        else:  # PTRs
	            filename = f"{data_source_name}_{data_type}_{metric_name}.csv"
	        filepath = save_path / filename
	        
	        # Save DataFrame to CSV
	        dataframe.to_csv(filepath, index=True)
	        saved_files.append(str(filepath))
	        
	        print_fl(f"  Saved {metric_name} {data_type}: {filename} ({dataframe.shape[0]} genes, {dataframe.shape[1]} columns)")
	    
	    print_fl(f"Successfully saved {len(saved_files)} {data_type} files.")
	    return saved_files


