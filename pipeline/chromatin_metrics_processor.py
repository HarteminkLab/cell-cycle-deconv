import numpy as np
import pandas as pd
from src.config import load_default_chrom_configs
from src.utils import print_fl, mkdir_safe
from src.figure_configs import save_figure_for_paper
from matplotlib import pyplot as plt
from src.global_config import GlobalConstants

# Formatting map for ptr plots for each metric
plot_formatting_map = {
	'promoter_occupancy': {
		'bw': 0.007,
		'ptr_lims': (0.95, 2.0),
		'cmap': 'Oranges',
		'name': "Promoter occupancy"
	},
	'nucleosome_entropy': {
		'bw': 0.007,		
		'ptr_lims': (0.95, 2.0),
		'cmap': 'Purples',
		'name': "Nucleosome entropy"
	},        
	'nucleosome_occupancy': {
		'bw': 0.007,
		'ptr_lims': (0.95, 2.0),
		'cmap': 'Blues',
		'name': "Nucleosome occupancy"
	}
}

fragment_lengths = {
	'small': (0, 100),
	'nucleosome': GlobalConstants.WIDER_NUCLEOSOME_FRAGMENT_LENGTHS,
	'all': (0, 260)
}

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
		self.fragment_lengths = fragment_lengths
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

	def compute_metrics_for_data(self, data_loader, dataset_key, debug=False,
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
			dataset_key, compute_raw_ptrs=compute_raw_ptrs)

		# Final summary
		time_str = timer.get_time()
		print_fl(f"Completed! Processed {len(transcript_set)} genes in {time_str}. "
			  f"Success: {successful_genes}, Failed: {failed_genes}")
		
		return results, peak_to_trough_results

	def normalize_chromatin_metrics(self):
		"""Normalize the raw chromatin metrics for the criteria:

		1. Replicate 1 and replicate 2 have matching per-gene variation 
		   distributions. Addresses comparable peak-to-trough ratio calculations
		   between replicates.
		2. Normalize the entropy distribution to be comparable to the occupancy
		   metrics:
				i.  Shift the distribution to the 2.5 percentile (thus moving the
					effective minimum value to zero)
				ii. Match the mean of the distribution to the average of the
					occupancy measures (for better plotting clarity between the 
					three measures)
		3. The deconvolved metrics remain the same except for the entropy. This
		   entropy score should be comparable to the raw entropy distributions.

		"""
		from src.transformations import normalize_replicate_variance_vectorized

		chromatin_keys = ['promoter_occupancy', 'nucleosome_entropy', 
			'nucleosome_occupancy']

		normalized_raw_rep1_metrics = {}
		normalized_raw_rep2_metrics = {}
		normalized_deconvolved_metrics = {}
		
		target_entropy_mean = np.array([
			self.raw_rep1_metrics['promoter_occupancy'].values.mean(),
			self.raw_rep2_metrics['promoter_occupancy'].values.mean(),
			self.raw_rep1_metrics['nucleosome_occupancy'].values.mean(),
			self.raw_rep2_metrics['nucleosome_occupancy'].values.mean(),
		]).mean()
		
		for chromatin_key in chromatin_keys:
			metric_rep1 = self.raw_rep1_metrics[chromatin_key]
			metric_rep2 = self.raw_rep2_metrics[chromatin_key]
			metric_deconvolved = self.deconvolved_chromatin_metrics[\
				chromatin_key].copy()
			
			if chromatin_key == 'nucleosome_entropy':            

				def _normalize_entropy_shift_mean(entropy_values, target_mean,
					shift_quantile=0.025):
				
					# Shift the distribution closer to 0 (for better peak to
					# trough calculation)
					entropy_values = entropy_values - np.quantile(entropy_values, q=shift_quantile)
					entropy_values = entropy_values / entropy_values.mean() * target_mean
					entropy_values[entropy_values < 0] = 0
					return entropy_values
				
				# Shift and scale the entropy values for all three data sources
				metric_rep1 = _normalize_entropy_shift_mean(metric_rep1, target_entropy_mean)
				metric_rep2 = _normalize_entropy_shift_mean(metric_rep2, target_entropy_mean)
				metric_deconvolved = _normalize_entropy_shift_mean(metric_deconvolved, target_entropy_mean)

			scaled_metric_rep1, scaled_metric_rep2 = normalize_replicate_variance_vectorized(
				metric_rep1, metric_rep2, None)
			
			normalized_raw_rep1_metrics[chromatin_key] = scaled_metric_rep1
			normalized_raw_rep2_metrics[chromatin_key] = scaled_metric_rep2
			normalized_deconvolved_metrics[chromatin_key] = metric_deconvolved

		self.normalized_raw_rep1_metrics = normalized_raw_rep1_metrics
		self.normalized_raw_rep2_metrics = normalized_raw_rep2_metrics
		self.normalized_deconvolved_metrics = normalized_deconvolved_metrics


	def plot_raw_scaled_distributions(self, chromatin_key, save_directory):
		metric_rep1 = self.raw_rep1_metrics[chromatin_key]
		metric_rep2 = self.raw_rep2_metrics[chromatin_key]
		deconvolved = self.deconvolved_chromatin_metrics[chromatin_key]
		scaled_rep1 = self.normalized_raw_rep1_metrics[chromatin_key]
		scaled_rep2 = self.normalized_raw_rep2_metrics[chromatin_key]
		scaled_deconvolved = self.normalized_deconvolved_metrics[chromatin_key]

		# All values distribution
		plt.figure(figsize=(9, 4))
		
		nrow, ncol = 2, 3

		def _plot_metrics_group(metric_rep1, metric_rep2, deconvolved, subplot_indices,
							  suffix):
			
			xlims = 0, 10
				
			# Adjust range of values to plot
			def plot_histogram(values, cutoffs=None, data_key=None):
				colors = {
					'raw_rep1': plt.cm.tab10(0),
					'raw_rep2': plt.cm.tab10(1),
					'deconvolved': plt.cm.tab10(2),
				}
				cutoffs = xlims if cutoffs is None else cutoffs
				values = values[(values > cutoffs[0]) & (values < cutoffs[1])]
				plt.hist(values, bins=40, color=colors[data_key], 
					alpha=0.25, density=True, label=data_key)

			plt.subplot(nrow, ncol, subplot_indices[0])
			plot_histogram(metric_rep1.values.flatten(), data_key='raw_rep1')
			plot_histogram(metric_rep2.values.flatten(), data_key='raw_rep2')
			# plot_histogram(deconvolved.values.flatten(), data_key='deconvolved')
			plt.title(f"All values {suffix}")
			plt.xlim(*xlims)
			plt.legend()

			# Per gene mean
			plt.subplot(nrow, ncol, subplot_indices[1])
			plot_histogram(metric_rep1.values.mean(1), data_key='raw_rep1')
			plot_histogram(metric_rep2.values.mean(1), data_key='raw_rep2')
			# plot_histogram(deconvolved.values.mean(1), data_key='deconvolved')

			plt.title(f"$\\mu$ per gene {suffix}")
			plt.xlim(*xlims)

			# Per gene std
			plt.subplot(nrow, ncol, subplot_indices[2])

			xlims = 0, 1
			plot_histogram(metric_rep1.values.std(1), xlims, 'raw_rep1')
			plot_histogram(metric_rep2.values.std(1), xlims, 'raw_rep2')
			# plot_histogram(deconvolved.values.std(1), xlims, 'deconvolved')
			plt.title(f"$\\sigma$ per gene {suffix}")
		
		_plot_metrics_group(metric_rep1, metric_rep2, deconvolved, [1, 2, 3], "(raw)")
		_plot_metrics_group(scaled_rep1, scaled_rep2, scaled_deconvolved, [4, 5, 6], "(normalized)")

		chromatin_title = f"{chromatin_key.replace('_', ' ')}"
		chromatin_title = chromatin_title[0:1].upper() + chromatin_title[1:]
		
		plt.suptitle(chromatin_title, fontweight='demi', fontsize=18)
		plt.tight_layout()

		if save_directory is not None:
			save_figure_for_paper(f"{save_directory}/{chromatin_key}.png")

	def plot_distribution_transformation(self, save_directory):
		"""Plot the effect of the normalization scheme on each of the chromatin measures"""
		self.plot_raw_scaled_distributions('promoter_occupancy', save_directory)
		self.plot_raw_scaled_distributions('nucleosome_entropy', save_directory)
		self.plot_raw_scaled_distributions('nucleosome_occupancy', save_directory)
		
	def compute_and_assign_normalized_ptr_values(self):
		"""Compute the peak to trough ratio values for the noramlized
		   chromatin values. 

		   todo: not yet in the pipeline, attempting to address replicate1 replicate2
		   discrepancy and entropy range scaling
		"""

		self.normalized_ptr_rep1 = self.compute_peak_to_trough_values(
		    self.normalized_raw_rep1_metrics, 'raw_rep1',
		    compute_raw_ptrs=True)
		self.normalized_ptr_rep2 = self.compute_peak_to_trough_values(
		    self.normalized_raw_rep2_metrics, 'raw_rep2',
		    compute_raw_ptrs=True)
		self.normalized_ptr_deconvolved = self.compute_peak_to_trough_values(
		    self.normalized_deconvolved_metrics, 'deconvolved')


	def compute_peak_to_trough_values(self, chromatin_metrics_dic, dataset_key, compute_raw_ptrs=False):
		"""Compute peak to trough values for a given run's dictionary of
		metric values"""
		from src.peak_to_trough import compute_ptr_tb, compute_quantile_ptr
		config = self.raw_replicate1_chromatin_loader.config

		peak_to_trough_values = {}
		ptr_lo, ptr_hi = 0.1, 0.9

		for key in chromatin_metrics_dic.keys():
			orfs_index = chromatin_metrics_dic[key].index
			metric_values_df = chromatin_metrics_dic[key]
			metric_values = metric_values_df.values

			# Deconvolved ptr values (mother and daughter branches)
			if not compute_raw_ptrs:
				ptr_values = np.apply_along_axis(lambda row: 
					compute_ptr_tb(config, row, lo=ptr_lo, hi=ptr_hi), axis=1, 
					arr=metric_values_df)

			# Raw data ptrs values (no config indexing)
			else:

				# Subset the columns such that we only
				# compute the non-recovery G1 timepoints in the PTR calculation
				# Use mu0 to remove recovery g1 timepoints
				skip_minutes = -int(np.round(config.params_dic['mu0']/10)*10)
				skip_index_max = skip_minutes//10 # Assume 1 index per minute

				ptr_values = np.apply_along_axis(lambda row: 
					compute_quantile_ptr(row, ptr_lo, ptr_hi), axis=1, arr=metric_values[:, skip_index_max:])

			ptrs_df = pd.DataFrame(ptr_values, columns=[key+"_ptr"],
				index=orfs_index)
			peak_to_trough_values[key] = ptrs_df

		return peak_to_trough_values

	def compute_chromatin_metrics_all_data(self, debug=False):
		"""Compute chromatin metrics and PTR values for each of the
		data loaders"""

		(self.deconvolved_chromatin_metrics,
		 self.deconvolved_chromatin_ptrs) = \
			self.compute_metrics_for_data(self.deconvolved_chromatin_loader, 
				'deconvolved', debug=debug, compute_raw_ptrs=False)

		(self.raw_rep1_metrics,
		 self.raw_rep1_ptrs) = self.compute_metrics_for_data(
			self.raw_replicate1_chromatin_loader, 'raw_rep1',
			debug=debug, compute_raw_ptrs=True)

		(self.raw_rep2_metrics,
		 self.raw_rep2_ptrs) = self.compute_metrics_for_data(
			self.raw_replicate2_chromatin_loader, 'raw_rep2',
			debug=debug, compute_raw_ptrs=True)

		# Normalize metrics and compute updated ptr values
		self.normalize_chromatin_metrics()
		self.compute_and_assign_normalized_ptr_values()

	def plot_raw_to_deconvolved_ptr_change(self, metric_name, normalized_metrics=True,
		ptr_lims=None):

		if normalized_metrics:
			raw_rep1_ptrs = self.normalized_ptr_rep1[metric_name]
			raw_rep2_ptrs = self.normalized_ptr_rep2[metric_name]
			deconv_ptrs = self.normalized_ptr_deconvolved[metric_name]
		else:
			raw_rep1_ptrs = self.raw_rep1_ptrs[metric_name]
			raw_rep2_ptrs = self.raw_rep2_ptrs[metric_name]
			deconv_ptrs = self.deconvolved_chromatin_ptrs[metric_name]
		
		bw, mapped_ptr_lims, cmap = plot_formatting_map[metric_name]['bw'],\
			plot_formatting_map[metric_name]['ptr_lims'], \
			plot_formatting_map[metric_name]['cmap']

		if ptr_lims is None:
			ptr_lims = mapped_ptr_lims

		# Subset to the genic transcripts
		genic_transcripts = self.all_transcripts_set[
			self.all_transcripts_set.transcript_class == 'genic'].index

		fig = plt.figure(figsize=(11., 4.25))

		plt.subplot(1, 3, 1)
		plot_ptr_change(raw_rep1_ptrs.loc[genic_transcripts], 
				raw_rep2_ptrs.loc[genic_transcripts], metric_name, self.selected_genes, 
				bw=bw, cmap=cmap, ptr_lims=ptr_lims)
		plt.title('Rep 1 vs Rep 2')
		plt.xlabel("Replicate 1 PTR")
		plt.ylabel("Replicate 2 PTR")

		plt.subplot(1, 3, 2)
		plot_ptr_change(raw_rep1_ptrs.loc[genic_transcripts], 
				deconv_ptrs.loc[genic_transcripts], metric_name, self.selected_genes, 
				bw=bw, cmap=cmap, ptr_lims=ptr_lims)
		plt.title('Rep 1 vs Deconvolved')
		plt.ylabel("Deconvolved PTR")
		plt.xlabel("Replicate 1 PTR")

		plt.subplot(1, 3, 3)
		plot_ptr_change(raw_rep2_ptrs.loc[genic_transcripts], 
				deconv_ptrs.loc[genic_transcripts], metric_name, self.selected_genes, 
				bw=bw, cmap=cmap, ptr_lims=ptr_lims)
		plt.title('Rep 2 vs Deconvolved')
		plt.xlabel("Replicate 2 PTR")
		plt.ylabel("Deconvolved PTR")

		suptitle = f"{metric_name.replace('_', ' ')}"
		suptitle = suptitle[0:1].upper() + suptitle[1:]
		plt.suptitle(f"{suptitle} PTR change\nfollowing deconvolution, n={len(raw_rep1_ptrs.loc[genic_transcripts])}", 
					fontweight='demi', fontsize=18)
		plt.tight_layout()

	def plot_combined_ptr_change(self, normalized=True):
		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=(11., 4.))

		genic_transcripts = self.all_transcripts_set[
			self.all_transcripts_set.transcript_class == 'genic'].index

		def plot_combined_raw_metric_ptr(metric_name, normalized=True):

			bw, ptr_lims, cmap = plot_formatting_map[metric_name]['bw'],\
				plot_formatting_map[metric_name]['ptr_lims'], \
				plot_formatting_map[metric_name]['cmap']

			if normalized:
				raw_rep1_ptrs = self.normalized_ptr_rep1[metric_name]
				raw_rep2_ptrs = self.normalized_ptr_rep2[metric_name]
				deconv_ptrs = self.normalized_ptr_deconvolved[metric_name]
			else:
				raw_rep1_ptrs = self.raw_rep1_ptrs[metric_name]
				raw_rep2_ptrs = self.raw_rep2_ptrs[metric_name]
				deconv_ptrs = self.deconvolved_chromatin_ptrs[metric_name]

			raw_combined_rep_ptrs = (raw_rep1_ptrs + raw_rep2_ptrs)/2.

			plot_ptr_change(raw_combined_rep_ptrs.loc[genic_transcripts], 
				deconv_ptrs.loc[genic_transcripts], metric_name, self.selected_genes, 
				bw=bw, cmap=cmap, ptr_lims=ptr_lims)

		plt.subplot(1, 3, 1)
		plot_combined_raw_metric_ptr('promoter_occupancy', normalized=normalized)
		plt.xlabel("Combined raw data PTR")
		plt.ylabel("Deconvolved PTR")
		plt.title("Promoter occupancy")

		plt.subplot(1, 3, 2)
		plot_combined_raw_metric_ptr('nucleosome_entropy', normalized=normalized)
		plt.xlabel("Combined raw data PTR")
		plt.ylabel("Deconvolved PTR")
		plt.title("Nucleosome entropy")

		plt.subplot(1, 3, 3)
		plot_combined_raw_metric_ptr('nucleosome_occupancy', normalized=normalized)
		plt.xlabel("Mean raw data PTR")
		plt.ylabel("Deconvolved PTR")
		plt.title("Nucleosome occupancy")

		plt.suptitle(f"PTR change following deconvolution, n={len(genic_transcripts)}", 
					fontweight='demi', fontsize=18)
		plt.tight_layout()


	def plot_combined_ptr_change_w_expression(self, high_cycling_tx_genes=[]):
		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=(11., 4.))

		genic_transcripts = self.all_transcripts_set[self.all_transcripts_set.transcript_class == 'genic'].index

		def plot_combined_raw_metric_ptr(metric_name):

			bw, ptr_lims, cmap = plot_formatting_map[metric_name]['bw'],\
				plot_formatting_map[metric_name]['ptr_lims'], \
				plot_formatting_map[metric_name]['cmap']

			raw_rep1_ptrs = self.raw_rep1_ptrs[metric_name]
			raw_rep2_ptrs = self.raw_rep2_ptrs[metric_name]
			raw_combined_rep_ptrs = ((raw_rep1_ptrs + raw_rep2_ptrs)/2.).loc[genic_transcripts]
			deconv_ptrs = self.deconvolved_chromatin_ptrs[metric_name].loc[genic_transcripts]

			plt.scatter(raw_combined_rep_ptrs, deconv_ptrs, 
				c='#ddd', s=2)
			plt.scatter(raw_combined_rep_ptrs.loc[high_cycling_tx_genes], deconv_ptrs.loc[high_cycling_tx_genes], 
				color=plt.cm.Greens(0.75), alpha=0.75, s=3, label=f"Cell cycle expression\nn={len(high_cycling_tx_genes)} (95th percentile)")

			plt.plot([-0.5, 5], [-0.5, 5], c='red', lw=0.75, zorder=10, ls='dotted')

			plt.ylim(*ptr_lims)
			plt.xlim(*ptr_lims)
			plt.legend()

		plt.subplot(1, 3, 1)
		plot_combined_raw_metric_ptr('promoter_occupancy')
		plt.xlabel("Combined raw data PTR")
		plt.ylabel("Deconvolved PTR")
		plt.title("Promoter occupancy")

		plt.subplot(1, 3, 2)
		plot_combined_raw_metric_ptr('nucleosome_entropy')
		plt.xlabel("Combined raw data PTR")
		plt.ylabel("Deconvolved PTR")
		plt.title("Nucleosome entropy")

		plt.subplot(1, 3, 3)
		plot_combined_raw_metric_ptr('nucleosome_occupancy')
		plt.xlabel("Mean raw data PTR")
		plt.ylabel("Deconvolved PTR")
		plt.title("Nucleosome occupancy")

		plt.suptitle(f"PTR change following deconvolution, n={len(genic_transcripts)}", 
					fontweight='demi', fontsize=18)
		plt.tight_layout()

	def create_plots(self):

		save_dir = f"{self.output_dir}/chromatin_metrics/figures"
		mkdir_safe(save_dir)

		# Create and save plots for PTR change
		self.plot_raw_to_deconvolved_ptr_change('promoter_occupancy')
		save_figure_for_paper(f"{save_dir}/promoter_occupancy_ptr.png")

		self.plot_raw_to_deconvolved_ptr_change('nucleosome_entropy')
		save_figure_for_paper(f"{save_dir}/nucleosome_entropy_ptr.png")

		self.plot_raw_to_deconvolved_ptr_change('nucleosome_occupancy')
		save_figure_for_paper(f"{save_dir}/nucleosome_occupancy_ptr.png")

		# Create combined chromatin metrics PTR plots
		# replicate 1 and 2 raw data are combined
		self.plot_combined_ptr_change()
		save_figure_for_paper(f"{save_dir}/raw_vs_deconvolved_all_metrics_ptrs.png")

		# The distributions of each metric, with transformation
		self.plot_distribution_transformation(save_dir)


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

		# Save normalized metrics
		self.save_results_to_csv(self.normalized_deconvolved_metrics, save_directory,
			"normalized_deconvolved", "metrics")
		self.save_results_to_csv(self.normalized_raw_rep1_metrics, save_directory,
			"normalized_raw_rep1", "metrics")  
		self.save_results_to_csv(self.normalized_raw_rep2_metrics, save_directory,
			"normalized_raw_rep2", "metrics")

		# Save normalized PTRs  
		self.save_results_to_csv(self.normalized_ptr_deconvolved, save_directory,
			"normalized_deconvolved", "ptrs")
		self.save_results_to_csv(self.normalized_ptr_rep1, save_directory,
			"normalized_raw_rep1", "ptrs")
		self.save_results_to_csv(self.normalized_ptr_rep2, save_directory,
			"normalized_raw_rep2", "ptrs")


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
			dataframe.index.name = 'transcript_name' # Name the index: based on orf_name or transcript name
			dataframe.to_csv(filepath, index=True)
			saved_files.append(str(filepath))
			
			print_fl(f"  Saved {metric_name} {data_type}: {filename} ({dataframe.shape[0]} genes, {dataframe.shape[1]} columns)")
		
		print_fl(f"Successfully saved {len(saved_files)} {data_type} files.")
		return saved_files


	def load_saved_metrics(self, load_dir, data_sources=None, load_ptrs=True):
		"""
		Load previously saved chromatin metrics and PTR data from CSV files.
		
		Parameters
		----------
		load_dir : str
			Directory where the CSV files are stored
		data_sources : list of str, optional
			List of data sources to load. If None, loads all available sources.
			Valid options: ['deconvolved', 'raw_rep1', 'raw_rep2']
		load_ptrs : bool, optional
			Whether to also load PTR data. Default is True.
			
		Returns
		-------
		tuple
			(metrics_dict, ptrs_dict) where:
			- metrics_dict: dict with data_source as keys, each containing metric DataFrames
			- ptrs_dict: dict with data_source as keys, each containing PTR DataFrames
			
		Example
		-------
		# Load all data
		metrics, ptrs = processor.load_saved_metrics('/path/to/saved/data')
		
		# Load only deconvolved data
		metrics, ptrs = processor.load_saved_metrics('/path/to/saved/data', 
													data_sources=['deconvolved'])
		
		# Load only metrics (no PTRs)
		metrics, _ = processor.load_saved_metrics('/path/to/saved/data', load_ptrs=False)
		"""
		import os
		import pandas as pd
		from pathlib import Path
		
		load_path = Path(load_dir)
		if not load_path.exists():
			raise FileNotFoundError(f"Directory {load_dir} does not exist")
		
		# Default data sources if none specified
		if data_sources is None:
			data_sources = ['deconvolved', 'raw_rep1', 'raw_rep2']
		
		# Initialize result dictionaries
		loaded_metrics = {}
		loaded_ptrs = {}
		
		print_fl(f"Loading saved data from {load_dir}...")
		
		for data_source in data_sources:
			print_fl(f"  Loading {data_source} data...")
			
			# Load metrics for this data source
			source_metrics = {}
			for metric_name in self.metric_types:
				# Filename pattern: {data_source}_{metric_name}.csv
				metric_filename = f"{data_source}_{metric_name}.csv"
				metric_filepath = load_path / metric_filename
				
				if metric_filepath.exists():
					df = pd.read_csv(metric_filepath, index_col=0)
					source_metrics[metric_name] = df
					print_fl(f"    Loaded {metric_name}: {df.shape[0]} genes, {df.shape[1]} timepoints")
				else:
					print_fl(f"    Warning: {metric_filename} not found, skipping...")
			
			if source_metrics:
				loaded_metrics[data_source] = source_metrics
			
			# Load PTRs for this data source if requested
			if load_ptrs:
				source_ptrs = {}
				for metric_name in self.metric_types:
					# Filename pattern: {data_source}_ptrs_{metric_name}.csv
					ptr_filename = f"{data_source}_ptrs_{metric_name}.csv"
					ptr_filepath = load_path / ptr_filename
					
					if ptr_filepath.exists():
						df = pd.read_csv(ptr_filepath, index_col=0)
						source_ptrs[metric_name] = df
						print_fl(f"    Loaded {metric_name} PTRs: {df.shape[0]} genes")
					else:
						print_fl(f"    Warning: {ptr_filename} not found, skipping...")
				
				if source_ptrs:
					loaded_ptrs[data_source] = source_ptrs
		
		# Summary
		metrics_loaded = sum(len(source_data) for source_data in loaded_metrics.values())
		ptrs_loaded = sum(len(source_data) for source_data in loaded_ptrs.values()) if load_ptrs else 0
		
		print_fl(f"Successfully loaded {metrics_loaded} metric datasets and {ptrs_loaded} PTR datasets")
		
		return loaded_metrics, loaded_ptrs


	def load_and_assign_saved_metrics(self, load_dir=None, data_sources=None):
		"""
		Load saved metrics and assign them to the class attributes.
		
		This is a convenience method that loads the data and assigns it to the same
		attribute names used by compute_chromatin_metrics_all_data().
		
		Parameters
		----------
		load_dir : str
			Directory where the CSV files are stored
		data_sources : list of str, optional
			List of data sources to load. If None, loads all available sources.
		"""
		if load_dir is None:
			load_dir = f"{self.output_dir}/chromatin_metrics"

		data_sources = ['deconvolved', 'raw_rep1', 'raw_rep2']
		loaded_metrics, loaded_ptrs = self.load_saved_metrics(load_dir, data_sources)

		# Assign to class attributes
		self.deconvolved_chromatin_metrics = loaded_metrics['deconvolved']
		self.deconvolved_chromatin_ptrs = loaded_ptrs.get('deconvolved', {})
		print_fl("Assigned deconvolved data to class attributes")
		
		self.raw_rep1_metrics = loaded_metrics['raw_rep1']
		self.raw_rep1_ptrs = loaded_ptrs.get('raw_rep1', {})
		print_fl("Assigned raw_rep1 data to class attributes")
		
		self.raw_rep2_metrics = loaded_metrics['raw_rep2']
		self.raw_rep2_ptrs = loaded_ptrs.get('raw_rep2', {})
		print_fl("Assigned raw_rep2 data to class attributes")

		# Load normalized data sources
		data_sources = ['normalized_deconvolved', 'normalized_raw_rep1', 'normalized_raw_rep2']
		loaded_metrics, loaded_ptrs = self.load_saved_metrics(load_dir, data_sources)

		# Assign to class attributes
		self.normalized_deconvolved_metrics = loaded_metrics['normalized_deconvolved']
		self.normalized_ptr_deconvolved = loaded_ptrs.get('normalized_deconvolved', {})
		print_fl("Assigned normalized_deconvolved data to class attributes")
		
		self.normalized_raw_rep1_metrics = loaded_metrics['normalized_raw_rep1']
		self.normalized_ptr_rep1 = loaded_ptrs.get('normalized_raw_rep1', {})
		print_fl("Assigned normalized_raw_rep1 data to class attributes")
		
		self.normalized_raw_rep2_metrics = loaded_metrics['normalized_raw_rep2']
		self.normalized_ptr_rep2 = loaded_ptrs.get('normalized_raw_rep2', {})
		print_fl("Assigned normalized_raw_rep2 data to class attributes")


def plot_ptr_change(raw_ptrs, deconv_ptrs, metric_name, selected_genes=[],
	cmap='Reds', bw=0.006, ptr_lims=(0.9, 4)):
	
	from src.DensityScatterPlotter import DensityScatterPlotter
	from matplotlib import pyplot as plt

	dsc_plotter = DensityScatterPlotter()
	dsc_plotter.plot_outline = True
	dsc_plotter.outline_color = '#eee'
	dsc_plotter.logz=True
	joined_data = pd.DataFrame({
		'x': raw_ptrs[metric_name+'_ptr'].values, 
		'y': deconv_ptrs[metric_name+'_ptr'].values
	}, index=raw_ptrs.index)

	# Fille nan values to ptr of 1.0
	joined_data = joined_data.fillna(1.)

	# Plot the PTRs
	dsc_plotter.set_data(joined_data.x, joined_data.y)
	dsc_plotter.bw = bw, bw
	dsc_plotter.cmap = cmap
	dsc_plotter.s = 5
	dsc_plotter.plot_ax(plt.gca())


	plt.plot([-0.5, 5], [-0.5, 5], c='red', lw=0.75, zorder=10, ls='dotted')

	# Plot the selected genes
	from src.sgd import get_orfname
	import matplotlib.patheffects as path_effects

	for gene_name in selected_genes:
		orf_name = get_orfname(gene_name)
		selected_data = joined_data.loc[orf_name]
		plt.scatter(selected_data.x, selected_data.y, marker='D', facecolor='none',
			lw=1, edgecolor='red', zorder=10)
		plt.text(selected_data.x, selected_data.y, gene_name, zorder=11,
			color='white', fontsize=8,
			path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
		
	plt.xlim(*ptr_lims)
	plt.ylim(*ptr_lims)
	plt.xlabel("Raw PTR")