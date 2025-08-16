import pandas as pd
import numpy as np
import os
from src.tf_sites import TFBindingSites
from src.deconvolved_chromatin_loader import DeconvolvedChromatinDataLoader
from src.transcripts_dataset import load_transcripts_sets
from src.peak_to_trough import compute_quantile_ptr_2d
import matplotlib.pyplot as plt
from src.figure_configs import save_figure_for_paper


class TranscriptionFactorProcessor:
	"""
	Analyzes chromatin accessibility around cell cycle transcription factor binding sites.
	
	Generates a DataFrame with mean accessibility profiles for CC TFs from Rossi dataset,
	filtered by Kelliher cell cycle TF list.
	"""
	
	def __init__(self, output_dir, window_size=40, fragment_range=(0, 100)):
		"""
		Initialize the ChromatinTFAnalyzer.
		
		Parameters:
		-----------
		output_dir : str
			Path to chromatin data directory
		window_size : int
			Base pairs around binding site (default ±40bp = 80bp total)
		fragment_range : tuple
			Fragment length filter range (default (0, 100))
		"""

		from src.utils import mkdir_safe

		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/figures_transcription_factors"
		mkdir_safe(self.save_dir)

		self.figures_dir = f"{self.output_dir}/Figures"

		self.window_size = window_size
		self.fragment_range = fragment_range
		
		# Data containers
		self.binding_sites = None
		self.tf_sites = None
		self.data_loader = None
		self.results_df = None
		
		# New containers for comprehensive analysis
		self.gene_promoters = None
		self.gene_associations = None
		self.comprehensive_df = None

		# Chromatin processor computes ptr threshold
		self._load_chromatin_processor()
		self._load_tf_datasets()
	
	# =================== NEW: File path helpers ===================
	
	def _get_results_filepath(self):
		"""Get filepath for basic results CSV."""
		return os.path.join(self.save_dir, 'binding_results.csv')
	
	def _get_comprehensive_filepath(self):
		"""Get filepath for comprehensive results CSV."""
		return os.path.join(self.save_dir, 'full_results.csv')
	
	def _results_exist(self):
		"""Check if basic results file exists."""
		return os.path.exists(self._get_results_filepath())
	
	def _comprehensive_results_exist(self):
		"""Check if comprehensive results file exists."""
		return os.path.exists(self._get_comprehensive_filepath())
	
	# =================== NEW: Save/Load methods ===================
	
	def save_results(self):
		"""
		Save results to disk.
		
		Saves:
		- binding_results.csv: Basic TF profiles (results_df)
		- full_results.csv: Comprehensive results with PTR and gene associations (comprehensive_df)
		"""
		if self.results_df is not None:
			self.results_df.to_csv(self._get_results_filepath())
			print(f"Saved basic results to: {self._get_results_filepath()}")
		
		if self.comprehensive_df is not None:
			self.comprehensive_df.to_csv(self._get_comprehensive_filepath())
			print(f"Saved comprehensive results to: {self._get_comprehensive_filepath()}")
	
	def load_results(self):
		"""
		Load previously saved results from disk.
		
		Returns:
		--------
		tuple
			(basic_loaded, comprehensive_loaded) - booleans indicating what was loaded
		"""
		basic_loaded = False
		comprehensive_loaded = False
		
		# Load basic results
		if self._results_exist():
			try:
				self.results_df = pd.read_csv(self._get_results_filepath(), index_col=[0, 1])
				print(f"Loaded basic results from: {self._get_results_filepath()}")
				basic_loaded = True
			except Exception as e:
				print(f"Warning: Failed to load basic results: {e}")
		
		# Load comprehensive results
		if self._comprehensive_results_exist():
			try:
				self.comprehensive_df = pd.read_csv(self._get_comprehensive_filepath(), index_col=[0, 1])
				print(f"Loaded comprehensive results from: {self._get_comprehensive_filepath()}")
				comprehensive_loaded = True
			except Exception as e:
				print(f"Warning: Failed to load comprehensive results: {e}")
		
		return basic_loaded, comprehensive_loaded
	
	def clear_saved_results(self):
		"""
		Remove saved result files from disk.
		"""
		files_to_remove = [self._get_results_filepath(), self._get_comprehensive_filepath()]
		
		for filepath in files_to_remove:
			if os.path.exists(filepath):
				os.remove(filepath)
				print(f"Removed: {filepath}")
			else:
				print(f"File not found: {filepath}")
	
	# =================== UPDATED: Main analysis methods ===================
	
	def build_dataframe(self, force_recompute=False):
		"""
		Main method that orchestrates the entire workflow.
		
		Parameters:
		-----------
		force_recompute : bool
			If True, recompute even if saved results exist
		
		Returns:
		--------
		pd.DataFrame
			MultiIndex ['tf', 'identifier'] with timepoint columns containing
			mean accessibility values across the 80bp window
		"""

		# Execute workflow steps
		self._prepare_binding_sites()

		# Check if we can load from disk
		if not force_recompute and self._results_exist():
			basic_loaded, _ = self.load_results()
			if basic_loaded and self.results_df is not None:
				print("Using saved basic results. Use force_recompute=True to regenerate.")
				return self.results_df
		
		print("Computing basic TF binding profiles...")

		self._load_chromatin_data()
		
		# Save results
		self.save_results()
		
		return self.results_df
	
	def build_comprehensive_dataframe(self, force_recompute=False):
		"""
		Build complete dataframe with TF profiles, PTR values, and gene associations.
		
		Parameters:
		-----------
		force_recompute : bool
			If True, recompute even if saved results exist
			
		Returns:
		--------
		pd.DataFrame
			MultiIndex ['tf', 'identifier'] with columns:
			- timepoint_0, timepoint_1, ... (accessibility profiles)
			- ptr (if include_ptr=True)
			- associated_genes (comma-separated gene names)
			- association_type ('overlap' or 'none')
		"""
		# Check if we can load comprehensive results from disk
		if not force_recompute and self._comprehensive_results_exist():
			_, comprehensive_loaded = self.load_results()
			if comprehensive_loaded and self.comprehensive_df is not None:
				print("Using saved comprehensive results. Use force_recompute=True to regenerate.")
				return self.comprehensive_df
		
		print("Building comprehensive dataframe with gene associations...")
		
		# First get the basic TF profiles (may load from disk if available)
		if self.results_df is None:
			self.build_dataframe(force_recompute=force_recompute)
		
		# Start with the TF profile data
		self.comprehensive_df = self.results_df.copy()
		
		# Add PTR values if requested
		self._add_ptr_values()
		
		# Save comprehensive results
		self.save_results()
		
		print(f"Comprehensive dataframe complete with {len(self.comprehensive_df)} sites")
		return self.comprehensive_df
	
	# =================== EXISTING METHODS (unchanged) ===================
	
	def _load_tf_datasets(self):
		"""
		Load and integrate TF binding datasets.
		- Initialize TFBindingSites()
		- Get Rossi-Kelliher intersection
		- Filter by confidence (value == 1000)
		- Sort by chr, start position
		"""
		print("Loading TF binding datasets...")
		
		# Initialize TF binding sites
		self.binding_sites = TFBindingSites()
		
		# Get Rossi data and filter for cell cycle TFs with high confidence
		rossi_sites = self.binding_sites.filtered_rossi_go_tf_binding_sites
		rossi_sites['identifier'] = rossi_sites['chr'].astype(str) +'_'+ rossi_sites['start'].astype(str)\
			+rossi_sites['strand']
		
		# Sort by chromosome and start position for efficient loading
		self.tf_sites = rossi_sites.sort_values(['chr', 'start']).set_index(['tf', 'identifier'])
		
		print(f"Found {len(self.tf_sites)} high-confidence binding sites for "
			  f"{len(self.tf_sites.reset_index().tf.unique())} cell cycle TFs")
	
	def _prepare_binding_sites(self):
		"""
		Prepare binding site coordinates and metadata.
		- Validate genomic coordinates
		- Calculate analysis windows
		- Store site metadata for DataFrame index
		"""
		print("Preparing binding site coordinates...")
		
		# Add window coordinates for analysis
		self.tf_sites['window_start'] = self.tf_sites['start'] - self.window_size
		self.tf_sites['window_end'] = self.tf_sites['end'] + self.window_size
		
		print(f"Prepared {len(self.tf_sites)} binding sites with ±{self.window_size}bp windows")
	
	
	def _load_chromatin_data(self):
		"""
		Load and process chromatin data for each binding site.
		- Initialize DeconvolvedChromatinDataLoader
		- For each site: load_mnase_span() with ±1000bp buffer
		- Extract ±40bp window with fragment filtering
		- Calculate mean across 80bp window for each timepoint
		"""
		print("Loading and processing chromatin data...")
		
		# Initialize chromatin data loader
		self.data_loader = DeconvolvedChromatinDataLoader(self.output_dir)
		
		# Storage for results
		results_list = []
		
		# Process each binding site
		for idx, (index, tf_site) in enumerate(self.tf_sites.iterrows()):

			tf, identifier = index

			if idx % 200 == 0:  # Progress indicator
				print(f"Processing site {idx + 1}/{len(self.tf_sites)}")

			try:
				# Calculate mean accessibility for this site
				mean_profile = self._calculate_site_mean(tf_site)
				
				# Store results with metadata
				site_result = {
					'tf': tf,
					'identifier': identifier,
					'mean_profile': mean_profile
				}
				results_list.append(site_result)
				
			except Exception as e:
				print(f"Warning: Failed to process site {identifier}: {e}")
				continue
		
		# Convert to structured DataFrame
		self._build_results_dataframe(results_list)
		
		print(f"Successfully processed {len(results_list)} binding sites")
	
	def _calculate_site_mean(self, tf_site):
		"""
		Calculate mean accessibility across genomic window and fragment lengths.
		
		Parameters:
		-----------
		tf_site : pd.Series
			Row from tf_sites DataFrame containing site information
			
		Returns:
		--------
		np.array
			Mean accessibility profile across timepoints
		"""
		# Define genomic regions
		tf_binding_span = (tf_site['window_start'], tf_site['window_end'])
		load_region = (tf_site['start'] - 1000, tf_site['start'] + 1000)
		
		# Load MNase data in expanded region
		self.data_loader.load_mnase_span(tf_site['chr'], load_region)
		
		# Extract specific window with fragment filtering
		binding_site_data = self.data_loader.subset_loaded_data(
			genomic_region=tf_binding_span,
			fragment_lengths=self.fragment_range
		)
		
		# Calculate mean across genomic positions and fragment lengths
		# Assuming dimensions are (timepoints, positions, fragment_lengths)
		mean_binding_data = binding_site_data.mean((1, 2))
		
		return mean_binding_data
	
	def _build_results_dataframe(self, results_list):
		"""
		Build the final results DataFrame with MultiIndex.
		
		Parameters:
		-----------
		results_list : list
			List of dictionaries containing results for each site
		"""
		if not results_list:
			raise ValueError("No valid results to build DataFrame")
		
		# Extract data components
		tf_names = [result['tf'] for result in results_list]
		identifiers = [result['identifier'] for result in results_list]
		profiles = np.array([result['mean_profile'] for result in results_list])
		
		# Create MultiIndex
		multi_index = pd.MultiIndex.from_arrays(
			[tf_names, identifiers], 
			names=['tf', 'identifier']
		)
		
		# Determine column names (timepoints)
		n_timepoints = profiles.shape[1]
		columns = [f'timepoint_{i}' for i in range(n_timepoints)]
		
		# Build DataFrame
		self.results_df = pd.DataFrame(
			data=profiles,
			index=multi_index,
			columns=columns
		)
	
	def get_tf_data(self, tf_name):
		"""
		Get data for a specific transcription factor.
		
		Parameters:
		-----------
		tf_name : str
			Name of the transcription factor
			
		Returns:
		--------
		pd.DataFrame
			Subset of results for the specified TF
		"""
		if self.results_df is None:
			raise ValueError("No results available. Run build_dataframe() first.")
		
		return self.results_df.loc[tf_name]
	
	def _load_gene_promoters(self):
		"""
		Load gene promoter dataset from transcripts.
		
		Stores:
		- self.gene_promoters: DataFrame with orf_name, chr, promoter_start, promoter_end
		"""
		print("Loading gene promoter data...")
		
		geneset, _ = load_transcripts_sets(self.output_dir, combined=False)
		
		# Create a clean DataFrame with necessary columns
		self.gene_promoters = pd.DataFrame({
			'orf_name': geneset.index,
			'chr': geneset['chr'],
			'promoter_start': geneset['promoter_start'],
			'promoter_end': geneset['promoter_end']
		}).reset_index(drop=True)
		
		print(f"Loaded {len(self.gene_promoters)} gene promoters")
	
	def _find_gene_associations(self):
		"""
		Find exact overlaps between TF binding sites and gene promoters.
		
		Stores:
		--------
		self.gene_associations : dict
			Mapping from TF site identifier to list of associated gene names
		"""
		print("Finding TF-gene associations...")
		
		self.gene_associations = {}
		association_count = 0
		
		# For each TF binding site
		for identifier, tf_site in self.tf_sites.iterrows():
			associated_genes = []
			
			# Find genes on the same chromosome
			same_chr_genes = self.gene_promoters[
				self.gene_promoters['chr'] == tf_site['chr']
			]
			
			# Check for overlaps with each gene promoter
			for _, gene in same_chr_genes.iterrows():
				if self._check_overlap(
					tf_site['start'], tf_site['end'],
					gene['promoter_start'], gene['promoter_end']
				):
					associated_genes.append(gene['orf_name'])
			
			# Store the associations (empty list if no associations)
			self.gene_associations[identifier] = associated_genes
			if associated_genes:
				association_count += 1
		
		print(f"Found gene associations for {association_count}/{len(self.tf_sites)} TF binding sites")
	
	def _check_overlap(self, tf_start, tf_end, promoter_start, promoter_end):
		"""
		Check if TF binding site overlaps with gene promoter region.
		
		Parameters:
		-----------
		tf_start, tf_end : int
			TF binding site coordinates
		promoter_start, promoter_end : int
			Gene promoter coordinates
			
		Returns:
		--------
		bool
			True if ranges overlap, False otherwise
		"""
		# Two ranges overlap if: start1 <= end2 and start2 <= end1
		return tf_start <= promoter_end and promoter_start <= tf_end
	
	def _integrate_gene_associations(self):
		"""
		Add gene association columns to results DataFrame.
		
		Adds columns:
		- associated_genes: comma-separated gene names or empty string
		- association_type: 'overlap' or 'none'
		"""
		print("Integrating gene association data...")
		
		# Prepare data for each site
		associated_genes_list = []
		association_types = []
		
		for identifier in self.comprehensive_df.index.get_level_values('identifier'):
			genes = self.gene_associations.get(identifier, [])
			
			if genes:
				associated_genes_list.append(','.join(genes))
				association_types.append('overlap')
			else:
				associated_genes_list.append('')
				association_types.append('none')
		
		# Add columns to dataframe
		self.comprehensive_df['associated_genes'] = associated_genes_list
		self.comprehensive_df['association_type'] = association_types
		
		n_with_genes = sum(1 for genes in associated_genes_list if genes)
		print(f"Added gene associations: {n_with_genes} sites with gene overlaps")
	
	def _load_chromatin_processor(self):
		from pipeline.chromatin_metrics_processor import ChromatinMetricsProcessor

		chromatin_processor = ChromatinMetricsProcessor(self.output_dir)
		chromatin_processor.load_and_assign_saved_metrics()
		promoter_values = chromatin_processor.deconvolved_chromatin_metrics['promoter_occupancy'].dropna().values.flatten()
		self.ptr_threshold = np.quantile(promoter_values, q=0.95)


	def _add_ptr_values(self):
		"""
		Calculate and add peak-to-trough ratios to results DataFrame.
		
		Adds column:
		- ptr: peak-to-trough ratio for each binding site
		"""
		print("Calculating peak-to-trough ratios...")
		
		# Get only the timepoint columns for PTR calculation
		timepoint_cols = [col for col in self.comprehensive_df.columns 
						 if col.startswith('timepoint_')]
		timepoint_data = self.comprehensive_df[timepoint_cols]
		
		# Calculate PTR values
		ptr_values = compute_quantile_ptr_2d(timepoint_data)
		
		# Add to dataframe
		self.comprehensive_df['ptr'] = ptr_values
		
		print(f"Added PTR values (mean: {np.nanmean(ptr_values):.3f})")
	
	def get_association_summary(self):
		"""
		Get summary statistics about TF-gene associations.
		
		Returns:
		--------
		dict
			Summary statistics about associations
		"""
		if self.comprehensive_df is None:
			return {"error": "No comprehensive results available. Run build_comprehensive_dataframe() first."}
		
		with_genes = self.comprehensive_df['association_type'] == 'overlap'
		
		return {
			"total_sites": len(self.comprehensive_df),
			"sites_with_genes": with_genes.sum(),
			"sites_without_genes": (~with_genes).sum(),
			"association_rate": with_genes.mean(),
			"genes_per_tf": self.comprehensive_df[with_genes].groupby(level='tf')['associated_genes'].apply(
				lambda x: x.str.split(',').apply(len).sum()
			).to_dict() if with_genes.any() else {}
		}
	
	def get_sites_for_gene(self, gene_name):
		"""
		Get all TF binding sites associated with a specific gene.
		
		Parameters:
		-----------
		gene_name : str
			Gene name (orf_name) to search for
			
		Returns:
		--------
		pd.DataFrame
			Subset of comprehensive results for sites associated with the gene
		"""
		if self.comprehensive_df is None:
			raise ValueError("No comprehensive results available. Run build_comprehensive_dataframe() first.")
		
		# Find sites where the gene appears in associated_genes
		mask = self.comprehensive_df['associated_genes'].str.contains(
			gene_name, na=False, regex=False
		)
		
		return self.comprehensive_df[mask]

	def statistical_test_for_promoters(self):
		cycling_site_locations = self.cell_cycle_site_counts_pivoted
		all_site_locations = self.all_site_counts_pivoted

		cell_cycle_location_props = cycling_site_locations / \
			cycling_site_locations.sum(1).values[:, None]
		all_location_props = all_site_locations / \
			all_site_locations.sum(1).values[:, None]

		joined_comparison = cycling_site_locations.join(all_site_locations, lsuffix='_cycling',
								   rsuffix='_all')
		total_joined = joined_comparison.sum()

		total_cycling = total_joined.gene_body_cycling+total_joined.promoter_cycling+\
			total_joined.intergenic_cycling

		total_all = total_joined.gene_body_all+total_joined.promoter_all+\
			total_joined.intergenic_all
		
		from src.stats_utils import test_z_proportion_change

		ret_text = test_z_proportion_change(total_joined.promoter_cycling, total_cycling,
							 total_joined.promoter_all, total_all)

		test_filepath = f"{self.save_dir}/promoter_statistical_test.txt"
		with open(test_filepath, 'w') as f:
			f.write(ret_text)
		print("Wrote to: ", test_filepath)

	def pivot_sites_data(self, sites):

		def _classify_site_with_genes(site, gene_boundaries):
			# Check if the site is within a gene's body: TSS-PAS

			# Check if the site is within a promoter: predefined promoter range
			chrom_strand_check = (site.chr == gene_boundaries.chr) & \
							(site.strand == gene_boundaries.strand)

			found_in_genes = gene_boundaries[chrom_strand_check & 
							(site.end > gene_boundaries.full_transcript_start) &
							(site.start < gene_boundaries.full_transcript_end)]

			found_in_promoter = gene_boundaries[chrom_strand_check & 
							(site.end > gene_boundaries.promoter_start) &
							(site.start < gene_boundaries.promoter_end)]

			if len(found_in_genes) > 0:
				classification = 'gene_body'
			elif len(found_in_promoter) > 0:
				classification = 'promoter'
			else:
				classification = 'intergenic'
			return classification

		from src.transcripts_dataset import load_transcripts_sets
		gene_boundaries, _  = load_transcripts_sets(self.output_dir)

		# Next we'll look through these sites and count the occurrence in various gene contexts
		tf_sites = self.tf_sites.copy()

		sites_with_peaks = sites.join(tf_sites, how='left').loc[self.sorted_boxplot_tfs_index]
		for site_index, site in sites_with_peaks.iterrows():
			classification = _classify_site_with_genes(site, gene_boundaries)
			sites_with_peaks.loc[site_index, 'genomic_classification'] = classification
		# Group by both TF and classification
		counts = sites_with_peaks.groupby(['tf', 'genomic_classification']).size().reset_index(name='count')

		# Pivot for easier viewing
		pivot_counts = counts.pivot(index='tf', columns='genomic_classification', values='count').fillna(0)\
			.loc[self.sorted_boxplot_tfs_index]
		return pivot_counts

	def classify_genomic_cell_cycle_sites(self):

		ptrs = self.comprehensive_df[['ptr']].dropna()
		cell_cycle_sites = ptrs[ptrs.ptr > self.ptr_threshold]

		self.cell_cycle_site_counts_pivoted = self.pivot_sites_data(cell_cycle_sites)
		self.all_site_counts_pivoted = self.pivot_sites_data(ptrs)

	def test_cycling_tfs_for_prom_testing(self):
		"""
		Perform statistical test to see which TFs are significantly bound to promoters
		"""
		
		from src.stats_utils import test_z_proportion_change
		def _test_p_value_promoter_change(self, tf):
			all_counts = self.all_site_counts_pivoted.loc[tf]
			cycling_counts = self.cell_cycle_site_counts_pivoted.loc[tf]

			total_all = all_counts.sum()
			total_cycling = cycling_counts.sum()
			res = test_z_proportion_change(cycling_counts.promoter, total_cycling, 
										   self.all_site_counts_pivoted.promoter.sum(), 
										   self.all_site_counts_pivoted.sum().sum(),
										   print_results=False, alpha=0.01)
			return res

		stat_test_results = pd.DataFrame()

		for tf in self.all_site_counts_pivoted.index:
			try:
				res = _test_p_value_promoter_change(self, tf)
				stat_test_results.loc[tf, 'p_value'] = res[2]
			except ZeroDivisionError:
				continue

		from statsmodels.stats.multitest import multipletests

		# Your p-values
		p_values = stat_test_results.p_value

		# Apply Benjamini-Hochberg correction
		rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
			p_values, 
			alpha=0.1,
			method='fdr_bh'  # Benjamini-Hochberg
		)
		stat_test_results['p_adjusted'] = p_adjusted
		total_cycling_sites_per_tf = self.cell_cycle_site_counts_pivoted.sum(1)
		stat_test_results['total_cycling_sites'] = total_cycling_sites_per_tf

		stats_results = stat_test_results[(stat_test_results.p_value < 0.2) & 
						  (stat_test_results.total_cycling_sites > 7)]

		self.significant_promoter_tfs = stats_results
		self.significant_promoter_tfs.to_csv(f"{self.save_dir}/significant_cycling_promoters.csv")

		return stats_results


	def plot_before_after_genomic_classifications(self):

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
		self.plot_genomic_classifications(mode='all', ax=ax1)
		self.plot_genomic_classifications(mode='cycling', ax=ax2)
		plt.subplots_adjust(wspace=0.3)
		save_figure_for_paper(f'{self.save_dir}/binding_locations_both.png')


	def plot_genomic_classifications(self, mode='cycling', ax=None):
		import matplotlib.pyplot as plt

		if mode == 'all':
			pivot_counts = self.all_site_counts_pivoted
			title = 'All TF binding locations'
			bar_colors = ['#777', '#444', '#bbb']
		else:
			pivot_counts = self.cell_cycle_site_counts_pivoted
			title = 'Cycling TF binding locations'
			bar_colors = [plt.cm.Oranges(0.45), plt.cm.Purples(0.6), '#bbb']

		# Assuming your pivot table is called 'pivot_counts'
		if ax is None:
			fig, ax = plt.subplots(figsize=(5, 7))

		# Counts per category
		counts = pivot_counts.sum(0)
		total = pivot_counts.values.sum()
		nonintergenic = (counts.gene_body+counts.promoter)
		nonintergenic_perc = nonintergenic/total*100
		promoter_num = (counts.promoter)
		promoter_perc = promoter_num/total*100

		# Plot with specific column order
		pivot_counts[['promoter', 'gene_body', 'intergenic']].plot(kind='barh', stacked=True, ax=ax,
			color=bar_colors, width=0.67)

		# Customize the plot
		ax.set_xlabel('Number of Sites')
		ax.set_ylabel('')

		ax.set_title(f'{title}\n'\
				  f'n={total:.0f}, {promoter_num:.0f} promoter-binding ({promoter_perc:.0f}%)', 
			fontweight='demi', fontsize=18, pad=13)

		ax.legend([f'Promoter, n={counts.promoter:.0f} ({counts.promoter/total*100:.0f}%)', 
					f'Within Gene, n={counts.gene_body:.0f} ({counts.gene_body/total*100:.0f}%)', 
					f'Intergenic, n={counts.intergenic:.0f} ({counts.intergenic/total*100:.0f}%)'], 
					loc='lower right')
		
		xmax = pivot_counts.values.sum(1).max()
		ax.set_xlim(0, xmax*1.6)
		ax.set_yticklabels(pivot_counts.index, fontsize=12)

		cc_color = plt.cm.Blues(0.7)
		# Color the y-tick labels based on cell cycle criteria
		for i, tf_name in enumerate(pivot_counts.index):

			if tf_name.upper() in self.binding_sites.cell_cycle_rossi_tfs:
				ax.get_yticklabels()[i].set_color(cc_color)  # Darker orange for text readability
				color = cc_color
			else:
				color = 'black'

			# Count labels
			tf_counts = pivot_counts.loc[tf_name]
			total_counts = tf_counts.values.sum()
			num_non_intergenic = tf_counts.promoter+tf_counts.gene_body
			num_promoter = tf_counts.promoter

			label = f"{total_counts:.0f}, {num_promoter:.0f}"\
					f" ({(num_promoter/total_counts)*100:.0f}%)"

			if mode == 'cycling' and tf_name in self.significant_promoter_tfs.index:

				p_value = self.significant_promoter_tfs.loc[tf_name].p_value
				label += " *"

				ax.axhspan(i-0.5, i+0.52, 0, 1, color='yellow', zorder=0, alpha=0.16)

				if p_value < 0.1:
					label += "*"

				if p_value < 0.01:
					label += "*"

			ax.text(total_counts+xmax*0.01, i, label, ha='left', va='center', color=color)

	
	def plot_tf_boxplots(self, column='ptr', figsize=(6, 7), 
						 show_outliers=False, 
						 show_points=True, 
						 point_color='#777', point_alpha=0.5, 
						 point_size=1, jitter_width=0.1):
		"""
		Plot box plots for specified column values grouped by transcription factor (tf).
		"""

		ptr_values = self.comprehensive_df[['ptr']].dropna()
		number_of_cell_cycle_sites = len(ptr_values[ptr_values.ptr > self.ptr_threshold])
		n = len(ptr_values)

		median_ptrs = ptr_values.reset_index()[['tf', 'ptr']].groupby('tf').median()\
			.rename(columns={'ptr': 'median'})
		
		# Count number of sites
		num_sites = ptr_values.dropna().reset_index()[['tf', 'ptr']]\
			.groupby('tf').count()
		num_sites = num_sites.sort_values('ptr').rename(columns={'ptr': 'num_total'})
		
		# Count number of cycling sites
		cycling_counts = ptr_values[ptr_values > self.ptr_threshold].reset_index()[['tf', 'ptr']]\
			.groupby('tf').count()
		cycling_counts = cycling_counts.sort_values('ptr').rename(columns={'ptr': 'num_cycling'})

		ptr_counts_df = cycling_counts.join(num_sites).join(median_ptrs)
		ptr_counts_df['prop_cycling'] = ptr_counts_df.num_cycling / ptr_counts_df.num_total
		ptr_counts_df = ptr_counts_df.sort_values(['prop_cycling', 'num_cycling', 'num_total'])
		
		# Subset by threshold, min 20
		num_threshold = 20
		ptr_counts_df = ptr_counts_df[ptr_counts_df.num_total > num_threshold]

		print(f"Number of transcription factors with >{num_threshold} sites", len(ptr_counts_df))
		
		# Sort by number of cycling counts
		sorted_tf_index = ptr_counts_df.index

		# Prepare data for box plots
		data_list = []
		tf_names = []
		
		for tf_name in sorted_tf_index:
			clean_data = ptr_values.loc[tf_name][column].dropna().values
			
			if len(clean_data) > 0:  # Only include if there's data
				data_list.append(clean_data)
				tf_names.append(tf_name)

		# Create the box plot
		fig, ax = plt.subplots(figsize=figsize)
		
		max_ptr_plot = 2.5
		max_xlim = max_ptr_plot + 0.02

		# Add jittered scatter points if requested
		if show_points:
			np.random.seed(42)  # For reproducible jitter
			for i, data in enumerate(data_list):
				# Create jittered y-positions
				y_pos = i + 1  # Box positions are 1-indexed
				y_jitter = np.random.uniform(-jitter_width, jitter_width, len(data))
				y_positions = y_pos + y_jitter

				plot_data = data.copy()
				plot_data[plot_data > max_ptr_plot] = max_ptr_plot
				
				# Plot the scattered points
				ax.scatter(plot_data, y_positions, alpha=point_alpha, 
						  color=point_color, s=point_size, zorder=1)

		boxprops = dict(linewidth=0)
		medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
		box_plot = ax.boxplot(data_list, labels=tf_names, patch_artist=True, 
							 showfliers=show_outliers, vert=False, boxprops=boxprops,
							 widths=0.7,
							 medianprops=dict(color='black', solid_capstyle='butt'),
							 showcaps=False, zorder=0)
		
		# Color the boxes
		color=plt.cm.Blues(0.4)
		for i, patch in enumerate(box_plot['boxes']):
			tf = sorted_tf_index[i]

			# Color the cell cycle tfs
			if tf.upper() in self.binding_sites.cell_cycle_rossi_tfs:
				patch.set_facecolor(color)
			else:
				patch.set_facecolor('#aaa')

			patch.set_alpha(1.0)
		
		# Customize the plot
		ax.set_xlabel(f'{column.upper()} Value', fontsize=12)
		ax.set_ylabel('Transcription Factor', fontsize=12)

		title = f"Cyclicity of transcription factor\nbinding, n={n}, {number_of_cell_cycle_sites} cycling ({number_of_cell_cycle_sites/n*100:.0f}%)"
		ax.set_title(title, fontsize=21, fontweight='demi', pad=13)
		
		# Add some statistics as text
		n_factors = len(tf_names)
		total_sites = sum(len(data) for data in data_list)
		
		tick_names = []
		numeric_tick_names = []
		for i, tf_name in enumerate(sorted_tf_index):
			tick_names.append(tf_name)
			
			num_total_sites = num_sites.loc[tf_name].num_total
			num_cycling_sites = cycling_counts.loc[tf_name].num_cycling
			tick_name = f"{num_total_sites}, {num_cycling_sites} ({num_cycling_sites/num_total_sites*100:.0f}%)"
			
			y_position = i+1
			x = 0.45
			numeric_tick_names.append(tick_name)

		right_side_ax = ax.twinx()
		right_side_ax.set_yticks(range(1, len(numeric_tick_names)+1))
		right_side_ax.set_yticklabels(numeric_tick_names, fontsize=12)
		right_side_ax.tick_params(axis='y', which='major', length=0, pad=5)

		ax.set_yticklabels(tick_names, fontsize=12)
		ax.set_ylim(0.5, len(sorted_tf_index)+0.5)
		right_side_ax.set_ylim(0.5, len(sorted_tf_index)+0.5)
		xticks = np.arange(1, max_xlim, 0.25)
		xticklabels = [f"{x}" for x in xticks]
		xticklabels[-1] = f"{max_ptr_plot}+"

		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)

		ax.set_xlim(0.95, max_xlim)
		plt.tight_layout()

		ax.axvline(self.ptr_threshold, c='black', lw=1, ls='dotted')

		# Color the y-tick labels based on cell cycle criteria
		for i, tf_name in enumerate(sorted_tf_index):
			if tf_name.upper() in self.binding_sites.cell_cycle_rossi_tfs:
				ax.get_yticklabels()[i].set_color(plt.cm.Blues(0.7))  # Darker orange for text readability
				right_side_ax.get_yticklabels()[i].set_color(plt.cm.Blues(0.7))  # Darker orange for text readability
			else:
				right_side_ax.get_yticklabels()[i].set_color('black')  # Default color for non-cell cycle TFs

		# Create custom legend
		legend_elements = [
			plt.Line2D([0], [0], color=color, lw=2, label='Cell cycle TF (Kelliher, 2018)'),
			plt.Line2D([0], [0], color='gray', lw=2, label='Non cell cycle')
		]
		ax.legend(handles=legend_elements)
			
		# Keep track of the transcription factors plotted and sorting
		self.sorted_boxplot_tfs_index = sorted_tf_index

		save_figure_for_paper(f"{self.save_dir}/factor_binding_cyclicity.png")

		return fig, ax, sorted_tf_index

	def layout_panel(self):
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 400, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/factor_binding_cyclicity.png',
			f'{self.save_dir}/binding_locations_both.png',
		]

		# Layout images horizontally with custom width proportions
		# Adjust these proportions based on your image content needs
		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			width_proportions=[1, 1.95],
			between_padding=30,
			margin=(30, 30),
			image_keys=['binding_cyclicity', 'locations_both']  # Custom keys
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=26,
			offset=(-15, 0)  # Adjust offset as needed
		)

		compositor.add_panel_label_to_image(
			'locations_both', 
			'C', 
			offset=(320, 0),
			font_size=26,
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental5.5_Transcription_Factors.png')