import pandas as pd
import numpy as np
from src.tf_sites import TFBindingSites
from src.deconvolved_chromatin_loader import DeconvolvedChromatinDataLoader
from src.transcripts_dataset import load_transcripts_sets
from src.peak_to_trough import compute_quantile_ptr_2d
import matplotlib.pyplot as plt


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
		self.output_dir = output_dir
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
	
	def build_dataframe(self):
		"""
		Main method that orchestrates the entire workflow.
		
		Returns:
		--------
		pd.DataFrame
			MultiIndex ['tf', 'identifier'] with timepoint columns containing
			mean accessibility values across the 80bp window
		"""
		# Execute workflow steps
		self._load_tf_datasets()
		self._prepare_binding_sites()
		self._load_chromatin_data()
		
		return self.results_df
	
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
		
		# Sort by chromosome and start position for efficient loading
		self.tf_sites = rossi_sites.sort_values(['chr', 'start'])
		
		print(f"Found {len(self.tf_sites)} high-confidence binding sites for "
			  f"{len(self.tf_sites.tf.unique())} cell cycle TFs")
	
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
		for idx, (identifier, tf_site) in enumerate(self.tf_sites.iterrows()):
			if idx % 200 == 0:  # Progress indicator
				print(f"Processing site {idx + 1}/{len(self.tf_sites)}")
			
			try:
				# Calculate mean accessibility for this site
				mean_profile = self._calculate_site_mean(tf_site)
				
				# Store results with metadata
				site_result = {
					'tf': tf_site['tf'],
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
	
	def get_summary_stats(self):
		"""
		Get summary statistics about the analysis results.
		
		Returns:
		--------
		dict
			Summary statistics
		"""
		if self.results_df is None:
			return {"error": "No results available. Run build_dataframe() first."}
		
		return {
			"total_sites": len(self.results_df),
			"unique_tfs": len(self.results_df.index.get_level_values('tf').unique()),
			"timepoints": len(self.results_df.columns),
			"tf_counts": self.results_df.index.get_level_values('tf').value_counts().to_dict()
		}
	
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
	
	def build_comprehensive_dataframe(self, include_ptr=True):
		"""
		Build complete dataframe with TF profiles, PTR values, and gene associations.
		
		Parameters:
		-----------
		include_ptr : bool
			Whether to calculate and include peak-to-trough ratios
			
		Returns:
		--------
		pd.DataFrame
			MultiIndex ['tf', 'identifier'] with columns:
			- timepoint_0, timepoint_1, ... (accessibility profiles)
			- ptr (if include_ptr=True)
			- associated_genes (comma-separated gene names)
			- association_type ('overlap' or 'none')
		"""
		print("Building comprehensive dataframe with gene associations...")
		
		# First get the basic TF profiles
		if self.results_df is None:
			self.build_dataframe()
		
		# Load gene promoter data
		self._load_gene_promoters()
		
		# Find gene associations
		self._find_gene_associations()
		
		# Start with the TF profile data
		self.comprehensive_df = self.results_df.copy()
		
		# Add PTR values if requested
		if include_ptr:
			self._add_ptr_values()
		
		# Add gene association information
		self._integrate_gene_associations()
		
		print(f"Comprehensive dataframe complete with {len(self.comprehensive_df)} sites")
		return self.comprehensive_df
	
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

	def plot_tf_boxplots(self, column='ptr', figsize=(6, 7), 
						 title="Cell cycling transcription factor binding", 
						 show_outliers=False, 
						 show_points=True, 
						 point_color='#aaa', point_alpha=1.0, 
						 point_size=1, jitter_width=0.1):
		"""
		Plot box plots for specified column values grouped by transcription factor (tf).
		"""

		ptr_values = self.comprehensive_df[['ptr']]

		median_ptrs = ptr_values.reset_index()[['tf', 'ptr']].groupby('tf').median()\
			.rename(columns={'ptr': 'median'})
		
		# Count number of sites
		num_sites = ptr_values.dropna().reset_index()[['tf', 'ptr']]\
			.groupby('tf').count()
		num_sites = num_sites.sort_values('ptr').rename(columns={'ptr': 'num_total'})
		
		# Count number of cycling sites
		cycling_counts = ptr_values[ptr_values > self.ptr_threshold].dropna().reset_index()[['tf', 'ptr']]\
			.groupby('tf').count()
		cycling_counts = cycling_counts.sort_values('ptr').rename(columns={'ptr': 'num_cycling'})

		ptr_counts_df = cycling_counts.join(num_sites).join(median_ptrs)
		ptr_counts_df['prop_cycling'] = ptr_counts_df.num_cycling / ptr_counts_df.num_total
		ptr_counts_df = ptr_counts_df.sort_values(['prop_cycling', 'num_cycling', 'num_total'])
		
		# Subset by threshold, min 20
		num_threshold = 20
		ptr_counts_df = ptr_counts_df[ptr_counts_df.num_total > num_threshold]
		
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
		
		# Create box plots
		color=plt.cm.Oranges(0.35)
		
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
						  color=point_color, s=point_size, zorder=0)

		box_plot = ax.boxplot(data_list, labels=tf_names, patch_artist=True, 
							 showfliers=show_outliers, vert=False)
		
		# Color the boxes
		for i, patch in enumerate(box_plot['boxes']):
			tf = sorted_tf_index[i]

			# Color the cell cycle tfs
			if tf.upper() in self.binding_sites.cell_cycle_rossi_tfs:
				patch.set_facecolor(color)
			else:
				patch.set_facecolor('#ddd')

			patch.set_alpha(0.25)
		
		# Customize the plot
		ax.set_xlabel(f'{column.upper()} Value', fontsize=12)
		ax.set_ylabel('Transcription Factor', fontsize=12)
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
			tick_name = f"n= {num_total_sites}, {num_cycling_sites} ({num_cycling_sites/num_total_sites*100:.0f}%)"
			
			y_position = i+1
			x = 0.45
			numeric_tick_names.append(tick_name)

		right_side_ax = ax.twinx()
		right_side_ax.set_yticks(range(1, len(numeric_tick_names)+1))
		right_side_ax.set_yticklabels(numeric_tick_names, fontsize=8)
		right_side_ax.tick_params(axis='y', which='major', length=0, pad=5)

		ax.set_yticklabels(tick_names, fontsize=10)
		ax.set_ylim(0, len(sorted_tf_index)+1)
		right_side_ax.set_ylim(0, len(sorted_tf_index)+1)
		xticks = np.arange(1, 2.25, 0.25)
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
				ax.get_yticklabels()[i].set_color(plt.cm.Oranges(0.7))  # Darker orange for text readability
			else:
				ax.get_yticklabels()[i].set_color('black')  # Default color for non-cell cycle TFs

		# Create custom legend
		legend_elements = [
		    plt.Line2D([0], [0], color=color, lw=2, label='Cell cycle TF (Kelliher, 2018)'),
		    plt.Line2D([0], [0], color='gray', lw=2, label='Non cell cycle')
		]
		ax.legend(handles=legend_elements)
			
		return fig, ax


