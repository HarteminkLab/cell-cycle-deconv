import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


class TranscriptTSSUpdateAnalyzer:
	"""
	Analyzes RNA transcript calls against reference datasets (SGD and Park TSS).
	
	Workflow:
	1. Load transcript calls from disk
	2. Join with SGD non-dubious genes and Park TSS datasets
	3. Remove duplicate transcripts 
	4. Categorize genes and count updates
	5. Generate summary statistics and histograms
	"""
	
	def __init__(self, output_dir: str):
		"""
		Initialize the analyzer with the output directory containing transcript calls.
		
		Args:
			output_dir: Directory containing called_transcripts_chr*.csv files
		"""
		self.output_dir = output_dir
		self.all_transcripts = None
		self.called_gene_transcripts = None
		self.joined_transcripts = None
		self.analysis_df = None
		self.stats = {}
		
	def load_transcripts_from_disk(self) -> pd.DataFrame:
		"""
		Load called transcripts from all chromosome files and combine them.
		
		Returns:
			Combined DataFrame of all called transcripts
		"""
		collect_transcripts = []
		for chrom in range(1, 17):  # Chromosomes 1-16
			chrom_file = f'{self.output_dir}/transcripts_calling/called_transcripts_chr{chrom}.csv'
			chrom_transcripts = pd.read_csv(chrom_file)
			collect_transcripts.append(chrom_transcripts)
		
		self.all_transcripts = pd.concat(collect_transcripts, ignore_index=True)
		
		# Filter to only gene-overlapping transcripts
		self.called_gene_transcripts = self.all_transcripts[
			~self.all_transcripts.overlapping_orf_name.isna()
		]

		self.called_gene_transcripts['length'] = self.called_gene_transcripts['end']-\
			self.called_gene_transcripts['start']
		
		return self.called_gene_transcripts
	
	def join_with_reference_datasets(self) -> pd.DataFrame:
		"""
		Join transcript calls with SGD genes and Park TSS datasets.
		
		Returns:
			DataFrame with transcript calls joined to reference data
		"""
		# Import reference datasets
		from src.sgd import read_nondubious_genes_dataset, read_park_TSS_PAS
		
		sgd_genes = read_nondubious_genes_dataset()
		park_tsss = read_park_TSS_PAS()
		
		# Prepare transcript data for joining
		join_df = self.called_gene_transcripts[
			['overlapping_orf_name', 'start', 'end', 'length']
		].set_index('overlapping_orf_name').copy()
		
		join_df = join_df.rename(columns={
			'start': 'rna_start', 
			'end': 'rna_stop'
		})
		
		# Join with SGD genes (right join to include all SGD genes)
		join_df = join_df.join(
			sgd_genes[['chr', 'strand', 'start', 'stop']], 
			how='right'
		)
		
		# Join with Park TSS data
		self.joined_transcripts = join_df.join(
			park_tsss[['Park_TSS']], 
			how='left'
		)
		
		return self.joined_transcripts
	
	def remove_duplicate_transcripts(self) -> pd.DataFrame:
		"""
		Remove duplicate transcript assignments to genes.
		Keep the transcript closest to the gene's start/stop.
		
		Returns:
			DataFrame with duplicates removed
		"""
		from pipeline.transcripts_caller_runner import remove_duplicate_transcripts
		
		# Remove duplicates from transcripts with RNA calls
		transcripts_with_rna = self.joined_transcripts.dropna(subset=['rna_start'])
		deduplicated = remove_duplicate_transcripts(transcripts_with_rna)
		
		# Combine with genes that have no RNA calls
		transcripts_no_rna = self.joined_transcripts[
			self.joined_transcripts.rna_start.isna()
		]
		
		self.joined_transcripts = pd.concat([deduplicated, transcripts_no_rna])
		return self.joined_transcripts
	
	def compute_reference_differences(self) -> pd.DataFrame:
		"""
		Compute differences between RNA calls and reference positions.
		
		Returns:
			DataFrame with computed differences
		"""
		from pipeline.transcripts_caller_runner import calculate_distance
		
		self.analysis_df = self.joined_transcripts.copy()
		
		# Compute RNA vs Park TSS differences
		rna_park_diffs = self.analysis_df.apply(
			lambda row: calculate_distance(
				row,
				start_keys=['rna_start', 'Park_TSS'],
				stop_keys=['rna_stop', 'Park_TSS'],
				compute_absolute=False
			), axis=1
		)
		
		# Compute RNA vs SGD differences  
		rna_sgd_diffs = self.analysis_df.apply(
			lambda row: calculate_distance(
				row,
				start_keys=['rna_start', 'start'],
				stop_keys=['rna_stop', 'stop'],
				compute_absolute=False
			), axis=1
		)
		
		self.analysis_df['rna_Park_difference'] = rna_park_diffs
		self.analysis_df['rna_SGD_differences'] = rna_sgd_diffs
		
		return self.analysis_df
	
	def compute_category_counts(self) -> Dict[str, int]:
		"""
		Compute the number of genes in each update category.
		
		Returns:
			Dictionary with category counts
		"""
		if self.analysis_df is None:
			raise ValueError("Must run compute_reference_differences() first")
		
		# Define selection criteria
		has_park = ~self.analysis_df.Park_TSS.isna()
		no_park = self.analysis_df.Park_TSS.isna()
		has_rna = ~self.analysis_df.rna_start.isna()
		no_rna = self.analysis_df.rna_start.isna()
		
		# Count categories
		self.stats = {
			'total_genes': len(self.analysis_df),
			'genes_with_park': len(self.analysis_df[has_park]),
			'genes_with_park_and_rna': len(self.analysis_df[has_park & has_rna]),
			'genes_with_park_no_rna': len(self.analysis_df[has_park & no_rna]),
			'genes_without_park': len(self.analysis_df[no_park]),
			'genes_without_park_with_rna': len(self.analysis_df[no_park & has_rna]),
			'genes_without_park_no_rna': len(self.analysis_df[no_park & no_rna])
		}
		
		return self.stats
	
	def print_category_summary(self) -> None:
		"""Print a summary of genes in each update category."""
		if not self.stats:
			self.compute_category_counts()
		
		print(f"Total genes (non-dubious): {self.stats['total_genes']}")
		print()
		print(f"Number with Park TSS: {self.stats['genes_with_park']}")
		print(f"  with RNA calls: {self.stats['genes_with_park_and_rna']}")
		print(f"  without RNA calls: {self.stats['genes_with_park_no_rna']}")
		print()
		print(f"Number without Park TSS: {self.stats['genes_without_park']}")
		print(f"  with RNA calls: {self.stats['genes_without_park_with_rna']}")
		print(f"  without RNA calls: {self.stats['genes_without_park_no_rna']}")
		print()
	
	def create_update_histograms(self, bins: Optional[np.ndarray] = None, 
								figsize: Tuple[int, int] = (4, 2),
								xlim: Tuple[int, int] = (-2000, 2000)) -> None:
		"""
		Create histograms showing TSS position differences for genes that will be updated.
		
		Args:
			bins: Histogram bins (default: 100bp bins from -2000 to 2000)
			figsize: Figure size tuple
			xlim: X-axis limits
		"""
		if self.analysis_df is None:
			raise ValueError("Must run compute_reference_differences() first")
		
		# Park TSS updates (genes with both Park TSS and RNA calls)
		has_park_and_rna = (~self.analysis_df.Park_TSS.isna() & 
						   ~self.analysis_df.rna_start.isna())
		park_updates = self.analysis_df.loc[has_park_and_rna]
		
		# SGD TSS updates (genes without Park TSS but with RNA calls)  
		no_park_with_rna = (self.analysis_df.Park_TSS.isna() & 
						   ~self.analysis_df.rna_start.isna())
		sgd_updates = self.analysis_df.loc[no_park_with_rna]
		

		self.sgd_updates = sgd_updates.rna_SGD_differences
		self.park_updates = park_updates.rna_Park_difference

		from src.tss_flowchart import create_gene_analysis_figure, LayoutConfig

		data_with_park = self.park_updates.values
		data_without_park = self.sgd_updates.values

		fig, axes = create_gene_analysis_figure(
		    data_with_park, data_without_park,
		    stats=self.stats
		)

	
	def run_full_analysis(self) -> Dict[str, int]:
		"""
		Run the complete analysis pipeline.
		
		Returns:
			Dictionary with category counts
		"""
		print("Loading transcripts from disk...")
		self.load_transcripts_from_disk()
		
		print("Joining with reference datasets...")
		self.join_with_reference_datasets()
		
		print("Removing duplicate transcripts...")
		self.remove_duplicate_transcripts()
		
		print("Computing reference differences...")
		self.compute_reference_differences()
		
		print("Computing category counts...")
		self.compute_category_counts()
		
		print("Analysis complete!")
		self.print_category_summary()
		
		print("Creating histograms...")
		self.create_update_histograms()
		
		return self.stats




