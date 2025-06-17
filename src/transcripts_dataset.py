
from src.sgd import read_nondubious_genes_dataset
import pandas as pd
import numpy as np


class TranscriptDatasetBuilder:
	"""
	A class to build comprehensive transcript datasets with updated TSS annotations
	and genomic region definitions for downstream analysis.
	"""
	
	def __init__(self, output_dir, max_tss_change=500, min_nongene_transcript_len=200):
		"""
		Initialize the TranscriptDatasetBuilder.
		
		Parameters:
		-----------
		output_dir : str
			Directory containing the transcript calling results
		max_tss_change : int, default=500
			Maximum allowed distance (bp) for TSS re-annotation from original call
		min_nongene_transcript_len : int, default=200
			Minimum length (bp) for non-genic transcripts to be included
		"""
		self.output_dir = output_dir
		self.max_tss_change = max_tss_change
		self.min_nongene_transcript_len = min_nongene_transcript_len
		
		# Data storage attributes
		self.gene_associated_transcripts = None
		self.nongenic_transcripts = None
		self.rna_called_tss_data = None
		self.filtered_nongenic_transcripts = None
		
	def load_transcripts(self):
		"""Load transcript data from all chromosomes and separate genic vs non-genic."""
		print("Loading transcripts from all chromosomes...")
		
		# Load the transcript boundary results from disk from each chromosome
		dfs_arr = []
		for chrom in range(1, 17):
			chrom_transcripts_df = pd.read_csv(f'{self.output_dir}/transcripts_calling/called_transcripts_chr{chrom}.csv')
			dfs_arr.append(chrom_transcripts_df)

		all_called_transcripts_df = pd.concat(dfs_arr)
		all_called_transcripts_df = all_called_transcripts_df.sort_values(['chromosome', 'start'])

		# Separate genic and non-genic transcripts
		self.nongenic_transcripts = all_called_transcripts_df[all_called_transcripts_df.overlapping_gene.isna()].copy()
		self.gene_associated_transcripts = all_called_transcripts_df[~all_called_transcripts_df.overlapping_gene.isna()].copy()

		# Name the non-gene transcripts
		nongenic_names = 'nogene_chr' + self.nongenic_transcripts['chromosome'].astype(str) + \
			'_' + self.nongenic_transcripts['start'].astype(str) + '_' + \
			self.nongenic_transcripts['end'].astype(str)
		self.nongenic_transcripts.index = nongenic_names
		self.nongenic_transcripts = self.nongenic_transcripts[['chromosome', 'strand', 'start', 'end', 'length']]

		# Filter the nongenic transcripts by length
		self.filtered_nongenic_transcripts = self.nongenic_transcripts[self.nongenic_transcripts['length'] > \
			self.min_nongene_transcript_len]
		
		print(f"Loaded {len(self.gene_associated_transcripts)} gene-associated transcripts")
		print(f"Loaded {len(self.nongenic_transcripts)} non-genic transcripts")
		
	def update_sgd_tss_annotations(self):
		"""
		Update TSS annotations with RNA-seq called TSSes, maintaining priority:
		RNA TSS (if within limits) > park_TSS > start/stop position
		"""
		from src.sgd import read_nondubious_genes_dataset, read_park_TSS_PAS

		print("Updating TSS annotations with RNA-seq calls...")
		
		# Call RNA-seq TSSes and store for comparison
		self.rna_called_tss_data = call_all_chromosome_TSSes(self.gene_associated_transcripts)

		self.tss_comparison = compare_all_chromosomes_TSS(self.rna_called_tss_data, 'Park_TSS',
													threshold=self.max_tss_change, plot=False)
		filtered_genic_tss = self.tss_comparison[np.abs(self.tss_comparison['tss_difference']) < self.max_tss_change]

		filtered_rna_TSSes = filtered_genic_tss.rna_TSS

		# Load the sgd genes and add the rna called TSSes
		new_sgd_genes = read_nondubious_genes_dataset()
		new_sgd_genes.loc[filtered_rna_TSSes.index, 'rna_called_TSS'] = filtered_rna_TSSes

		# Include the Park annotations
		park_TSS_PAS_genes = read_park_TSS_PAS()
		new_sgd_genes = new_sgd_genes.join(park_TSS_PAS_genes[['Park_TSS', 'Park_PAS']])

		# Add TSS and PAS columns
		new_sgd_genes = create_tss_column(new_sgd_genes)
		new_sgd_genes = create_pas_column(new_sgd_genes)

		# Then convert columns to ints
		for c in ['TSS', 'PAS']:
			new_sgd_genes[c] = new_sgd_genes[c].astype(int)

		from src.sgd_dataset_helpers import define_genomic_region

		new_sgd_genes = define_genomic_region(new_sgd_genes, 'TSS', (-300, 0),
		                     ['promoter_start', 'promoter_end'])
		new_sgd_genes = define_genomic_region(new_sgd_genes, 'TSS', (0, 500),
		                     ['gene_body_start', 'gene_body_end'])

		self.updated_sgd_genes = new_sgd_genes

	def define_nongenic_dataset(self):

		from src.sgd_dataset_helpers import define_new_strand_specific_key
		from src.sgd_dataset_helpers import define_genomic_region

		# Next, create non-genic transcripts dataset, can we meerge it with the updated sgd genes?
		# maybe with a lot of null columns
		nongenic_transcripts = self.filtered_nongenic_transcripts.copy()

		nongenic_transcripts = define_new_strand_specific_key(nongenic_transcripts, 'TSS', 
		    'start', 'end')
		nongenic_transcripts = define_new_strand_specific_key(nongenic_transcripts, 'PAS', 
		    'end', 'start')

		nongenic_transcripts = define_genomic_region(nongenic_transcripts, 'TSS', (-300, 0),
		                     ['promoter_start', 'promoter_end'])
		nongenic_transcripts = define_genomic_region(nongenic_transcripts, 'TSS', (0, 500),
		                     ['transcript_body_start', 'transcript_body_end'])
		nongenic_transcripts.index.name = 'transcript_name'
		nongenic_transcripts = nongenic_transcripts.rename(columns={'chromosome': 'chr',
			'end': 'stop'})

		self.nongenic_transcripts_dataset = nongenic_transcripts

	def save_results(self):
		"""Save the updated transcripts data frames to disk"""

		save_dir = f"{self.output_dir}/transcripts_calling"

		# Save the updated sgd genes to disk
		self.updated_sgd_genes.to_csv(f"{save_dir}/updated_transcripts_geneset.csv")
		self.nongenic_transcripts_dataset.to_csv(f"{save_dir}/nongenic_transcripts_set.csv")


def create_genomic_coordinate_column(df, column_name, priority_list):
	"""
	Create a genomic coordinate column (TSS, PAS, etc.) using a customizable priority list
	
	Parameters:
	-----------
	df : pandas.DataFrame
		Gene dataset with columns including strand, start, stop, and any coordinate columns
	column_name : str
		Name of the column to create (e.g., 'TSS', 'PAS')
	priority_list : list
		List of column names or special keywords in priority order.
		Special keywords:
		- 'strand_specific_start': + strand uses 'start', - strand uses 'stop'
		- 'strand_specific_stop': + strand uses 'stop', - strand uses 'start'
		
	Returns:
	--------
	pandas.DataFrame
		DataFrame with new column added
	"""
	# Make a copy to avoid modifying the original
	df_copy = df.copy()
	
	# Initialize column with NaN
	df_copy[column_name] = np.nan
	
	def apply_strand_specific_logic(df, keyword, mask):
		"""Apply strand-specific logic for special keywords"""
		if keyword == 'strand_specific_start':
			# + strand uses start, - strand uses stop
			mask_plus = mask & (df['strand'] == '+')
			mask_minus = mask & (df['strand'] == '-')
			df.loc[mask_plus, column_name] = df.loc[mask_plus, 'start']
			df.loc[mask_minus, column_name] = df.loc[mask_minus, 'stop']
		elif keyword == 'strand_specific_stop':
			# + strand uses stop, - strand uses start
			mask_plus = mask & (df['strand'] == '+')
			mask_minus = mask & (df['strand'] == '-')
			df.loc[mask_plus, column_name] = df.loc[mask_plus, 'stop']
			df.loc[mask_minus, column_name] = df.loc[mask_minus, 'start']
	
	# Apply priorities in order
	for priority_source in priority_list:
		# Find genes that still need coordinate assignment
		mask_remaining = df_copy[column_name].isna()
		
		if not mask_remaining.any():
			# All genes have been assigned, no need to continue
			break
			
		if priority_source in ['strand_specific_start', 'strand_specific_stop']:
			# Handle special strand-specific keywords
			apply_strand_specific_logic(df_copy, priority_source, mask_remaining)
		else:
			# Handle regular column names
			if priority_source in df_copy.columns:
				mask_available = mask_remaining & df_copy[priority_source].notna()
				df_copy.loc[mask_available, column_name] = df_copy.loc[mask_available, priority_source]
			else:
				print(f"Warning: Column '{priority_source}' not found in dataset")
	
	return df_copy


def create_tss_column(df, priority_list=['rna_called_TSS', 'Park_TSS', 'strand_specific_start']):
	"""
	Create a TSS column using a customizable priority list
	"""
	return create_genomic_coordinate_column(df, 'TSS', priority_list)


def create_pas_column(df, priority_list=['Park_PAS', 'strand_specific_stop']):
	"""
	Create a PAS (Polyadenylation Site) column using a customizable priority list
	"""
	return create_genomic_coordinate_column(df, 'PAS', priority_list)


def load_all_transcripts_from_runs(output_dir):
	# Load the transcript boundary results from disk from each chromosome
	dfs_arr = []
	for chrom in range(1, 17):
		chrom_transcripts_df = pd.read_csv(f'{output_dir}/antisense_calling/called_transcripts_chr{chrom}.csv')
		dfs_arr.append(chrom_transcripts_df)

	all_called_transcripts_df = pd.concat(dfs_arr)
	all_called_transcripts_df = all_called_transcripts_df.sort_values(['chromosome', 'start'])

	nongenic_transcripts = all_called_transcripts_df[all_called_transcripts_df.overlapping_gene.isna()].copy()
	gene_associated_transcripts = all_called_transcripts_df[~all_called_transcripts_df.overlapping_gene.isna()].copy()

	# Name the non-gene transcripts
	nongeneic_names = 'nogene_chr' + nongenic_transcripts['chromosome'].astype(str) + \
		'_' + nongenic_transcripts['start'].astype(str) + '_' + \
		nongenic_transcripts['end'].astype(str)
	nongenic_transcripts.index = nongeneic_names
	nongenic_transcripts = nongenic_transcripts[['chromosome', 'strand', 'start', 'end', 'length']]

	return gene_associated_transcripts, nongenic_transcripts



def call_all_chromosome_TSSes(caller_results_df):

	called_arr = []
	for chrom in range(1, 17):
		chrom_results = caller_results_df[caller_results_df.chromosome == chrom]
		called_results = call_chromosome_TSSes(chrom_results, chrom)
		called_arr.append(called_results)

	all_chroms_df = pd.concat(called_arr)

	return all_chroms_df


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

def compare_all_chromosomes_TSS(combined_rna_TSSes, reference_tss_column='TSS', threshold=500,
	plot=False):
	"""
	Compare RNA-seq called TSSes across all chromosomes with existing annotations.
	
	Parameters:
	caller_results_dict: dict, {chromosome: AntisenseTranscriptCaller.results_df}
	reference_tss_column: str, 'TSS' or 'park_TSS'
	
	Returns:
	DataFrame with all TSS comparisons and creates histogram
	"""

	import matplotlib.pyplot as plt
	from src.sgd import read_park_TSS_PAS
	
	# Get SGD gene data
	park_genes = read_park_TSS_PAS()
	
	# Join with RNA-seq TSSes
	tss_comparison = park_genes.join(combined_rna_TSSes)[[reference_tss_column, 'rna_TSS']]
	#tss_comparison = tss_comparison.dropna()  # Remove genes without RNA calls
	
	# Calculate differences
	tss_comparison['tss_difference'] = tss_comparison['rna_TSS']-\
		tss_comparison[reference_tss_column]
	
	# Negate crick strand, such that differences are 5' oriented
	crick_selection = park_genes.strand == '-'
	tss_comparison[crick_selection] = -tss_comparison[crick_selection]
	
	# Create histogram
	if plot:
		plt.figure(figsize=(4, 2))
		plt.hist(tss_comparison['tss_difference'], bins=100, alpha=0.7, edgecolor='black')
		plt.xlabel(f'Called RNA-seq TSS - Park TSS, bp (strand corrected)')
		plt.ylabel('Count')
		plt.title(f'Change in TSS calls using stranded RNA-seq', fontweight='demi',
				 fontsize=12)
		plt.xlim(-1000, 1000)
		plt.axvline(threshold, c='red', alpha=0.75)
		plt.axvline(-threshold, c='red', alpha=0.75)
	
	# Add summary statistics
	median_diff = tss_comparison['tss_difference'].median()
	mean_diff = tss_comparison['tss_difference'].mean()
	
	print(f"Total genes with RNA-seq TSS calls: {len(tss_comparison)}")
	print(f"Median difference: {median_diff:.1f}bp")
	print(f"Mean difference: {mean_diff:.1f}bp")
	print(f"Std deviation: {tss_comparison['tss_difference'].std():.1f}bp")
	
	return tss_comparison

def load_transcripts_sets(output_dir, combined=False):
    geneset = pd.read_csv(f"{output_dir}/transcripts_calling/updated_transcripts_geneset.csv")
    nongenic_set = pd.read_csv(f"{output_dir}/transcripts_calling/nongenic_transcripts_set.csv")
    geneset = geneset.set_index('orf_name')
    nongenic_set = nongenic_set.set_index('transcript_name')

    # Combined dataset with common keys
    if combined:
		preserve_keys = ['chr', 'strand', 'start', 'stop', 'length']
		combined_gene_nongenic = pd.concat([geneset[preserve_keys], nongenic_set[preserve_keys]])
		return combined_gene_nongenic
    
    return geneset, nongenic_set

