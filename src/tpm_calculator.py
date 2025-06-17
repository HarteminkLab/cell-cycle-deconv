
from src.utils import print_fl


class TPMGenerator:
	"""
	A class to generate TPM datasets from BAM files for gene expression analysis.
	
	This class encapsulates the workflow from your notebook to process RNA-seq BAM files
	and generate both read counts and TPM values for genes across multiple time points.
	"""
	
	def __init__(self, gene_dataset, output_directory):
		"""
		Initialize the TPM generator.
		
		Parameters:
		-----------
		gene_dataset : pandas.DataFrame
			Gene dataset with ORF information containing gene annotations and lengths
		output_directory : str, default 'datasets'
			Base directory for saving results. TPM files will be saved to <output_directory>/TPM
		"""
		self.transcript_boundaries_sets = gene_dataset
		self.output_directory = output_directory
	
	def process_replicate(self, replicate_bam_df, chroms=range(1, 17), verbose=True):
		"""
		Process a single replicate to generate read counts and TPM values.
		
		Parameters:
		-----------
		replicate_bam_df : pandas.DataFrame
			DataFrame with columns ['replicate', 'time', 'full_path'] containing BAM file paths
		verbose : bool, default True
			Whether to print progress messages
			
		Returns:
		--------
		tuple : (read_counts_df, tpm_df)
			- read_counts_df: DataFrame with read counts for each gene at each time point
			- tpm_df: DataFrame with TPM values for each gene at each time point
		"""
		from src.timer import Timer
		from src.read_bam import read_rna_bam
		from src.transcription import calculate_read_counts, convert_to_TPM_all_times
		
		timer = Timer()
		all_times_read_counts = self.transcript_boundaries_sets[[]].copy()

		for _, row in replicate_bam_df.iterrows():
			time = row.time
			if verbose:
				print_fl(f"Reading BAM file for time {time} minutes")

			rna_reads = read_rna_bam(row.full_path, time, timer, chroms=chroms, log=True)
			orf_reads = calculate_read_counts(self.transcript_boundaries_sets, rna_reads)
			all_times_read_counts.loc[:, time] = orf_reads

			if verbose:
				print_fl(f"Done. {timer.get_time()}")
		
		tpm_values = convert_to_TPM_all_times(all_times_read_counts, self.transcript_boundaries_sets['length'])
		
		return all_times_read_counts, tpm_values
	
	def process_multiple_replicates(self, bam_df, replicate_ids=None, verbose=True,
		chroms=range(1, 17)):
		"""
		Convenience method to process multiple replicates and save all results.
		
		Parameters:
		-----------
		bam_df : pandas.DataFrame
			DataFrame with all BAM file paths for all replicates
		replicate_ids : list, optional
			List of replicate IDs to process. If None, processes all unique replicates
		verbose : bool, default True
			Whether to print progress messages
		"""
		if replicate_ids is None:
			replicate_ids = bam_df['replicate'].unique()
		
		for rep_id in replicate_ids:
			if verbose:
				print_fl(f"\n=== Processing Replicate {rep_id} ===")
			
			replicate_bam = bam_df[bam_df.replicate == rep_id]
			read_counts, tpm_values = self.process_replicate(replicate_bam, verbose=verbose, chroms=chroms)
			self.save_results(read_counts, tpm_values, rep_id)
	
	def save_results(self, read_counts, tpm_values, replicate_id):
		"""
		Save read counts and TPM values to CSV files.
		
		Parameters:
		-----------
		read_counts : pandas.DataFrame
			Read counts DataFrame
		tpm_values : pandas.DataFrame
			TPM values DataFrame
		replicate_id : int or str
			Replicate identifier for filename
		"""
		import os
		
		# Create TPM subdirectory in the output directory
		tpm_dir = f'{self.output_directory}/TPM'
		os.makedirs(tpm_dir, exist_ok=True)
		
		read_counts.to_csv(f'{tpm_dir}/replicate{replicate_id}_gene_expression_counts.csv')
		tpm_values.to_csv(f'{tpm_dir}/replicate{replicate_id}_gene_expression_TPM.csv')
		
		print_fl(f"Saved results for replicate {replicate_id} to {tpm_dir}")

