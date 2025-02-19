
# Obsolete






# # Main class to compute the chromatin occupancies for all genes for yl2 dataset

# # append to system path for access to src files
# import sys
# sys.path.append('.')

# import pandas as pd
# from src.timer import Timer
# from src.chromatin_metrics import ChromatinMetrics, fragment_lengths_definitions


# def compute_rep2_chrom_entropies():
# 	"""Main script to read through all replicate 2 filenames
# 	collect the gene counts and save to disk"""
		
# 	replicate2_filenames = [(0, '../_archive/data/bam/cell_cycle/mnase/DMAH82_MNase_rep2_0_min.bam'),
# 							 (10, '../_archive/data/bam/cell_cycle/mnase/DMAH83_MNase_rep2_10_min.bam'),
# 							 (20, '../_archive/data/bam/cell_cycle/mnase/DMAH84_MNase_rep2_20_min.bam'),
# 							 (30, '../_archive/data/bam/cell_cycle/mnase/DMAH85_MNase_rep2_30_min.bam'),
# 							 (40, '../_archive/data/bam/cell_cycle/mnase/DMAH86_MNase_rep2_40_min.bam'),
# 							 (50, '../_archive/data/bam/cell_cycle/mnase/DMAH87_MNase_rep2_50_min.bam'),
# 							 (60, '../_archive/data/bam/cell_cycle/mnase/DMAH88_MNase_rep2_60_min.bam'),
# 							 (70, '../_archive/data/bam/cell_cycle/mnase/DMAH89_MNase_rep2_70_min.bam'),
# 							 (80, '../_archive/data/bam/cell_cycle/mnase/DMAH90_MNase_rep2_80_min.bam'),
# 							 (90, '../_archive/data/bam/cell_cycle/mnase/DMAH91_MNase_rep2_90_min.bam'),
# 							 (100, '../_archive/data/bam/cell_cycle/mnase/DMAH92_MNase_rep2_100_min.bam'),
# 							 (110, '../_archive/data/bam/cell_cycle/mnase/DMAH93_MNase_rep2_110_min.bam'),
# 							 (120, '../_archive/data/bam/cell_cycle/mnase/DMAH94_MNase_rep2_120_min.bam'),
# 							 (130, '../_archive/data/bam/cell_cycle/mnase/DMAH95_MNase_rep2_130_min.bam'),
# 							 (140, '../_archive/data/bam/cell_cycle/mnase/DMAH96_MNase_rep2_140_min.bam')]

# 	save_filename = 'output/all_chromatin_gene_entropies_rep2.csv'

# 	# Configure the length spans we are going to collect
# 	small_frag_span, mid_frag_span, nucleosome_len_span = fragment_lengths_definitions()

# 	# Gene counts for all time points
# 	all_gene_entropies = pd.DataFrame()

# 	# Timer initialization
# 	timer = Timer()
# 	timer.start()

# 	# Iterate through each bam file/timepoint
# 	for timepoint, filename in replicate2_filenames:

# 		print(f"Computing the entropy scores for the time point {timepoint}, the filepath is: {filename}")

# 		# Read the bam file
# 		print(f"Reading in the bam file...")
# 		chromatin_metrics = ChromatinMetrics(nucleosome_len_span, mid_frag_span, small_frag_span)
# 		chromatin_metrics.read_bam_file(filename, timepoint)
		
# 		cur_entropies = chromatin_metrics.compute_all_gene_entropies()

# 		# Append the gene counts to the super list for each timepoint
# 		cur_entropies['time'] = timepoint
# 		all_gene_entropies = pd.concat([all_gene_entropies, cur_entropies])
# 		all_gene_entropies.to_csv(save_filename)

# 		print(f"Finished time point {timepoint} - {timer.get_time()}")

# 	print(f"Completed in {timer.get_time()}")

# 	all_gene_entropies.to_csv(save_filename)
# 	print(f"Saved to file: {save_filename}")


# if __name__ == '__main__':
# 	compute_rep2_chrom_entropies()
