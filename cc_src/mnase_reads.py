

import pandas as pd


YL_MNASE_PATH = 'output/mnase/yl_rep{}_mnase_reads/yl_rep{}_mnase_reads_chr{}.h5'
DATA_KEY = 'mnase_data'


def filter_reads(mnase_reads, start, end, chromosome=None, sample=None, len_span=None):

	selected_reads = mnase_reads

	if chromosome is not None:
		selected_reads = selected_reads[selected_reads.chr == chromosome]

	if sample is not None:
		selected_reads = selected_reads[selected_reads['sample'] == sample]

	if len_span is not None:
		selected_reads = selected_reads[(selected_reads['length'] > len_span[0]) & 
										(selected_reads['length'] < len_span[1])]

	selected_reads = selected_reads[(selected_reads.mid > start) & 
									(selected_reads.mid < end)]

	return selected_reads


def save_mnase_reads(all_fragments, chrom, replicate):
	"""
	Save the mnase reads from disk
	"""
	save_path = YL_MNASE_PATH.format(replicate, replicate, chrom)
	with pd.HDFStore(save_path, complevel=9, complib='zlib') as store:
		store[DATA_KEY] = all_fragments
		print(f"Wrote file: {save_path} to disk.")


def load_mnase_reads(chrom, replicate):
	"""
	Load the mnase reads from disk
	"""
	save_path = YL_MNASE_PATH.format(replicate, replicate, chrom)
	print(f"Retrieving file: {save_path} from disk.")
	with pd.HDFStore(save_path, complevel=9, complib='zlib') as store:
		data_retrieved = store[DATA_KEY]
	return data_retrieved



def save_gene_chrom_reads(replicate_filenames, replicate):

	from src.timer import Timer
	from cc_src.read_bam import read_mnase_bam

	# Gene counts for all time points
	all_fragments = pd.DataFrame()
	chroms = range(1, 17)

	# Timer initialization
	timer = Timer()
	timer.start()

	for chrom in range(1, 17):

		print(f"Creating MNase-seq file for chromosome {chrom}")
		chrom_reads = pd.DataFrame()

		# Iterate through each bam file/timepoint
		for timepoint, filename in replicate_filenames:

			print(f"Load MNase-seq for time point {timepoint}, the filepath is: {filename}")

			mnase_reads = read_mnase_bam(filename, sample=timepoint, timer=timer,
										 chroms=[chrom])
			chrom_reads = pd.concat([chrom_reads, mnase_reads])

		save_mnase_reads(chrom_reads, chrom, replicate)
		print(f"\nDone...Elapsed time: {timer.get_time()}")

	print(f"Completed in {timer.get_time()}")

