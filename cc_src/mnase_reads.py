

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
