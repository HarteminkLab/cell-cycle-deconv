

import pandas as pd


YL_REP2_MNASE_PATH = 'output/yl_rep2_mnase_reads/yl_rep2_mnase_reads_chr{}.h5'
DATA_KEY = 'mnase_data'

def save_mnase_reads(all_fragments, chrom):
	"""
	Save the mnase reads from disk
	"""
	save_path = YL_REP2_MNASE_PATH.format(chrom)
	with pd.HDFStore(save_path, complevel=9, complib='zlib') as store:
		store[DATA_KEY] = all_fragments
		print(f"Wrote file: {save_path} to disk.")


def load_mnase_reads(chrom):
	"""
	Load the mnase reads from disk
	"""
	save_path = YL_REP2_MNASE_PATH.format(chrom)
	print(f"Retrieving file: {save_path} from disk.")
	with pd.HDFStore(save_path, complevel=9, complib='zlib') as store:
		data_retrieved = store[DATA_KEY]
	return data_retrieved
