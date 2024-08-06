
from src.geneset import get_deconvolved_geneset
import numpy as np
import pandas as pd
import glob

def load_gene_expression_fs(gene_expression_dir, geneset=None):

	file_paths = glob.glob(f'{gene_expression_dir}/*_f_*.npy')
	
	if geneset is None:
		geneset = get_deconvolved_geneset()

	gene_expression_f = None

	# For each deconvolved gene, load the ptr values and place them into the PTRs dataframe
	for path in file_paths:
		filename = path.split('/')[-1]
		orf_name = filename.split('_')[2].replace('.npy', '')

		# Skip genes not in our analysis set
		# for runs in which we haven't filtered for low coverage genes yet
		if not orf_name in geneset.index.values: continue

		loaded_f = np.load(path)

		if gene_expression_f is None:
			m = len(loaded_f)
			gene_expression_f = pd.DataFrame(index=geneset.index, columns=np.arange(m))

		gene_expression_f.loc[orf_name] = loaded_f

	return gene_expression_f


def load_f_files(chromatin_dir, geneset=None):
	"""Load all of the gene F results into a dataframe, flatten the F images for the dataframe."""

	if geneset is None:
		geneset = get_deconvolved_geneset()
	chromatin_dir = chromatin_dir

	f_filepaths = glob.glob(f'{chromatin_dir}/*_f_*.npy')

	# Load the F images for each deconvolved gene
	current_f = np.load(f_filepaths[0])
	old_shape = current_f.shape
	print("Shape of the loaded F:", current_f.shape)
	# Using the old y fragment length definitions
	# Adjust to 11 x 34
	print(current_f.shape)
	if current_f.shape[1] == 340:
		current_f = pad_10_34_f_img(current_f)
		current_f = current_f.reshape((old_shape[0], -1)) # Then flatten rows and columns
		print("Shape of the adjusted F shape:", current_f.shape)

	m_times, u_vals = current_f.shape
	all_gene_fs_df = pd.DataFrame(index=geneset.index, 
		columns=np.arange(m_times*u_vals))

	from src.timer import Timer

	timer = Timer()
	i = 0

	# For each deconvolved gene, load the ptr values and place them into the PTRs dataframe
	for path in f_filepaths:
		filename = path.split('/')[-1]
		orf_name = filename.split('_')[2]

		# Skip genes not in our analysis set
		# for runs in which we haven't filtered for low coverage genes yet
		if not orf_name in geneset.index.values: continue

		current_f = np.load(path)

		# Using the old y fragment length definitions
		# Adjust to 11 x 34
		if current_f.shape[1] == 340:
			current_f = pad_10_34_f_img(current_f)

		all_gene_fs_df.loc[orf_name] = current_f.flatten()
		
		if i % 1000 == 0:
			timer.print_time(f"{i+1}/{len(f_filepaths)}")
		i += 1

	return all_gene_fs_df


def pad_10_34_f_img(current_f):
	"""Old chromatin run has image shape of 10x34, we are now using 11x34 so pad this old dataset with zeros
	to make analysis easier"""
	from src.global_config import GlobalConstants

	f_rshp = current_f.reshape((-1, 10, 34))
	shape = f_rshp.shape

	# Concatenate the original array with the zero array along the second dimension
	zeros = np.zeros((shape[0], 1, shape[2]))
	padded_f_rshp = np.concatenate((f_rshp, zeros), axis=1)
	return padded_f_rshp
