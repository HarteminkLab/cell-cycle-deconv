
import os
import pysam
import pandas as pd


def read_mnase_bam(filename, sample=None, timer=None, chroms=list(range(1, 17)),
	log=False):
	"""
	Read mnase data from bam file. Return a pandas dataframe of x start coordinate, 
	x end coordinate, chromosome, fragment length, and sequence. BAM File
	"""
	
	import pysam

	samfile = pysam.AlignmentFile(filename, "rb")

	count = 0
	data = {'start':[], 'length': [], 'stop': [],
			'mid': [], 'chr': [], 'sample': []}

	for chrom in chroms:

		if log:
			print(f"Chromosome {chrom} - {timer.get_time()}")

		# get chromosome reads
		try:
			itr = samfile.fetch(str(chrom))
		except ValueError:
			itr = samfile.fetch("chr{}".format(_toRoman(chrom)))

		for read in itr:

			# skip second read in pair
			# equivalent to filtering to include only "-f 32" 
			# flag in samtools
			if not read.mate_is_reverse: continue

			length = read.template_length
			start = read.pos+1
			stop = start + length - 1
			count += 1

			data['start'].append(start)
			data['length'].append(length)
			data['stop'].append(stop)
			data['mid'].append(start + length//2)
			data['chr'].append(chrom)
			data['sample'].append(sample)

	samfile.close()
	df = pd.DataFrame(data=data)

	return df



def read_rna_bam(filename, sample=None, timer=None, chroms=range(1, 17), log=False):
	"""Load an individual RNA-seq file and return a dataframe"""

	samfile = pysam.AlignmentFile(filename, "rb")

	data = {'start':[], 'strand': [], 'length': [],
			'chr': [], 'stop': [], 'sample': []}
	for chrom in chroms:

		if log:
			print(f"{chrom}", end="..")

		# get chromosome reads
		try:
			itr = samfile.fetch(str(chrom))
		except ValueError:
			itr = samfile.fetch("chr{}".format(_toRoman(chrom)))

		for read in itr:

			# skip unmapped reads, i.e. -F 4
			if read.is_unmapped: continue

			length = read.reference_length
			position = read.pos+1 # first base begins at 1
			strand = '-'
			if read.is_reverse: strand = '+'

			data['start'].append(position)
			data['strand'].append(strand)
			data['chr'].append(chrom)
			data['sample'].append(sample)
			data['length'].append(length)
			data['stop'].append(position + length) # inclusive stop nucleotide

	samfile.close()
	df = pd.DataFrame(data=data)

	return df


def _toRoman(number):
	"""
	Convert number to roman numeral
	"""
	try:
		return {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII',
		 8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII', 13: 'XIII', 
		 14: 'XIV', 15: 'XV', 16: 'XVI'}[number]
	except KeyError:
		return -1
	

def _fromRoman(roman):
	"""
	Convert Roman numeral to number
	"""


	roman = roman.replace('chr', '')

	try:
		return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, 
			"VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, 
			"XI": 11, "XII": 12, "XIII": 13, "XIV": 14, 
			"XV": 15, "XVI": 16}[roman]
	except KeyError:
		return -1


def get_rna_seq_filepaths_df(on_cluster=False):
	# First make a dataframe that contains the metadata and filepaths
	# for the RNA-seq data. This will make things convenient
	# for when we want to read from disk

	if on_cluster:
		parent_directory = '/usr/xtmp/tqtran/data/cell_cycle/rna'
	else:
		parent_directory = '/Users/trung/Research/_archive/data/bam/cell_cycle/rna'

	path1 = f'{parent_directory}/replicate_1/'
	path2 = f'{parent_directory}/replicate_2/'

	rows = []

	rep1_ls = os.listdir(path1)
	rep2_ls = os.listdir(path2)

	for filename in rep1_ls:
		if filename.endswith('bam'):
			fil_spl = filename.split('_')
			row = {'replicate': fil_spl[2].replace('rep', ''), 
				   'time': fil_spl[3], 'full_path': path1 + filename}
			rows.append(row)
			
	for filename in rep2_ls:
		if filename.endswith('bam'):
			fil_spl = filename.split('_')
			row = {'replicate': fil_spl[2].replace('rep', ''), 
				   'time': fil_spl[3], 'full_path': path2 + filename}
			rows.append(row)

	bam_df = pd.DataFrame.from_records(rows)
	bam_df['time'] = bam_df['time'].astype(int)
	bam_df['replicate'] = bam_df['replicate'].astype(int)
	bam_df = bam_df.sort_values(['replicate', 'time'])
	return bam_df.reset_index(drop=True)
