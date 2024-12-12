
import sys
sys.path.append('.')

import pandas as pd
from src.GenomeDeconvolution import GenomeDeconvolution


def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument

	Usage:

		<output> <index>

	Will run the find optimal gamma procedure on the chromatin. 
	Example script to deconvolve the first 10k window of the genome

		python src/deconvolve_test_windows.py outdir 1 0

		Example test windows: Early, Late, CLB2

		,chr,start,end
		10,194000,204000
		3,260000,270000
		16,770000,780000
	"""

	from src.RealDataReplication import read_n_fr_b, read_no_copy_correction_n_fr_b

	system_args = tuple(sys.argv)
	outdir, copy_correct, index = system_args[1], int(system_args[2]), \
		int(system_args[4])

	genome_deconvolution = GenomeDeconvolution(save_dir=outdir)

	genome_10K_windows = pd.read_csv('data/reference_data/sacCer3_test_10k_windows.csv')

	current_genomic_span = genome_10K_windows.loc[index]
	chrom, span = current_genomic_span.chr, (current_genomic_span.start, current_genomic_span.end)

	genome_deconvolution.load_chrom_span(chrom, span)

	# Replication-related values
	if copy_correct:
		N, fr, b = read_n_fr_b(chrom, span)

	# No copy correction
	else:
		N, fr, b = read_no_copy_correction_n_fr_b(genome_deconvolution.combined_model.H)

	print(f"Deconvolving chr{chrom}, {span[0]}-{span[1]}")
	genome_deconvolution.combined_model.deconvolve(N=N, f_replication=fr, b=b)

	genome_deconvolution.save_to_disk()


if __name__ == '__main__':
	main()

