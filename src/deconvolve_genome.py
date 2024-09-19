
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

		python src/deconvolve_genome.py output/deconvolved_genome_g0066_10k_10x10_2024_09_19 0

	"""

	system_args = tuple(sys.argv)
	outdir, index = system_args[1], int(system_args[2])

	genome_deconvolution = GenomeDeconvolution(save_dir=outdir)

	genome_10K_windows = pd.read_csv('data/reference_data/sacCer3_genome_10k_windows.csv')
	current_genomic_span = genome_10K_windows.loc[index]
	chrom, span = current_genomic_span.chr, (current_genomic_span.start, current_genomic_span.end)

	print(f"Deconvolving chr{chrom}, {span[0]}-{span[1]}")
	genome_deconvolution.load_chrom_span(chrom, span)
	genome_deconvolution.combined_model.deconvolve()

	genome_deconvolution.save_to_disk()


if __name__ == '__main__':
	main()

