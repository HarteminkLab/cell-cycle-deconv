
import sys
sys.path.append('.')

import pandas as pd
from src.GenomeDeconvolution import GenomeDeconvolution


def generate_random_100_windows():
	"""Generate random 100 1kb windows and save to disk"""
	pass


def main():
	"""
	Run deconvolution on random set of 1k windows to find the optimal gamma value

	Usage:

		<output> <index>

	Will run the find optimal gamma procedure on the chromatin. 
	Example script to deconvolve the first 10k window of the genome

		python src/deconvolve_genome.py output/deconvolved_genome_g0066_10k_10x10_2024_09_19 0

	"""

	system_args = tuple(sys.argv)
	outdir, index = system_args[1], int(system_args[2])

	genome_deconvolution = GenomeDeconvolution(save_dir=outdir)

	genome_random_100_windows = pd.read_csv('data/reference_data/saccer3_genome_random_1k_windows.csv')
	current_genomic_span = genome_random_100_windows.loc[index]
	chrom, span = current_genomic_span.chrom, (current_genomic_span.start, current_genomic_span.end)

	print(f"[{index}] Deconvolving chr{chrom}, {span[0]}-{span[1]}")
	genome_deconvolution.load_chrom_span(chrom, span)
	genome_deconvolution.combined_model.deconvolve_find_optimal_gamma()

	genome_deconvolution.save_to_disk()
	genome_deconvolution.save_gamma_to_disk()


if __name__ == '__main__':
	main()

