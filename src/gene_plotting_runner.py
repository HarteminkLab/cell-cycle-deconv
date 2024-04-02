
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.gene_plotter import GeneLocusPlotter
from src.gene_plotter import DeconvolutionPlotter
from src.timer import Timer

# Global variables
timer = Timer()
save_dir_raw = 'output/yl_rep2_gene_loci_2023_10_24/raw'
save_dir_deconv = 'output/yl_rep2_gene_loci_2023_10_24/deconv_gb_nuc_entropy'

deconv_plotter = DeconvolutionPlotter()
locus_plotter = GeneLocusPlotter()


def main():

	# parse arguments
	if len(sys.argv) < 2: 
		print_fl("Usage: python gene_plotting_runner.py <chr>")
		sys.exit(1)

	commands = sys.argv[1].split(';')
	chrom = int(sys.argv[1])

	# -------------------------------------------------------

	count = 0

	print_fl(f"Creating plots for chromosome {chrom}...")
	# Set the current chromosome
	locus_plotter.set_chrom(chrom)
	chrom_genes = locus_plotter.geneset[locus_plotter.geneset.chr == chrom]
	
	# Then create plots for all genes on the chromosome
	for orf_name, gene in chrom_genes.iterrows():    

		print_fl(f"   {gene.gene} / {orf_name} {count}/{len(chrom_genes)}", end="...")
		plot_and_save(locus_plotter, deconv_plotter, save_dir_raw, save_dir_deconv, count, orf_name, gene)

		print_fl(f"...Done. {timer.get_time()}")
		count += 1

	print_fl(f"Completed chromosome {chrom}...{timer.get_time()}")


 # ---------------------- For scripting --------------------------------------------


def plot_and_save(locus_plotter, deconv_plotter, save_dir_raw, save_dir_deconv, index, orf_name, gene):
	"""
	Create the raw and deconvolution plots for a gene and save to disk
	"""

	raw_savepath = f'{save_dir_raw}/{gene.gene}_{orf_name}_raw.png'
	deconv_savepath = f'{save_dir_deconv}/{gene.gene}_{orf_name}_deconv.png'

	locus_plotter.set_gene(orf_name)
	fig = locus_plotter.plot()

	try:
		plt.savefig(raw_savepath, dpi=200)
		plt.cla()
		plt.clf()
		plt.close(fig)
	except ValueError:
		print_fl(f"Error saving {raw_savepath}. Maybe something wrong with the reads in this gene.")

	fig = deconv_plotter.plot_gene(gene.gene)
	plt.savefig(deconv_savepath, dpi=200)
	plt.cla()
	plt.clf()
	plt.close(fig)


def print_fl(val='', end='\n'):

    contents = str(val) + end

    sys.stdout.write(contents)
    sys.stdout.flush()

 # ---------------------- End function definitions --------------------------------------------

if __name__ == '__main__':
	main()
