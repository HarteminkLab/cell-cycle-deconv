
import sys
sys.path.append('.')

import sys
from src.utils import mkdirs_safe, parse_bool, print_fl


def main():

	system_args = tuple(sys.argv)
	command = system_args[1]

	print(f"Pipeline arguments: " , system_args)

	# 1. Deconvolve individual replication profiles, learn cell cycle parameters from MNase-seq
	if command == 'replication':

		(_, command, output_directory, replicate, chrom, num_epochs, cold_start) = system_args

		mkdirs_safe([output_directory])

		cold_start = parse_bool(cold_start)
		chrom = int(chrom)
		replicate = int(replicate)
		num_epochs = int(num_epochs)

		# 1. Compute replication profiles for each replicate using chromosome 4
		from pipeline.fit_replication_profiles import main as fit_replication_profile
		fit_replication_profile(chrom=chrom, replicate=replicate, num_epochs=num_epochs, 
			output_directory=output_directory, from_CLOCCS=cold_start)


	# 2. Compute combined replication profiles for all chromosomes
	elif command == 'combined_replication':

		print_fl(f"Generating replication profiles for all chromosomes")
		(_, command, output_directory) = system_args

		from src.CombinedReplicateDeconvolutionRunner import main as fit_combined_replication
		from src.CombinedReplicationDeconvolution import load_config_from_replication_runs

		# Load the replicate 1 and 2 configs from the individual runs, these should
		# have been fit on chromosome 4
		config1, config2 = load_config_from_replication_runs(output_directory, chrom=4)

		# For each chromosome, create the replication profiles for each of the chromosomes and save to disk
		for chrom in range(1, 17):
			print_fl(f"Chromosome {chrom}")
			combined_runner = fit_combined_replication(chrom, 1, output_directory, 4, config1, config2)

	# 3. Deconvolve the gene expression for all genes
	elif command == 'deconvolve_expression':

		from src.geneset import get_deconvolved_geneset

		genes = get_deconvolved_geneset()

		print_fl(f"Deconvolve gene expression for all genes")
		(_, command, output_directory) = system_args

		from src.timer import Timer

		timer = Timer()

		output_directory = 'output/prototype_pipeline_subset/'
		save_genes_directory = f"{output_directory}/genes_deconvolution/"
		mkdirs_safe([save_genes_directory])

		index = 0
		for _, gene in genes.iterrows():

			index += 1

			gene_name = gene['gene']

			print(f"[{index}/{len(genes)}] Deconvolving {gene_name}", end="...")
			
			try: 
				expression_find_gamma = runner.deconvolve_gene(gene_name)
			except:
				print(f"  Failed. Skipping.")
				continue

			runner.save_to_disk(save_genes_directory)
			print(f"Done. {timer.get_time()}")

	else:
		raise ValueError(f"Invalid command" + command)

	# Fit the cell cycle parameters using the previous individual replicate fits
	# as initial parameters
	# fit_combined_replication_profile(chrom=4, num_epochs=1)

	# Generate replication profiles for all chromosomes
	# generate_replication_profiles()





if __name__ == '__main__':
	main()
