
import sys
sys.path.append('.')

import numpy as np
from src.utils import print_fl
from src.CombinedReplicationDeconvolution import CombinedReplicationDeconvolution
from src.config import load_default_chrom_configs
import matplotlib.pyplot as plt


def main(chrom, num_epochs, out_dir):
	"""
	Run the combined deconvolution of the replication profile for a chromosome

	Usage:
		python src.SingleReplicateDeconvolutionRunner.py <output> <chrom> <num_epochs>

	Example:
		python src.SingleReplicateDeconvolutionRunner.py output/replication_deconv 1 20

	"""

	# Start runner
	runner = CombinedReplicateDeconvolutionRunner(chrom, out_dir)
	runner.start_runs(num_epochs=num_epochs)
	runner.save_to_disk()


class CombinedReplicateDeconvolutionRunner():
	"""Run the deconvolution and save to disk"""


	def __init__(self, chrom, save_dir):

		self.chrom = chrom
		self.save_dir = save_dir

		print_fl("Loading initial parameters from CLOCCS runs")
		config1, config2 = load_default_chrom_configs()

		print_fl("Shifting mu0, gamma1, and gamma2 for the alpha parameter")
		config1.shift_parameters_for_alpha()
		config2.shift_parameters_for_alpha()

		self.deconvolution = CombinedReplicationDeconvolution(config1, config2, chr=chrom)

	def start_runs(self, num_epochs):

		# Setup runs
		self.deconvolution.setup_deconvolution()

		print("Running initial combined deconvolution for F, N, and B")
		result = self.deconvolution.iterative_deconvolution_updates(20, verbose=False)

		def update_function(epoch, update_params_df, Hs, Fs, Ns, Bs):
			"""Update function that runs for each epoch for progressive
			updating"""

			self.current_epoch = epoch

			if epoch % 10 == 0:
				self.update_params_df = update_params_df
				self.Hs = Hs
				self.Fs = Fs
				self.Ns = Ns
				self.Bs = Bs
				self.save_to_disk()

		# Run the deconvolution updates
		self.update_params_df, self.Hs, self.Fs, self.Ns, self.Bs = \
			self.deconvolution.run_epochs(num_epochs=num_epochs, function_update=update_function)

	def save_to_disk(self):

		from src.utils import mkdirs_safe

		mkdirs_safe([self.save_dir])

		N_save_path = f'{self.save_dir}/combined_chr{self.chrom}_N.npy'
		B_save_path = f'{self.save_dir}/combined_chr{self.chrom}_B.npy'
		F_save_path = f'{self.save_dir}/combined_chr{self.chrom}_F.npy'
		H_save_path = f'{self.save_dir}/combined_chr{self.chrom}_H.npy'
		parameters_save_path = f'{self.save_dir}/combined_chr{self.chrom}_parameters.csv'
		fig_path = f'{self.save_dir}/combined_chr{self.chrom}.png'

		np.save(N_save_path, self.Ns[self.current_epoch])
		np.save(B_save_path, self.Bs[self.current_epoch])
		np.save(H_save_path, self.Hs[self.current_epoch])
		np.save(F_save_path, self.Fs[self.current_epoch])
		self.update_params_df.to_csv(parameters_save_path)

		fig = self.deconvolution.plot_heatmaps()
		plt.suptitle(f"Combined replicate"
			f" deconvolution,\nChromosome {self.chrom}, epoch={self.current_epoch}")
		plt.savefig(fig_path, dpi=150)
		plt.close(fig)

		print_fl(f"Saved to: {N_save_path}")
		print_fl(f"Saved to: {B_save_path}")
		print_fl(f"Saved to: {H_save_path}")
		print_fl(f"Saved to: {F_save_path}")
		print_fl(f"Saved to: {parameters_save_path}")
		print_fl(f"Saved to: {fig_path}")

if __name__ == '__main__':
	main()
