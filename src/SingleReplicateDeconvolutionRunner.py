
import sys
sys.path.append('.')

import numpy as np
from src.utils import print_fl
from src.RealDataReplication import RealDataReplicationDeconvolution
from src.rg1_refactor import load_default_chrom_configs


def main():
	"""
	Run the deconvolution of a chromosome

	Usage:
		python src.SingleReplicateDeconvolutionRunner.py <output> <replicate> <chrom> <num_epochs>

	Example:
		python src.SingleReplicateDeconvolutionRunner.py output/replication_deconv 1 1 20

	"""

	system_args = tuple(sys.argv)
	(_, out_dir, replicate, chrom, num_epochs) = system_args
	replicate = int(replicate)
	chrom = int(chrom)
	num_epochs = int(num_epochs)

	print_fl(("Arguments: ", system_args))

	# Start runner
	runner = SingleReplicateDeconvolutionRunner(chrom, replicate, out_dir)
	runner.start_runs(num_epochs=num_epochs)
	runner.save_to_disk()


class SingleReplicateDeconvolutionRunner():
	"""Run the deconvolution on a single replicate and save to disk"""


	def __init__(self, chrom, replicate, save_dir):

		self.chrom = chrom
		self.replicate = replicate
		self.save_dir = save_dir

		print_fl("Loading initial parameters from CLOCCS runs")
		config1, config2 = load_default_chrom_configs()

		config = config1 if replicate == 1 else config2
		print_fl(str(config.params_dic))

		print_fl("Shifting initial parameters for alpha")
		config.shift_parameters_for_alpha()
		print_fl(str(config.params_dic))

		self.deconvolution = RealDataReplicationDeconvolution(config, replicate=replicate, chr=chrom)

	def start_runs(self, num_epochs):

		from src.optimize_H import create_bounds_params_from_config, run_epochs

		# Setup runs
		self.deconvolution.setup_deconvolution(self.deconvolution.config)
		self.deconvolution.iterative_deconvolution_updates(20, verbose=False)

		def update_function(epoch, update_params_df, Hs, Fs, Ns, Bs):
			"""Update function that runs for each epoch for progressive
			updating"""
			if epoch % 10 == 0:
				self.update_params_df = update_params_df
				self.Hs = Hs
				self.Fs = Fs
				self.Ns = Ns
				self.Bs = Bs
				self.save_to_disk()

		# Run
		params_df = create_bounds_params_from_config(self.deconvolution.config)
		self.update_params_df, self.Hs, self.Fs, self.Ns, self.Bs = \
			run_epochs(self.deconvolution, params_df, num_epochs, function_update=update_function)

	def save_to_disk(self):

		from src.utils import mkdirs_safe

		mkdirs_safe([self.save_dir])

		N_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_Ns.npy'
		B_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_Bs.npy'
		F_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_Fs.npy'
		H_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_Hs.npy'
		parameters_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_parameters.csv'

		np.save(N_save_path, self.Ns)
		np.save(B_save_path, self.Bs)
		np.save(H_save_path, self.Hs)
		np.save(F_save_path, self.Fs)
		self.update_params_df.to_csv(parameters_save_path)

		print_fl(f"Saved to: {N_save_path}")
		print_fl(f"Saved to: {B_save_path}")
		print_fl(f"Saved to: {H_save_path}")
		print_fl(f"Saved to: {F_save_path}")
		print_fl(f"Saved to: {parameters_save_path}")


if __name__ == '__main__':
	main()
