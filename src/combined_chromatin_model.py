
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.chromatin_model import ChromatinModel
from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver
from src.utils import print_fl


class CombinedChromatinModel:
	"""
	In this class we will construct the combined chromatin deconvolution model
	"""

	def __init__(self, config1, config2):

		# Rename the config such that when plotting with the title
		# the combined model name is used.
		# Config2 will not be used for plotting
		config1.name = f"Combined, $\\alpha$={config1.alpha},{config2.alpha}"
		config2.name = f"Combined, $\\alpha$={config1.alpha},{config2.alpha}"

		self.chrom1_model = ChromatinModel(config1)
		self.chrom2_model = ChromatinModel(config2)

		bin_size = 32, 32
		self.chrom1_model.bin_width = bin_size[0]
		self.chrom2_model.bin_width = bin_size[0]

		self.chrom1_model.bin_height = bin_size[1]
		self.chrom2_model.bin_height = bin_size[1]


	def load_combined_mnase_gene(self, gene_name):
		"""This takes the place of load_mnase_gene, as we don't need the
		replicate parameter anymore"""
		self.chrom1_model.load_mnase_gene(gene_name)
		self.chrom2_model.load_mnase_gene(gene_name)

		# Then setup the deconvolution bin histogram data as G for each
		self.chrom1_model.create_deconvolution_bins()
		self.chrom2_model.create_deconvolution_bins()


	def	setup_deconv_model(self, gamma=0.006, gamma_prime=0):
		from src.helpers import calcH
		from src.model import Model

		chrom1_model = self.chrom1_model
		chrom2_model = self.chrom2_model

		self.gamma = gamma

		# Next we will need to setup the deconvolution model to combine the H
		# and the deconvolution G data
		self.G1 = chrom1_model.G
		self.G2 = chrom2_model.G

		# Combine the two G datasets row-wise
		self.G = np.concatenate([self.G1, self.G2])

		image_shape = self.chrom1_model.deconv_hist_unflattened.shape[1:]
		self.image_shape = image_shape

		# Create the first replicates model and H
		self.deconv1_model = Model(chrom1_model.config, chrom1_model.gene_name, chrom1_model.gamma)
		self.H1, self.H1pos = calcH(chrom1_model.config.intervals_wt1, chrom1_model.timepoints)

		# And the second
		self.deconv2_model = Model(chrom2_model.config, chrom2_model.gene_name, chrom2_model.gamma)
		self.H2, self.H2pos = calcH(chrom2_model.config.intervals_wt1, chrom2_model.timepoints)

		# Combine the H matrices
		self.H = np.concatenate([self.H1, self.H2])

		# Use the deconv1 model for deconvolution
		# We shouldn't need anything from model2 at this point
		self.deconv_model = self.deconv1_model
		self.deconv_model.gamma = self.gamma

		self.solver = ChromatinDeconvolveSolver(self.deconv1_model, self.H, self.G, 
			image_shape=image_shape, wavelet_name='bior4.4', gamma_prime=gamma_prime)
		self.solver.define_deconvolution_problem()

		# For plotting results
		self.chrom1_model.solver = self.solver
		self.chrom2_model.solver = self.solver
		self.chrom1_model.deconv_model = self.deconv_model
		self.chrom2_model.deconv_model = self.deconv_model

		self.found_optimal_success = None
		self.deconvolved_f_value = None
		self.gamma_prime = gamma_prime


	def deconvolve(self, verbose=False, gamma=0.006, gamma_prime=0):

		from src.timer import Timer

		timer = Timer()

		self.setup_deconv_model(gamma, gamma_prime)

		print_fl(f"Deconvolving combined model with gamma={self.gamma}, gamma_prime={gamma_prime}")
		print_fl(f"Deconvolving bin size: {self.chrom1_model.bin_width}x{self.chrom1_model.bin_height}")
		print_fl(f"of G shape: {self.G.shape}")

		self.solver.solve(gamma_value=self.gamma, verbose=verbose)

		self.set_results(self.solver.f.value, 
						  self.solver.rn, self.solver.sn,
						  self.solver.gamma.value)

		print_fl(f"Deconvolved in : {timer.get_time()}")
		print_fl(f"The fitting norm is {self.solver.rn:.2f}, "
			  f"the smoothing norm is: {self.solver.sn:.2f}")


	def deconvolve_find_optimal_gamma(self):
		"""
		Find the optimal gamma value
		"""
		from src.timer import Timer

		from src.find_gamma_chromatin import FindOptimalGammaChromatin

		timer = Timer()

		# Let's stick to no spatial smoothing for now
		self.setup_deconv_model(gamma_prime=0)
		self.find_gamma_chromatin = FindOptimalGammaChromatin(self.solver)
		self.found_optimal_success = self.find_gamma_chromatin.find_optimal(silence=False)
		self.gamma = self.find_gamma_chromatin.gamma

		self.set_results(self.solver.f.value, 
						  self.solver.rn, self.solver.sn,
						  self.solver.gamma.value)

		print_fl(f"Found optimal gamma in: {timer.get_time()}")
		print_fl(f"Find optimal success: {self.found_optimal_success}")
		print_fl(f"The fitting norm is {self.solver.rn:.2f}, "
			  f"the smoothing norm is: {self.solver.sn:.2f}")

	def set_results(self, f, rn, sn, gamma):
		"""Following completion of deconvolution or find gamma deconvolution, we
		will need to set the results to the appropriate fields.

		TODO: This may need to be cleaned up but for now each of the chrom models
		have plotting code individually, so we set the results in each of them
		"""

		self.f = f
		self.rn = rn
		self.sn = sn
		self.gamma = gamma

		self.chrom1_model.deconvolved_f_value = f
		self.chrom1_model.rn = rn
		self.chrom1_model.sn = sn
		self.chrom1_model.gamma = gamma

		self.chrom2_model.deconvolved_f_value = f
		self.chrom2_model.rn = rn
		self.chrom2_model.sn = sn
		self.chrom2_model.gamma = gamma

		# Set the f, rn, and sn from the results to the model
		self.pred_G = np.matmul(self.H, f) 
		tp1 = self.chrom1_model.timepoints
		self.pred_G1 = self.pred_G[0:len(tp1)]
		self.pred_G2 = self.pred_G[len(tp1):]

		self.chrom1_model.compute_ptr()
		self.chrom2_model.compute_ptr()


	def create_deconvolution_plots_abbreviated_flipped(self, ge_model=None, vmax=30):
		"""Create the deconvolution plot defined in chromatin_model.py
		"""

		fig = self.chrom1_model.create_deconvolution_plots_abbreviated_flipped(ge_model=ge_model, vmax=vmax)
		return fig

	def plot_raw_prediction(self, replicate, vmax=20):
		"""Plot the resulting comparison between the raw and predicted data"""

		if replicate == 1:
			title = self.chrom1_model.define_title().replace("Combined", "Combined-Rep.1")
			fig = self.chrom1_model.plot_prediction_comparison(self.pred_G1, title, vmax)
		else:
			title = self.chrom1_model.define_title().replace("Combined", "Combined-Rep.2")
			fig = self.chrom2_model.plot_prediction_comparison(self.pred_G2, title, vmax)

		return fig

	def save_deconvolved_outputs(self, out_dir, index, using_default_flag):

		orf_name = self.deconv_model.orf_name
		gene_name = self.deconv_model.gene_name
		f = self.solver.f.value
		G1 = self.G1
		G2 = self.G2

		f_save_path = f'{out_dir}/{index}_f_{orf_name}_{gene_name}.npy'
		g1_save_path = f'{out_dir}/{index}_g1_{orf_name}_{gene_name}.npy'
		g2_save_path = f'{out_dir}/{index}_g2_{orf_name}_{gene_name}.npy'
		ptr_save_path = f'{out_dir}/{index}_ptr_{orf_name}_{gene_name}.npy'
		meta_save_path = f'{out_dir}/{index}_meta_{orf_name}_{gene_name}.csv'

		#---------- Save to disk -------------

		# Save the f to disk
		np.save(g1_save_path, G1)
		np.save(g2_save_path, G2)

		# Save the f to disk
		np.save(f_save_path, f)

		# Save the ptr to disk
		np.save(ptr_save_path, self.chrom1_model.f_ptrs)

		# Save meta information
		from datetime import datetime
		run_date = datetime.now().strftime("%D")

		df = pd.DataFrame({
			'rn': self.solver.rn, 'sn': self.solver.sn, 
				'gm': self.solver.gamma.value,
			'config1': self.chrom1_model.config.name,
			'config2': self.chrom2_model.config.name,
			'model1_path': self.chrom1_model.config.model_wt1_file,
			'model2_path': self.chrom2_model.config.model_wt1_file,
			'run_date': run_date,
			'replicate': "combined",
			'image_shape': str(self.image_shape),
			'rep1_+1': self.chrom1_model.computed_plus_one,
			'rep1_+2': self.chrom2_model.computed_plus_one
			},
			index=[self.deconv_model.orf_name])
		df.to_csv(meta_save_path, float_format="%.4f")

		print_fl(f"Saved to {g1_save_path}...")
		print_fl(f"Saved to {g2_save_path}...")
		print_fl(f"Saved to {f_save_path}...")
		print_fl(f"Saved to {ptr_save_path}...")
		print_fl(f"Saved to {meta_save_path}...")

def load_chromatin_model_from_disk(gene_name, chromatin_dir, f_only=False):

	from src.config import load_yl_rg1_vst_config

	import os
	import glob
	
	f_pattern = os.path.join(chromatin_dir, f'*_f_*{gene_name}*')
	ptr_pattern = os.path.join(chromatin_dir, f'*_ptr_*{gene_name}*')
	meta_pattern = os.path.join(chromatin_dir, f'*_meta_*{gene_name}*')

	#g_filepath = glob.glob(g_pattern)[0]
	f_filepath = glob.glob(f_pattern)[0]
	ptr_filepath = glob.glob(ptr_pattern)[0]
	meta_filepath = glob.glob(meta_pattern)[0]

	config = load_yl_rg1_vst_config(1)

	f = np.load(f_filepath)

	if f_only:
		return f

	ptr = np.load(ptr_filepath)
	meta_data = pd.read_csv(meta_filepath)
	meta_data = meta_data.iloc[0]

	config = load_yl_rg1_vst_config(1)
	chromatin_model = ChromatinModel(config)

	chromatin_model.load_mnase_gene(gene_name)
	chromatin_model.create_deconvolution_bins()
	chromatin_model.setup_deconv_model()
	chromatin_model.setup_solver()

	chromatin_model.deconvolved_f_value = f
	chromatin_model.f_ptrs = ptr.flatten()

	chromatin_model.solver.rn = meta_data.rn
	chromatin_model.solver.sn = meta_data.sn
	chromatin_model.solver.gamma.value = meta_data.gm

	return chromatin_model
