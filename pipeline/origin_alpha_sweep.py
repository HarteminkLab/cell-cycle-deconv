import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from src.config import load_default_chrom_configs
from src.combined_chromatin_model import CombinedChromatinModel

class OriginAlphaSweep:
	"""
	Class for analyzing origins of replication with alpha parameter sweeps.
	This class handles the analysis of replication origins, focusing on efficiency
	and peak replication timing, as well as examining chromatin footprints.
	"""
	
	def __init__(self, output_dir: str = 'output/draft1_run/'):
		"""
		Initialize the OriginAlphaSweep class.
		
		Args:
			output_dir: Directory containing replication timing data
		"""
		self.output_dir = output_dir
		self.replication_timing_loader = None
		self.origins = None
		self.config1_alpha_adjusted = None
		self.config2_alpha_adjusted = None

		self.config1, self.config2 = load_default_chrom_configs()
			
		self.combined_model = CombinedChromatinModel(
			config1=self.config1,
			config2=self.config2
		)

		
	def load_data(self) -> None:
		"""
		Load required data for analysis.
		"""
		# Import necessary modules
		from src.replication_timing import ReplicationTiming
		from src.origins import load_origins
		
		
		# Load replication timing data
		self.replication_timing_loader = ReplicationTiming(output_dir=self.output_dir)
		self.replication_timing_loader.compute_peak_annotations()
		
		# Load origins data
		self.origins = load_origins()

		
	def adjust_alpha_configs(self, alpha1: int = 14, alpha2: int = 12) -> None:
		"""
		Adjust alpha parameters in chromosome configurations.
		
		Args:
			alpha1: Alpha value for config1
			alpha2: Alpha value for config2
		"""
		# from src.config import load_default_chrom_configs
		
		# # Load and modify configurations with adjusted alpha values
		# self.config1_alpha_adjusted, self.config2_alpha_adjusted = load_default_chrom_configs()
		# self.config1_alpha_adjusted.modify_alpha(alpha1)
		# self.config2_alpha_adjusted.modify_alpha(alpha2)
		
		# # Combine H values
		# self.modified_H = np.concatenate([
		# 	self.config1_alpha_adjusted.H, 
		# 	self.config2_alpha_adjusted.H
		# ])

		# todo, for the deconvolution sweep, we'll modify the config with trial alpha values
		pass
		

	def find_efficient_origins(self, top_n: int = 30) -> pd.DataFrame:
		"""
		Find the most efficient origins at peak replication timing locations.
		
		Args:
			top_n: Number of top origins to consider
			
		Returns:
			DataFrame containing filtered efficient origins
		"""
		if self.origins is None or self.replication_timing_loader is None:
			self.load_data()
			
		eff_key = 'derived_origin_efficiency_from_mcguffee_et_al_2013'
		efficient_origins = []
		
		for oridb, origin in self.origins.sort_values(
			eff_key, ascending=False).head(top_n).iterrows():
			
			entry = self.replication_timing_loader.load_replication_entry_for(
				origin.chr, origin.pos)
			
			if entry.peak_type:
				efficient_origins.append({
					'oridb': oridb,
					'replication_time': entry.replication_time,
					'peak_type': entry.peak_type,
					'activation_time': origin.activation_time,
					'efficiency': origin[eff_key],
					'chr': origin.chr,
					'pos': origin.pos,
					'strand': origin.strand
				})
		
		return pd.DataFrame(efficient_origins)
	
	def print_efficient_origins(self, top_n: int = 30) -> None:
		"""
		Print the most efficient origins at peak replication timing locations.
		
		Args:
			top_n: Number of top origins to consider
		"""
		efficient_origins = self.find_efficient_origins(top_n)
		
		for _, row in efficient_origins.iterrows():
			print(f"{row['oridb']} {row['replication_time']} {row['peak_type']} "
				  f"{row['activation_time']} {row['efficiency']}")
	
	def load_origin_mnase_data(self, oridb: str, window: int = 1000) -> None:
		"""
		Load MNase data for a specific origin.
		
		Args:
			oridb: Origin ID to analyze
			window: Window size around origin position
		"""
		if self.origins is None:
			self.load_data()
			
		origin = self.origins.loc[oridb]
		chrom = origin.chr
		win_2 = window // 2
		mnase_span = (origin.pos - win_2, origin.pos + win_2 + 1)
		
		self.combined_model.load_mnase_span(chrom, mnase_span)
		self.current_origin = origin
			
		
	def plot_raw_data(self, vmax: int = 20) -> Tuple:
		"""
		Plot raw chromatin data for the loaded origin.
		
		Args:
			vmax: Maximum value for colormap
			
		Returns:
			Tuple of figure objects
		"""
		if self.combined_model is None:
			raise ValueError("Combined model not initialized. Call load_origin_mnase_data first.")
			
		fig1 = self.combined_model.chrom1_model.plot_raw_data(figsize=(2, 7), vmax=vmax)
		fig2 = self.combined_model.chrom2_model.plot_raw_data(figsize=(2, 7), vmax=vmax)
		
		return fig1, fig2
	
	def select_footprint_boundary(self, footprint_bounds: List[int] = [-5, 8, 3, 10],
								 plot: bool = True) -> np.ndarray:
		"""
		Select and visualize the footprint boundary for the current origin.
		
		Args:
			footprint_bounds: Boundaries for the footprint [left, right, bottom, top]
			plot: Whether to generate visualization plots
			
		Returns:
			NumPy array of selected footprint images
		"""
		if self.combined_model is None or not hasattr(self, 'current_origin'):
			raise ValueError("Origin data not loaded. Call load_origin_mnase_data first.")
			
		from src.plot_helpers import plot_rect2
		
		G_imgs = self.combined_model.G.reshape((self.combined_model.G.shape[0], 26, -1))
		mid_bin = G_imgs.shape[2] // 2
		
		if self.current_origin.strand == '+':
			footprint_hspan = mid_bin + footprint_bounds[0], mid_bin + footprint_bounds[1]
		else:
			footprint_hspan = mid_bin - footprint_bounds[1], mid_bin - footprint_bounds[0]
		
		footprint_vspan = footprint_bounds[2], footprint_bounds[3]
		
		origin_selection_imgs = G_imgs[:, 
									  footprint_vspan[0]:footprint_vspan[1],
									  footprint_hspan[0]:footprint_hspan[1]]
		
		if plot:
			plt.figure(figsize=(11, 3))
			
			def plot_im(ind):
				plt.imshow(G_imgs[ind], cmap='magma_r', origin='lower', vmax=20,
						  aspect='auto')
				plt.xticks([])
				plt.yticks([])
				ax = plt.gca()
				plot_rect2(ax, footprint_hspan[0], footprint_vspan[0], 
						  footprint_hspan[1], footprint_vspan[1], color='blue', 
						  fill=False, lw=1, zorder=10.)
				plt.ylabel(f"{ind}")
			
			plt.subplot(2, 3, 1)
			plot_im(0)
			plt.title("Origin window")
			
			plt.subplot(2, 3, 4)
			plot_im(5)
			
			plt.subplot(2, 3, 2)
			plt.imshow(origin_selection_imgs[5], cmap='magma_r', origin='lower', vmax=20,
					  aspect='auto')
			plt.title("Selected footprint")
			plt.xticks([])
			plt.yticks([])
			
			plt.subplot(2, 3, 3)
			plt.plot(origin_selection_imgs.mean((1, 2)))
			plt.title("Footprint occupancy over time")

		self.footprint = origin_selection_imgs
			
		return origin_selection_imgs
	
	def analyze_origin(self, oridb: str, window: int = 1000, 
					 footprint_bounds: List[int] = [-5, 8, 3, 10],
					 alpha1: int = 14, alpha2: int = 12) -> np.ndarray:
		"""
		Perform a complete analysis of a single origin.
		
		Args:
			oridb: Origin ID to analyze
			window: Window size around origin position
			footprint_bounds: Boundaries for the footprint [left, right, bottom, top]
			alpha1: Alpha value for config1
			alpha2: Alpha value for config2
			copy_correct: Whether to apply copy correction
			
		Returns:
			NumPy array of selected footprint images
		"""
		if self.origins is None:
			self.load_data()
			
		# self.adjust_alpha_configs(alpha1, alpha2)
		self.load_origin_mnase_data(oridb, window)
		self.combined_model.setup_deconv_model()
		self.plot_raw_data()

		return self.select_footprint_boundary(footprint_bounds)


	def setup_footprint_deconvolution(self, replicate, alpha):
		model = self.combined_model
		solver = model.solver
		footprint = self.footprint

		config = self.config1 if replicate == 1 else self.config2
		config.modify_alpha(alpha)

		num_rep1_tps = len(self.config1.timepoints)
		num_rep2_tps = len(self.config2.timepoints)

		rep1_sub_indices = np.arange(num_rep1_tps)
		rep2_sub_indices = np.arange(num_rep1_tps, footprint.shape[0])

		n_diag = np.diag(model.N)
		n1_diag = n_diag[rep1_sub_indices]
		n2_diag = n_diag[rep2_sub_indices]
		N1 = np.diag(n1_diag)
		N2 = np.diag(n2_diag)

		if replicate == 1:
			H = model.H1
			rep_N = N1
			rep_G_footprint = footprint[rep1_sub_indices].reshape((num_rep1_tps, -1))
			config = self.config1
		else:
			H = model.H2
			rep_N = N2
			rep_G_footprint = footprint[rep2_sub_indices].reshape((num_rep2_tps, -1))
			config = self.config2

		self.footprint_config = config

		solver.H = H
		solver.N = rep_N
		solver.G = rep_G_footprint
		solver.config = config

		self.solver = solver

	def deconvolve_footprint(self):
		footprint_img_shape = self.footprint.shape
		self.footprint_F = self.solver.deconvolve_G_iteratively(gamma=0.01, kappa=1, verbose=True)
		self.footprint_F_imgs = self.footprint_F.reshape((-1, 
			footprint_img_shape[1], footprint_img_shape[2]))

	def plot_deconvolved_footprint(self):
		i_tps = self.footprint_config.get_timepoints_for_branch('i')
		t_tps = self.footprint_config.get_timepoints_for_branch('t')
		b_tps = self.footprint_config.get_timepoints_for_branch('b')
		
		plt.plot(i_tps, self.footprint_F_imgs.mean((1, 2))[self.footprint_config.i_indices()], label="Recovery")
		plt.plot(t_tps, self.footprint_F_imgs.mean((1, 2))[self.footprint_config.t_indices()], label="Mother")
		plt.plot(b_tps, self.footprint_F_imgs.mean((1, 2))[self.footprint_config.b_indices()], label='Daughter')


