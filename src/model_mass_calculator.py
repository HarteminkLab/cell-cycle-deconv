
import numpy as np
from scipy.stats import norm
from scipy.special import comb

CohortNotFirstGenNotFirstRep = "CohortNotFirstGenNotFirstRep"
CohortFirstGenNotFirstRep = "CohortFirstGenNotFirstRep"
CohortFirstGenFirstRep = "CohortFirstGenFirstRep"
MaxNumCellCycles = 6


class MassCalculator:
	def __init__(self):

		mu0, lambda_val, delta, sigma0, sigmav, alpha = (
			-21.0, 79.3787, 1.1588, 15.2752, 0.1611, 0
			)
		self.gamma1 = 0.157
		self.gamma2 = 0.4
		self.parameters = mu0, lambda_val, delta, sigma0, sigmav, alpha


	def get_position_distribution_parameters(self, g, r, time):

		mu0, lambda_val, delta, sigma0, sigmav, alpha = self.parameters

		# Assume velocity of 1
		muv = 1

		sd_g = np.sqrt(sigma0 * sigma0 + sigmav * sigmav * time * time)
		mean_g = mu0 + muv * time - lambda_val * r - delta * g
		return mean_g, sd_g


	def get_cohort_mode(self, g, r):

		# We are in a cohort that is not the original generation's first cell cycle
		if g != 0:
			cohort_mode = CohortNotFirstGenNotFirstRep

		# We are in a cohort that is the first generation, but after the first cell cycle
		elif r != 0:
			cohort_mode = CohortFirstGenNotFirstRep

		# This is the original cohort's first cell cycle
		else:
			cohort_mode = CohortFirstGenFirstRep

		return cohort_mode


	def get_cell_cycle_subinterval_mass(self, time, g, r):

		cohort_mode = self.get_cohort_mode(g, r)  # Assuming this method is defined

		if cohort_mode == CohortFirstGenNotFirstRep:
			return np.zeros(6)

		chooseval = comb(max(0, r - 1), max(0, g - 1))

		mu0, lambda_val, delta, sigma0, sigmav, alpha = self.parameters
		params = self.get_position_distribution_parameters(g, r, time)  # Assuming this method is defined
		mean_g, sd_g = params

		gamma1 = self.gamma1
		gamma2 = self.gamma2

		mass_g1, mass_g2, mass_s = 0, 0, 0
		mass_r, mass_dg1, mass_cg1 = 0, 0, 0

		for c in range(MaxNumCellCycles):
			current_r_mass, current_dg1_mass = 0, 0

			cg1_left_boundary = c * lambda_val
			g1_right_boundary = (c + gamma1) * lambda_val

			current_cg1_mass = (norm.cdf(g1_right_boundary, mean_g, sd_g) -
								norm.cdf(cg1_left_boundary, mean_g, sd_g))

			if c == 0:
				if g == 0:
					current_r_mass = norm.cdf(0, mean_g, sd_g) - norm.cdf(-np.inf, mean_g, sd_g)
					current_r_mass = max(current_r_mass, 0)
					current_cg1_mass = 0
					current_r_mass += norm.cdf(gamma1 * lambda_val, mean_g, sd_g) - norm.cdf(0, mean_g, sd_g)
				else:
					current_cg1_mass = 0
					current_dg1_mass = norm.cdf(gamma1*lambda_val, mean_g, sd_g) - norm.cdf(-delta, mean_g, sd_g)


			mass_g1 += current_r_mass + current_cg1_mass + current_dg1_mass
			mass_r += current_r_mass
			mass_dg1 += current_dg1_mass
			mass_cg1 += current_cg1_mass

			g2_left_boundary = (c + gamma2) * lambda_val
			g2_right_boundary = (c + 1.0) * lambda_val
			mass_g2 += norm.cdf(g2_right_boundary, mean_g, sd_g) - norm.cdf(g2_left_boundary, mean_g, sd_g)

			s_left_boundary = (c + gamma1) * lambda_val
			s_right_boundary = (c + gamma2) * lambda_val
			mass_s += norm.cdf(s_right_boundary, mean_g, sd_g) - norm.cdf(s_left_boundary, mean_g, sd_g)

		mass_g1 *= chooseval
		mass_s *= chooseval
		mass_g2 *= chooseval
		mass_r *= chooseval
		mass_cg1 *= chooseval
		mass_dg1 *= chooseval

		masses = [mass_g1, mass_s, mass_g2, mass_r, mass_cg1, mass_dg1]

		return masses


	def compute_subinterval_mass(self, time):

		g1Proportion = 0.0
		g2Proportion = 0.0
		sProportion = 0.0
		rProportion = 0.0
		cg1Proportion = 0.0
		dg1Proportion = 0.0

		for g in range(MaxNumCellCycles + 1):
			for r in range(g, MaxNumCellCycles + 1):
				cohort_masses = self.get_cell_cycle_subinterval_mass(time, g, r)

				g1Proportion += cohort_masses[0]
				sProportion += cohort_masses[1]
				g2Proportion += cohort_masses[2]

				rProportion += cohort_masses[3]
				cg1Proportion += cohort_masses[4]
				dg1Proportion += cohort_masses[5]

		total_intervals_sum = g1Proportion + sProportion + g2Proportion
		g1Proportion /= total_intervals_sum
		sProportion /= total_intervals_sum
		g2Proportion /= total_intervals_sum

		total_g1_proportion = g1Proportion / (rProportion + cg1Proportion + dg1Proportion)
		rProportion *= total_g1_proportion
		cg1Proportion *= total_g1_proportion
		dg1Proportion *= total_g1_proportion

		return total_g1_proportion, rProportion, cg1Proportion, dg1Proportion, \
				sProportion, g2Proportion


	def compute_proportions_all_times(self, timepoints):
		"""Compute the mass for each subinterval for each time point"""

		import pandas as pd

		all_total_g1_proportion = []
		all_rProportion = []
		all_cg1Proportion = []
		all_dg1Proportion = []
		all_sProportion = []
		all_g2Proportion = []

		for time in timepoints:
			(total_g1_proportion, rProportion, cg1Proportion, dg1Proportion, \
					sProportion, g2Proportion) = self.compute_subinterval_mass(time)

			all_total_g1_proportion.append(total_g1_proportion)
			all_rProportion.append(rProportion)
			all_cg1Proportion.append(cg1Proportion)
			all_dg1Proportion.append(dg1Proportion)
			all_sProportion.append(sProportion)
			all_g2Proportion.append(g2Proportion)

		sub_mass_df = pd.DataFrame({
			'time': timepoints,
			'total_g1': all_total_g1_proportion,
					 'R': all_rProportion,
					 'CG1': all_cg1Proportion,
					 'DG1': all_dg1Proportion,
					 'S': all_sProportion,
					 'G2': all_g2Proportion})

		self.sub_mass_df = sub_mass_df
		return sub_mass_df


	def plot_subinterval_mass_port(self):
	  
		from matplotlib import pyplot as plt
		from src.plot_helpers import plot_stacked_curves
	  
		sub_mass_df = self.sub_mass_df
		tp = sub_mass_df.time
		phase_columns = sub_mass_df.columns[2:]

		plt.figure(figsize=(6, 4))
		plt.ylim(0, 2.)

		vectors = []

		for phase in phase_columns:
			cur_vec = sub_mass_df[phase]
			vectors.append(cur_vec)
			
		plot_stacked_curves(tp, vectors, phase_columns)

		plt.legend()
