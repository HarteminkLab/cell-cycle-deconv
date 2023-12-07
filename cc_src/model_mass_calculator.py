
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

		gamma1 = 0.157
		gamma2 = 0.4

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
					current_dg1_mass = norm.cdf(0, mean_g, sd_g) - norm.cdf(-delta, mean_g, sd_g)


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


