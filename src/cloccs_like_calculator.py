

import numpy as np
import math


CohortNone = "CohortNone"
CohortNotFirstGenNotFirstRep = "CohortNotFirstGenNotFirstRep"
CohortFirstGenNotFirstRep = "CohortFirstGenNotFirstRep"
CohortFirstGenFirstRep = "CohortFirstGenFirstRep"

MAXNUMCELLCYCLES = 5


class LikeCalculator:
	"""A port of the CLOCCS log-likelihood calculator.

		This will give us the expected DNA content at each experimental timepoint.
	"""

	def __init__(self):

		# Replicate 2 posteriors
		self.mu0 = -20.1510
		self.delta = 10.8953
		self.sigma0 = 10.6310
		self.sigmav = 0.1297
		self.lambda_val = 60.2093
		self.gamma1 = 0.0017
		self.gamma2 = 0.2837
		self.mua1 = 7.5849
		self.sigmaa1 = 0.0011
		self.mua2 = 1.0103
		self.sigmaa2 = 0.0156
		self.mut = -2.4831
		self.sigmat = 0.0755

	def get_cohort_cell_cycle_state_probabilities(self, logflo, g, r, 
			cell_cycle_time, log_fluorescence_time, timer):

		cohort_probs = np.zeros(3)

		# All CLOCCS parameters
		mu0 = self.mu0
		delta = self.delta
		sigma0 = self.sigma0
		sigmav = self.sigmav
		lambda_val = self.lambda_val
		gamma1 = self.gamma1
		gamma2 = self.gamma2
		mua1 = self.mua1
		sigmaa1 = self.sigmaa1
		mua2 = self.mua2
		sigmaa2 = self.sigmaa2
		mut = self.mut
		sigmat = self.sigmat

		a1 = mua1
		a2 = mua2


		# Convert negative precision value of mut to
		# std
		tau = math.exp(mut)

		cohort_mode = self.get_cohort_mode(g, r)

		# The cell cycle progression parameters
		position_norm_params = self.get_position_distribution_parameters(g, r, cell_cycle_time)

		istarG1 = 0.0
		istarG2M = 0.0
		istarS = 0.0

		mean_g = position_norm_params[0]
		sd_g = position_norm_params[1]


		choose_r, choose_g = max(r, 1), max(g, 1)
		choose_val = math.comb(choose_r - 1, choose_g - 1)
		g1CDFMass = 0

		from scipy.stats import norm

		# # The cohort membership distribution logic...
		g1FirstCellCycleMass = norm.cdf(gamma1 * lambda_val, loc=mean_g, scale=sd_g)
		g2MPositionProb = 0.0

		w1t = a2 / (lambda_val * (gamma2 - gamma1))
		meanSdenom = sd_g * sd_g * w1t * w1t + tau * tau
		sdS = math.sqrt(sd_g * sd_g * tau * tau / meanSdenom)

		# The left boundary for non-first generation cells, the lgr left limit
		# support position in the truncated normal distribution
		if(cohort_mode == CohortNotFirstGenNotFirstRep):  # g > 0, r > 0

			g1FirstCellCycleMass -= norm.cdf(-delta, mean_g, sd_g)

		# Recovery cells use the recovery boundary, which is negative infinity,
		# subtracting 0, but left here for completion and understanding
		elif (cohort_mode == CohortFirstGenFirstRep):  # 0, 0
			g1FirstCellCycleMass -= 0

		# If 1st generation and not 1st reproductive instance, we have no mass, thus
		# return no probabilities, these cells do not have a previous generation
		# contributing mass to it
		if (cohort_mode == CohortFirstGenNotFirstRep):  # 0, 1+
			cohort_probs[0] = 0
			cohort_probs[1] = 0
			cohort_probs[2] = 0
			return cohort_probs


		# The CDF mass for the first cell cycle is compeleted
		g1CDFMass = g1FirstCellCycleMass

		# Now add the G1 masses for the subsequent cell cycles
		for c in range(1, MAXNUMCELLCYCLES):
			g1CDFMass += (norm.cdf((c + gamma1) * lambda_val, mean_g, sd_g) - norm.cdf(c * lambda_val, mean_g, sd_g))

		# Standard Normal of fluorescence centered on G1
		istarG1 += g1CDFMass * norm.pdf(logflo, a1, tau)

		# G2 component
		for c in range(0, MAXNUMCELLCYCLES):

			# CDF between (G2) - (End of S) of belonging to cell cycle c
			g2MPositionProb += (norm.cdf((c + 1.0) * lambda_val, mean_g, sd_g) -
					norm.cdf((c + gamma2) * lambda_val, mean_g, sd_g))

		istarG2M += g2MPositionProb * norm.pdf(logflo, a1 + a2, tau)

		# S component
		for c in range(0, MAXNUMCELLCYCLES):

			w0t = a1 - (a2 * (gamma1 + c)) / (gamma2 - gamma1)
			meanS = (w1t * sd_g * sd_g * (logflo - w0t) + mean_g * tau * tau) / meanSdenom

			# Where in the time to place the gamma 1 and gamma 2 positions
			# dependent on the current cell cycle number and cell cycle length
			gammaLoc2 = (c + gamma2) * lambda_val
			gammaLoc1 = (c + gamma1) * lambda_val

			# The positional component for S, what is the probability
			# of the positions for the gamma1 and gamma2 locations
			# The difference is the current S positions probability?
			probSGamma2 = norm.cdf(gammaLoc2, meanS, sdS)
			probSGamma1 = norm.cdf(gammaLoc1, meanS, sdS)
			currentSPositionProb = probSGamma2 - probSGamma1

			# The flow component for S
			# A linear interpolation of:
			# time dimension: gamma1 to gamma2 (for the current cell cycle)
			# log-fluorescence dimension: a1 to a1+a2
			sFlowMean = w0t + w1t * mean_g
			currentSFlowProb = norm.pdf(logflo, sFlowMean, math.sqrt(meanSdenom))
			istarS += currentSPositionProb * currentSFlowProb

		cohort_probs[0] = istarG1 * choose_val
		cohort_probs[1] = istarS * choose_val
		cohort_probs[2] = istarG2M * choose_val

		return cohort_probs

	def get_position_distribution_parameters(self, g, r, cc_time):

		sigma0 = self.sigma0
		sigmav = self.sigmav
		lambda_val = self.lambda_val
		mu0 = self.mu0
		muv = 1.
		delta = self.delta

		sd_g = math.sqrt(sigma0 * sigma0 + sigmav * sigmav * cc_time * cc_time)
		mean_g = mu0 + muv * cc_time - lambda_val * r - delta * g
		return mean_g, sd_g


	def get_cohort_mode(self, g, r):

		cohort_mode = CohortNone

		# We are in a cohort that is not the original generation's first cell cycle
		if (g != 0): # g > 0, r > 0
			cohort_mode = CohortNotFirstGenNotFirstRep

		# We are in a cohort that is the first generation, but after the first cell cycle
		elif(r != 0): # g = 0, r > 0
			cohort_mode = CohortFirstGenNotFirstRep

		# This is the original cohort's first cell cycle
		else: # g = 0, r = 0
			cohort_mode = CohortFirstGenFirstRep
		
		return cohort_mode
