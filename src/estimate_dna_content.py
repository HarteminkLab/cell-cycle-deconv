import numpy as np
from scipy.stats import norm
from math import comb


# Built off of the helpers.py class
# We will use this file to estimate the DNA content through the time course
#
# Changes made:
#
# S phase is tracked, and the progress through S phase is tracked to get an estimate of the DNA content.
# Will no longer sum to total to 1, so we can get an accurate count of DNA content through the time course.
#



# The initial population mass, used in the Qr and Mgr calculations
START = 1000

# Maximum number of cell cycle "runs"
MAX_RUNS = 10

def calcH_unnormalized(model_intervals, timepoints):
	parameters, relations, initial_timepoints, top_timepoints, bottom_timepoints, _ = model_intervals

	if len(parameters) == 8:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, halted = parameters
	else:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted = parameters

	initial_partial_H = [np.zeros((len(timepoints), len(lst)-1)) for lst in initial_timepoints]
	top_partial_H = [np.zeros((len(timepoints), len(lst)-1)) for lst in top_timepoints]
	bottom_partial_H = [np.zeros((len(timepoints), len(lst)-1)) for lst in bottom_timepoints]

	# For each timepoint in the experiment, (rows in g)
	for i, t in enumerate(timepoints):

		Q = 0

		# Compute the Qr value or mass at a given timepoint in the experiment
		# We are doing this for each run (cell cycle)
		for r in range(MAX_RUNS + 1):
			Q += Qr(mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha)

		# We also want to have a fraction of the initial population, so t=0
		# over the expected mass at our current time
		frac_init = Qr(mu0, sigma0, sigmav, delta, lambda_val, t, 0, alpha) / Q

		# Now we will construct our initial branch's columns
		# Enumerate through the timepoints of the initial branch
		for idx, tp in enumerate(initial_timepoints):

			# Compute the cdf for the initial branch timepoint
			cdf = norm.cdf(tp, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
			initial_partial_H[idx][i, :] = np.diff(cdf) * frac_init

		# For the top timepoint, we will be computing the cdf
		# to compute the mass for each timepoint interval
		# i.e.   CG1, and postG1
		# this is for the first cohort and cell cycle {0, 0}
		for runs in range(1, MAX_RUNS + 1):

			# Enumerate through the timepoints for each subinterval belonging to the to top timepoints
			for idx, tp in enumerate(top_timepoints):

				# Compute the cdf for the top branch timepoint
				cdf = norm.cdf(tp + runs * lambda_val, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
				top_partial_H[idx][i, :] += np.diff(cdf) * frac_init

		# This computes the cohorts for the other cohorts {1+, 1+}
		# For the top and bottom branches
		for r in range(1, MAX_RUNS + 1):
			for g in range(1, r + 1):

				# How much mass is there for the current cohort
				frac_rest = Mgr(mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha) / Q

				if frac_rest > 1e-10:
					trun_cdf = norm.cdf(r * lambda_val + (g-1) * delta - alpha, loc=t-mu0, 
						scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
					trun_denom = 1 - trun_cdf

					for idx, tp in enumerate(top_timepoints):
						for runs in range(r + 1, MAX_RUNS + 1):
							cdf = norm.cdf(tp + runs * lambda_val + g * delta, loc=t-mu0, 
									scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
							portion = cdf * 0 if trun_denom == 0 else (cdf - trun_cdf) / trun_denom
							top_partial_H[idx][i, :] += np.diff(portion) * frac_rest

					for idx, tp in enumerate(bottom_timepoints):
						cdf = norm.cdf(tp + r * lambda_val + g * delta, loc=t-mu0, 
							scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
						portion = cdf * 0 if trun_denom == 0 else (cdf - trun_cdf) / trun_denom
						bottom_partial_H[idx][i, :] += np.diff(portion) * frac_rest

	Hsegments = {}
	for i in range(len(relations)):
		relation = relations[i]

		for idx in range(1, len(relation) - 1, 2):
			label = relation[idx]
			num = int(relation[idx + 1])
			if label == 'i':
				matrix = initial_partial_H[num]
			elif label == 't':
				matrix = top_partial_H[num]
			elif label == 'b':
				matrix = bottom_partial_H[num]
			if idx == 1:
				Hsegments[i] = matrix
			else:
				Hsegments[i] += matrix

	H, Hpos, cur_start = np.hstack(list(Hsegments.values())), {}, 0
	for i in range(len(Hsegments)):
		cur_len = Hsegments[i].shape[1]
		cur_end = cur_start + cur_len
		Hpos[i] = [cur_start, cur_end]
		cur_start = cur_end

	# compute the expected alive and halted mass at each timepoint
	mass_dic = get_alive_halted_mass(model_intervals, timepoints)
	H_w_halted = np.zeros((H.shape[0], H.shape[1]+1))
	H_w_halted = H

	# Normalize the matrix to sum to 1.
	H_w_halted = H_w_halted / H_w_halted.sum(axis=1).reshape((H_w_halted.shape[0], 1))

	# Multiply the expected alive and halted cells such that the 
	# Halted curves will represent expected growth.
	for i in range(len(timepoints)):
		time = timepoints[i]
		halted, alive, total = mass_dic[time]

		# Adjust the H matrix for the alive cells
		# columns up to the last column
		H_w_halted[i, :-1] = H_w_halted[i, :-1]*alive

		# Add the halted cells proportion as the last column
		H_w_halted[i, -1] = halted

	return H_w_halted, Hpos
	

def Qr(mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha):
	"""
	I believe this returns the mass of cells at a given time and reproductive instance. 
	Seemingly starting with a mass
	of 1000
	"""
	if r == 0:
		return START
	else:

		# For each of the reproductive instances r, compute the amount of mass that will contribute
		N = 0
		for i in range(r):
			normval = 1 - norm.cdf(r * lambda_val + i * delta - alpha, loc=t-mu0, 
				scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
			N += normval * START * comb(r-1, i)

		return N


def Mgr(mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha):
	if g == 0:
		if r == 0:
			return START
		else:
			return 0
	elif g > 0:
		if r < g:
			return 0
		else:
			normval = 1 - norm.cdf(r * lambda_val + (g-1) * delta - alpha, loc=t-mu0, 
				scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
			return normval * START * comb(r-1, g-1)
	else:
		raise ValueError('g should be greater than or equal to 0.')

def get_alive_halted_mass(model_intervals, timepoints):
	
	parameters, relations, initial_timepoints, \
	top_timepoints, bottom_timepoints, _ = model_intervals
	if len(parameters) == 8:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, halted = parameters
	else:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted = parameters

	haltedMass = None
	mass_dic = {}
	for time in timepoints:

		Q = 0
		for r in range(MAX_RUNS + 1):
				Q += Qr(mu0, sigma0, sigmav, delta, lambda_val, time, r, alpha)
		if time == 0:
			haltedMass = Q*halted
		aliveMass = Q-haltedMass

		# dictionary of halted, alive, and total mass
		mass_dic[time] = (haltedMass, aliveMass, Q)

	return mass_dic


