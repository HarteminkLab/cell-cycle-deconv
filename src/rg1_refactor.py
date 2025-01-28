
import numpy as np
import pandas as pd


# Number of positions in H devoted to each cell cycle phase
# G1+PostG1 equals a power of 2 (64)
# Add 1 because the timepoints are inclusive (?)
G1_NUM_TPS = 22
POSTG1_NUM_TPS = 42


class RG1Model(object):
	"""Model to handle loading, saving, and configuring deconvolution model runs..

	This model will aim to replace the create_models and config classes....
	"""

	def load_from_posteriors(self, posteriors_filepath, timepoints, alpha=0):

		self.alpha = alpha

		params_dic = read_cloccs_posteriors(posteriors_filepath)	

		# Keep parameters relevant for the creation of H
		keep_params = ['mu0', 'delta','sigma0', 'sigmav','lambda','gamma1','gamma2', 'halted']
		filtered_params_dic = {key: params_dic[key] for key in keep_params}
		filtered_params_dic['alpha'] = self.alpha

		self.params_dic = filtered_params_dic
		self.timepoints = timepoints
		self.update_timepoints()

	def update_timepoints(self):
		self.create_timepoints_df()
		self.create_branch_Hpos_map()
		
	def create_timepoints_df(self):

		mu0 = self.params_dic['mu0']
		lambda_val = self.params_dic['lambda']
		gamma1 = self.params_dic['gamma1']
		delta = self.params_dic['delta']
		start_of_s = gamma1*lambda_val

		# for now, let's set alpha to 0
		alpha = self.alpha

		# ***** here, is alpha handled here? or mu0 is defined/specified differently... ***
		# Ambiguious specification
		#
		#       mu0 includes g1 time if mu0 is longer than g1, this had not come up when start of s
		#       was typically 0.0
		#
		#       Now, when start of s > 0, mu0 is any additional length that is not accounted for as 
		#       G1 (when mu0 is negative).
		#
		#       When mu0 is positive, this would indicate the first recovery G1 is shorter than
		#       the expected G1 length.
		#
		# The old model that includes alpha, from create_models.pl from
		# Xin's code
		# rg1_time_span = mu0, start_of_s 
		# cg1_time_span = -alpha, start_of_s
		# dg1_time_span = -delta-alpha, start_of_s
		# postg1_time_span = start_of_s, lambda_val-alpha

		# Assumption: the length of alpha, is additional time spent in G1 that isn't accounted for in FACS
		#             because of the cell-wall degradation timing.
		#
		#            The first cell cycle G1 does not include this degradation, thus alpha is not included in
		#            this timing. In that case the length of G1 is indeed:
	    #            (alpha + lambda*gamma1) ---- 
	    #
		# 
		# Now we are modeling gamma1 differently, with MNase, we don't consider the FACS limitations. Thus
		# CG1 can be modeled as 0 to lambda*gamma1. Meaning we need to add in the alpha time as the true length
		# of additional time spent in G1. 
		#
		# This in turn affects mu0: Now defined as a difference/"delta" from the expected start of G1 for 
		# the first recovery cell cycle.

		# Therefore if we are translating from the CLOCCS fits to the updated model's fit... we need to 
		# add alpha to mu0 to account for the additional length of G1. gamma1, gamma2 are both translated forward
		# so mu0 needs to as well...
		#
		# see the function: shift_parameters_for_alpha
		#
		# todo: this is an ongoing justification.... and affects the initialization of the parameter
		# fitting for the replication deconvolution.
		rg1_time_span = mu0, start_of_s
		cg1_time_span = 0, start_of_s
		dg1_time_span = -delta, start_of_s
		postg1_time_span = start_of_s, lambda_val

		rg1_timepoints = np.linspace(rg1_time_span[0], rg1_time_span[1], G1_NUM_TPS+1)
		cg1_timepoints = np.linspace(cg1_time_span[0], cg1_time_span[1], G1_NUM_TPS+1)
		dg1_timepoints = np.linspace(dg1_time_span[0], dg1_time_span[1], G1_NUM_TPS+1)
		postg1_timepoints = np.linspace(postg1_time_span[0], postg1_time_span[1], POSTG1_NUM_TPS+1)

		Hpos = np.concatenate([
					range(0, G1_NUM_TPS), # RG1
					range(G1_NUM_TPS, G1_NUM_TPS*2), # CG1 and DG1
					range(G1_NUM_TPS, G1_NUM_TPS*2), # CG1 and DG1
					range(G1_NUM_TPS*2, G1_NUM_TPS*2+POSTG1_NUM_TPS), # CG1 and DG1
					[G1_NUM_TPS*2+POSTG1_NUM_TPS], # Halted
				])

		# Create the dataframe that maps the H positions directly to the the timepoints
		timepoints_dataframe = pd.DataFrame({

			# these timepoints correspond to the start and end of cdf spans
			# therefore we are interested in keeping track of the span that
			# these timepoints correspond
			'timepoint_start': np.concatenate([rg1_timepoints[0:-1], 
				cg1_timepoints[0:-1], 
				dg1_timepoints[0:-1], 
				postg1_timepoints[0:-1], [0]]),

			'timepoint_end': np.concatenate([rg1_timepoints[1:], 
				cg1_timepoints[1:], 
				dg1_timepoints[1:], 
				postg1_timepoints[1:], [0]],),

			'phase': np.concatenate([
				np.repeat('RG1', G1_NUM_TPS),
				np.repeat('CG1', G1_NUM_TPS),
				np.repeat('DG1', G1_NUM_TPS),
				np.repeat('postG1', POSTG1_NUM_TPS),
				np.repeat('Halted', 1),
				])
			,
			'Hpos':
				np.concatenate([
					range(0, G1_NUM_TPS), # RG1
					range(G1_NUM_TPS, G1_NUM_TPS*2), # CG1
					range(G1_NUM_TPS, G1_NUM_TPS*2), # DG1 (same indices)
					range(G1_NUM_TPS*2, G1_NUM_TPS*2+POSTG1_NUM_TPS), # Post G1
					[G1_NUM_TPS*2+POSTG1_NUM_TPS], # Halted
				]),
			})
		
		self.timepoints_df = timepoints_dataframe.set_index('phase')

		self.branch_phase_mapping = {
			'i': ['RG1', 'postG1'],
			't': ['CG1', 'postG1'],
			'b': ['DG1', 'postG1'],
		}

		# Create the dataframe that contains the per branch timepoints and
		# h positions
		self.create_branch_Hpos_map()


	def create_branch_Hpos_map(self):
		"""Create a data frame that contains the index,
		and timepoints information for each branch and phase.
		"""

		branch_Hpos_df = pd.DataFrame()

		for branch in ['i', 't', 'b']:

			phases = self.branch_phase_mapping[branch]
			tps_df = self.timepoints_df
			indices = np.array([])

			for phase in phases:
				current_phase_df = tps_df.loc[phase].copy()
				current_phase_df['branch'] = branch
				branch_Hpos_df = pd.concat([branch_Hpos_df, current_phase_df])
		branch_Hpos_df = branch_Hpos_df.reset_index().set_index(['branch', 'phase'])

		self.branch_Hpos_df = branch_Hpos_df


	def get_Hpositions_for_phase(self, phase):

		if phase == 'H' or phase == 'Halted':
			return np.array([-1])

		elif phase in ['S', 'G2M']:

			s_indices, g2m_indices = self.get_s_g2m_indices()

			if phase == 'S':
				return s_indices
			elif phase == 'G2M':
				return g2m_indices

		return self.timepoints_df.loc[phase].Hpos.values

	def get_Hpositions_for_branch(self, branch):
		return self.branch_Hpos_df.loc[branch].Hpos.values
		
	def get_timepoints_for_phase(self, phase):

		if phase in ['S', 'G2M']:

			s_indices = self.get_Hpositions_for_phase('S')
			postG1_rows = self.timepoints_df.loc['postG1']

			if phase == 'S':
				return postG1_rows[postG1_rows.Hpos <= s_indices[-1]].timepoint_start.values
			else:
				return postG1_rows[postG1_rows.Hpos > s_indices[-1]].timepoint_start.values

		else:
			return self.timepoints_df.loc[phase].timepoint_start.values

	def get_timepoints_for_branch(self, branch):
		return self.branch_Hpos_df.loc[branch].timepoint_start.values

	def get_key_timepoints_in_raw(self, full=False):

		alpha = self.alpha
		mu0, lambda_len, gamma1, gamma2 = (self.params_dic['mu0'],
			self.params_dic['lambda'],
			self.params_dic['gamma1'],
			self.params_dic['gamma2'],
		)

		# Estimate the first S from mu0, lambda, gamma1, and gamma2
		cg1_length = gamma1*lambda_len+alpha
		s_start = (lambda_len*gamma1)
		s_end = (lambda_len*gamma2)
		s_length = s_end - s_start

		# For the first cell cycle, mu0 includes the first G1
		# so S starts when Recovery (mu0) ends
		first_s_start = mu0
		first_s_end = mu0+s_length
		lambda_len = lambda_len

		# The end of the first cycle is computed
		# by taking the cell cycle length, subtracting the length of S 
		# (to get G1 and G2/M)
		# Then subtract out what the first G1 would be.
		# Then offset by mu0 length to get the actual timepoint for the 
		# end of the first cell cycle
		g1_recovery_would_start_here = first_s_start - cg1_length
		end_of_first_lambd = g1_recovery_would_start_here+lambda_len

		if full:
			return (g1_recovery_would_start_here, cg1_length, lambda_len,
				s_length, mu0, first_s_start, first_s_end, end_of_first_lambd)

		return mu0, first_s_start, first_s_end, end_of_first_lambd

	def get_s_g2m_indices(self):
		# Compute the G2M and S indices by collecting the length of S

		postg1_indices = self.get_Hpositions_for_phase('postG1')

		gamma1 = self.params_dic['gamma1']
		gamma2 = self.params_dic['gamma2']
		lambda_val = self.params_dic['lambda']

		s_start = lambda_val*gamma1
		s_end = lambda_val*gamma2

		s_len = s_end - s_start
		postg1_tps = self.get_timepoints_for_phase('postG1')

		s_indices = postg1_indices[(postg1_tps >= s_start) & (postg1_tps < s_end)]
		g2m_indices = postg1_indices[postg1_indices > s_indices[-1]]

		return s_indices, g2m_indices
		
	def calculate_H(self):

		from src.calcH_single_g1 import calcH
		from src.global_config import GlobalConstants

		model_intervals = self.retrieve_model_intervals_for_calcH()
		self.H, _ = calcH(model_intervals, self.timepoints)

		return self.H


	def retrieve_model_intervals_for_calcH(self):
		"""Calculate H function requires a very specific format for the parameters, this
		needs to be refactored, but for now, we will create the list of objects specifically
		in the format that the calculate H function desires."""

		params_dic = self.params_dic

		mu0, delta, sigma0, sigmav, lambda_val, gamma1, gamma2, alpha, halted = \
		(params_dic['mu0'], 
		 params_dic['delta'], 
		 params_dic['sigma0'], 
		 params_dic['sigmav'], 
		 params_dic['lambda'], 
		 params_dic['gamma1'], 
		 params_dic['gamma2'], 
		 params_dic['alpha'], 
		 params_dic['halted'])
		beta = 0

		parameters = -mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted
		relations = [['RG1', 'i', '0'],
					 ['CG1', 't', '0'],
					 ['DG1', 'b', '0'],
					 ['postG1', 'i', '1', 't', '1', 'b', '1']]
		def timepoints_list_for_branch(branch):
			phases = self.branch_phase_mapping[branch]
			timepoints_for_branch = []

			for phase in phases:

				# The timepoints for the calculate H function requires the entire span for cdf calculation
				# so we will need to append the final end to the array
				phase_timepoint_starts = self.branch_Hpos_df.loc[branch].loc[phase].timepoint_start.values
				phase_timepoint_ends = self.branch_Hpos_df.loc[branch].loc[phase].timepoint_end.values
				phase_timepoints = np.concatenate([phase_timepoint_starts, phase_timepoint_ends[-1:]])

				timepoints_for_branch.append(phase_timepoints)

			return timepoints_for_branch

		i_timepoints = timepoints_list_for_branch('i')
		t_timepoints = timepoints_list_for_branch('t')
		b_timepoints = timepoints_list_for_branch('b')

		return (parameters, relations, i_timepoints, t_timepoints, b_timepoints, None)

	def shift_parameters_for_alpha(self):
		"""From Guo, the CLOCCS estimates need to be adjusted because CLOCCS 
		assumes cells that have divided, but the cell wall has not been degraded yet, to 
		be in G2M.
		Guo estimated previously that this delay (alpha) was about 30% of the cell cycle 
		(exact value is in the paper).

		Thus we can use this value as the time/proportion that mu0 and gamma1 and gamma2
		should be shifted for the initialization
		"""
		alpha = self.alpha
		params_dic = self.params_dic

		lambda_ = params_dic['lambda']
		old_mu0 = params_dic['mu0']
		old_gamma1 = params_dic['gamma1']
		old_gamma2 = params_dic['gamma2']

		mu0 = old_mu0+alpha

		gamma_shift = alpha/lambda_ 

		gamma1 = old_gamma1 + gamma_shift
		gamma2 = old_gamma2 + gamma_shift

		params_dic = params_dic.copy()
		params_dic['mu0'] = mu0
		params_dic['gamma1'] = gamma1
		params_dic['gamma2'] = gamma2

		# alpha is now embedded into the mu0, gamma1, and gamma2 values so
		# we can set it to 0
		self.params_dic = params_dic
		self.alpha = 0
		self.update_timepoints()


def read_cloccs_posteriors(posteriors_filepath):
	params = {}
	with open(posteriors_filepath, 'r') as f:
		lines = f.readlines()
		for line in lines[1:-1]:
			line_spl = line.split()
			params[line_spl[0]] = float(line_spl[1])

	return params


def load_default_chrom_configs():

	from src.global_config import GlobalConstants

	config1 = RG1Model()
	config1.load_from_posteriors('data/2019_cloccs_fits/yl_2019_replicate1/posteriors.txt',
							  GlobalConstants.CHROM_WT1_TIMEPOINTS, alpha=22)

	config2 = RG1Model()
	config2.load_from_posteriors('data/2019_cloccs_fits/yl_2019_replicate2/posteriors.txt',
							  GlobalConstants.CHROM_WT2_TIMEPOINTS, alpha=20)


	return config1, config2

