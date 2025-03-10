
import numpy as np
import pandas as pd


# Number of positions in H devoted to each cell cycle phase
# G1+PostG1 equals a power of 2 (64)
# Add 1 because the timepoints are inclusive (?)
G1_NUM_TPS = 42
POSTG1_NUM_TPS = 22


class RG1Model(object):
	"""Model to handle loading, saving, and configuring deconvolution model runs..

	This model will aim to replace the create_models and config classes....
	"""
	def __init__(self, config_type='distinct'):
		# single or distinct mother and daughter g1s
		self.config_type = config_type

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

		if self.config_type == 'shared':
			Hpositions = np.concatenate([
						range(0, G1_NUM_TPS), # RG1
						range(G1_NUM_TPS, G1_NUM_TPS*2), # CG1 and DG1
						range(G1_NUM_TPS, G1_NUM_TPS*2), # CG1 and DG1
						range(G1_NUM_TPS*2, G1_NUM_TPS*2+POSTG1_NUM_TPS), # CG1 and DG1
						[G1_NUM_TPS*2+POSTG1_NUM_TPS], # Halted
					])
		elif self.config_type == 'distinct':
			Hpositions = np.concatenate([
						range(0, G1_NUM_TPS), # RG1
						range(G1_NUM_TPS, G1_NUM_TPS*2), # CG1
						range(G1_NUM_TPS*2, G1_NUM_TPS*3), # DG1 (same indices)
						range(G1_NUM_TPS*3, G1_NUM_TPS*3+POSTG1_NUM_TPS), # Post G1
						[G1_NUM_TPS*3+POSTG1_NUM_TPS], # Halted
					])
		else:
			raise ValueError(f"Unknown config type: {self.config_type}")

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
				Hpositions,
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

	# Helper functions for common getters
	def i_indices(self):
		return self.get_Hpositions_for_branch('i')

	def t_indices(self):
		return self.get_Hpositions_for_branch('t')

	def b_indices(self):
		return self.get_Hpositions_for_branch('b')

	def cg1_indices(self):
		return self.get_Hpositions_for_phase('CG1')

	def rg1_indices(self):
		return self.get_Hpositions_for_phase('RG1')

	def dg1_indices(self):
		return self.get_Hpositions_for_phase('DG1')

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

		from src.calcH_single_g1 import calcH as single_calcH
		from src.calcH_separate_G1 import calcH as separate_calcH
		from src.global_config import GlobalConstants

		model_intervals = self.retrieve_model_intervals_for_calcH()

		if self.config_type == 'shared':
			self.H, _ = single_calcH(model_intervals, self.timepoints)
		elif self.config_type == 'distinct':
			self.H, _ = separate_calcH(model_intervals, self.timepoints)
		else:
			raise ValueError(f"Invalid config type {self.config_type}")

		return self.H

	def plot_H(self):

		from src.plot_helpers import color_for_key
		import matplotlib.pyplot as plt
		from src.figure_configs import FiguresConfig

		H = self.H
		rg1_cols = self.get_Hpositions_for_phase('RG1')
		cg1_cols = self.get_Hpositions_for_phase('CG1')
		dg1_cols = self.get_Hpositions_for_phase('DG1')
		post_g1_cols = self.get_Hpositions_for_phase('postG1')

		H_cols = np.array([H.shape[1]-1])

		phases = ['H', 'RG1', 'CG1', 'DG1', 'postG1']
		cols_list = [H_cols, rg1_cols, cg1_cols, dg1_cols, post_g1_cols]

		plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT_EXTRAWIDE)
		plt.subplot(1, 2, 1)

		plt.title("H convolution matrix", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		plt.imshow(H, vmax=H[:, :-1].max(), aspect='auto', cmap='Reds',
				  extent=[0, H.shape[1], self.timepoints[-1], 0])

		plt.subplot(1, 2, 2)

		plt.title("Phase proportions over time", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		x = self.timepoints
		prev = np.zeros(len(x))

		label_mapping = {
			'H': 'Halted',
			'RG1': 'Recovery-G1',
			'CG1': 'Mother-G1',
			'DG1': 'Daughter-G1',
			'postG1': 'post-G1',
		}
		for i in range(len(phases)):
			phase = phases[i]
			cols = cols_list[i]
			color = color_for_key(phase)
			mass = H[:, cols].sum(axis=1)

			if self.config_type == 'shared':
				if phase in ['CG1', 'DG1']:
					mass = mass/2.

			y = prev+mass

			plt.fill_between(x, prev, y, color=color, label=label_mapping[phase])
			prev = y
		plt.legend()


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

	def compute_branch_lengths(self):
		"""Compute the branch lengths to determine the distribution of weights for smoothing"""

		rg1_tps = self.get_timepoints_for_phase('RG1')
		cg1_tps = self.get_timepoints_for_phase('CG1')
		dg1_tps = self.get_timepoints_for_phase('DG1')
		postg1_tps = self.get_timepoints_for_phase('postG1')

		length_rg1 = rg1_tps[-1]-rg1_tps[0]
		length_cg1 = cg1_tps[-1]-cg1_tps[0]
		length_dg1 = dg1_tps[-1]-dg1_tps[0]
		length_postg1 = postg1_tps[-1]-postg1_tps[0]

		length_rg1, length_cg1, length_dg1, length_postg1

		recovery_smoothing_tps_length = length_rg1+length_postg1
		top_smoothing_tps_length = length_cg1+length_postg1
		bottom_smoothing_tps_length = length_dg1+length_postg1

		print("Length of of each of the branches:")
		print("Recovery: ", recovery_smoothing_tps_length)
		print("Mother: ", top_smoothing_tps_length)
		print("Daughter: ", bottom_smoothing_tps_length)
		print()

		print("1/Proportion of the mother branch:")
		print("Recovery: ", top_smoothing_tps_length/recovery_smoothing_tps_length)
		print("Mother: ", top_smoothing_tps_length/top_smoothing_tps_length)
		print("Daughter: ", top_smoothing_tps_length/bottom_smoothing_tps_length)
		print()

		# Length of smoothing constraints
		initial_constraint_length = length_rg1 + length_postg1
		top_constraint_length = length_cg1 + length_postg1*2
		bottom_constraint_length = length_dg1 + length_postg1*2

		print("1/Proportion of the mother branch, with padded postg1:")
		print("Recovery: ", top_constraint_length/initial_constraint_length)
		print("Mother: ", top_constraint_length/top_constraint_length)
		print("Daughter: ", top_constraint_length/bottom_constraint_length)
		print()


def read_cloccs_posteriors(posteriors_filepath):
	params = {}
	with open(posteriors_filepath, 'r') as f:
		lines = f.readlines()
		for line in lines[1:-1]:
			line_spl = line.split()
			params[line_spl[0]] = float(line_spl[1])

	return params


def load_default_configs(config_type='distinct', from_CLOCCS=True,
	mode='chromatin'):

	from src.global_config import GlobalConstants

	config1 = RG1Model(config_type=config_type)

	if from_CLOCCS:
		config1.load_from_posteriors('data/2019_cloccs_fits/yl_2019_replicate1/posteriors.txt',
								  GlobalConstants.CHROM_WT1_TIMEPOINTS, alpha=22)
	else:
		print("todo: Loading testing config from replication deconvolution")
		config1.alpha = 0
		config1.params_dic = {
		    'mu0': 9.361663,
		    'lambda': 60.397553,
		    'delta': 14.577271,
		    'sigma0': 5.884135,
		    'sigmav': 0.044401,
		    'gamma1': 0.586077,
		    'gamma2': 1.000000,
		    'halted': 0.000050, 
		    'alpha': 0,
		}

	config1.replicate = 1

	config2 = RG1Model(config_type=config_type)

	if from_CLOCCS:
		config2.load_from_posteriors('data/2019_cloccs_fits/yl_2019_replicate2/posteriors.txt',
								  GlobalConstants.CHROM_WT2_TIMEPOINTS, alpha=20)
	else:
		config2.alpha = 0
		config2.params_dic = {
			'mu0': 20.0238,
			'delta': 9.154,
			'sigma0': 3.919,
			'sigmav': 0.115,
			'lambda': 60.00,
			'gamma1': 0.663,
			'gamma2': 1.0,
			'halted': 1.7984e-06,
			'alpha': 0}
	config2.replicate = 2

	if from_CLOCCS:
		print("Shifting mu0, gamma1, and gamma2, for alpha...")
		config1.shift_parameters_for_alpha()
		config2.shift_parameters_for_alpha()

	if mode == "expression":
		config1.timepoints = GlobalConstants.EXPRESSION_WT1_TIMEPOINTS
		config2.timepoints = GlobalConstants.EXPRESSION_WT2_TIMEPOINTS
	else:
		config1.timepoints = GlobalConstants.CHROM_WT1_TIMEPOINTS
		config2.timepoints = GlobalConstants.CHROM_WT2_TIMEPOINTS

	# Generate H for each replicate
	config1.update_timepoints()
	config1.calculate_H()
	config2.update_timepoints()
	config2.calculate_H()

	return config1, config2


def load_default_expression_configs(config_type='distinct', from_CLOCCS=False):
	return load_default_configs(config_type=config_type, mode='expression', from_CLOCCS=from_CLOCCS)

def load_default_chrom_configs(config_type='distinct', from_CLOCCS=False):
	return load_default_configs(config_type=config_type, mode='chromatin', from_CLOCCS=from_CLOCCS)
