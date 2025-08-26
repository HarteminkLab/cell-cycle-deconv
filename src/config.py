
import numpy as np
import pandas as pd
from src.global_config import GlobalConstants


# Number of indices assigned to each phase
# RG1, DG1, and CG1 have an equivalent number of timepoints for ease
# of computation. This approximation allows for an approximately 1 min per index
# deconvolution for MG1 and postG1
# DEFAULT_G1_NUM_TPS = 44
# DEFAULT_POSTG1_NUM_TPS = 84

DEFAULT_G1_NUM_TPS = 64
DEFAULT_POSTG1_NUM_TPS = 64

APPROX_MIN_PER_NUM_INDICES = 0.5

DEFAULT_REPLICATION_PARENT_DIRECTORY = 'output/draft3_run/'

class ModelConfig(object):
	"""Model to handle loading, saving, and configuring deconvolution model runs..

	This model will aim to replace the create_models and config classes....
	"""
	def __init__(self, config_type='distinct', g1_num_tps=None, postg1_num_tps=None):
		# single or distinct mother and daughter g1s
		self.config_type = config_type

		if g1_num_tps is None:
			self.g1_num_tps = DEFAULT_G1_NUM_TPS
		else:
			self.g1_num_tps = g1_num_tps

		if postg1_num_tps is None:
			self.postg1_num_tps = DEFAULT_POSTG1_NUM_TPS
		else:
			self.postg1_num_tps = DEFAULT_POSTG1_NUM_TPS

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
		self.calculate_H()

	def load_from_dic(self, dic_path, timepoints):
		from src.utils import load_dict_from_json

		self.params_dic = load_dict_from_json(dic_path)

		# No longer using alpha after the CLOCCS model
		# this should be set to 0 when it is not used
		self.alpha = self.params_dic['alpha'] 

		# If the indices does exist in the dictionary, set it
		if "num_g1_indices" in self.params_dic:
			self.num_g1_tps = self.params_dic['num_g1_indices']

		self.timepoints = timepoints
		self.update_timepoints()
		self.calculate_H()

	def save_to_path(self, dic_path):
		from src.utils import save_dict_to_json
		save_dict_to_json(self.params_dic, dic_path)
		print(f"Saved to : {dic_path}")

	def update_timepoints(self):
		self.create_timepoints_df()
		self.create_branch_Hpos_map()
		
	def create_timepoints_df(self):

		mu0 = self.params_dic['mu0']
		lambda_val = self.params_dic['lambda']
		gamma1 = self.params_dic['gamma1']
		delta = self.params_dic['delta']
		start_of_s = gamma1*lambda_val
		g1_num_tps = self.g1_num_tps
		postg1_num_tps = self.postg1_num_tps

		alpha = self.alpha

		rg1_time_span = mu0, start_of_s
		cg1_time_span = -alpha, start_of_s
		dg1_time_span = -delta-alpha, start_of_s
		postg1_time_span = start_of_s, lambda_val-alpha

		rg1_timepoints = np.linspace(rg1_time_span[0], rg1_time_span[1], g1_num_tps+1)
		cg1_timepoints = np.linspace(cg1_time_span[0], cg1_time_span[1], g1_num_tps+1)
		dg1_timepoints = np.linspace(dg1_time_span[0], dg1_time_span[1], g1_num_tps+1)
		postg1_timepoints = np.linspace(postg1_time_span[0], postg1_time_span[1], postg1_num_tps+1)

		if self.config_type == 'shared':
			Hpositions = np.concatenate([
						range(0, g1_num_tps), # RG1
						range(g1_num_tps, g1_num_tps*2), # CG1 and DG1
						range(g1_num_tps, g1_num_tps*2), # CG1 and DG1
						range(g1_num_tps*2, g1_num_tps*2+postg1_num_tps), # CG1 and DG1
						[g1_num_tps*2+postg1_num_tps], # Halted
					])
		elif self.config_type == 'distinct':
			Hpositions = np.concatenate([
						range(0, g1_num_tps), # RG1
						range(g1_num_tps, g1_num_tps*2), # CG1
						range(g1_num_tps*2, g1_num_tps*3), # DG1 (same indices)
						range(g1_num_tps*3, g1_num_tps*3+postg1_num_tps), # Post G1
						[g1_num_tps*3+postg1_num_tps], # Halted
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
				np.repeat('RG1', g1_num_tps),
				np.repeat('CG1', g1_num_tps),
				np.repeat('DG1', g1_num_tps),
				np.repeat('postG1', postg1_num_tps),
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

	def modify_alpha(self, new_alpha, num_g1_tps=None):

		total_branch_tps = self.g1_num_tps+self.postg1_num_tps

		if num_g1_tps is None:
			num_g1_tps = int(new_alpha/APPROX_MIN_PER_NUM_INDICES)

		self.g1_num_tps = num_g1_tps
		self.postg1_num_tps = total_branch_tps-num_g1_tps

		self.params_dic['alpha'] = new_alpha
		self.alpha = new_alpha
		self.update_timepoints()
		self.calculate_H()

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

	def postg1_indices(self):
		return self.get_Hpositions_for_phase('postG1')

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
		first_s_start = -mu0
		first_s_end = -mu0+s_length

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

		return first_s_start, first_s_end, end_of_first_lambd, cg1_length, s_length, lambda_len

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

	def retrieve_mass(self):
		from src.calcH_separate_G1 import get_alive_halted_mass
		model_intervals = self.retrieve_model_intervals_for_calcH()
		mass_dic = get_alive_halted_mass(model_intervals, self.timepoints)
		mass_df = pd.DataFrame(np.array(list(mass_dic.values())),
			columns=['halted', 'alive', 'total'],
			index=mass_dic.keys())
		return mass_df

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

	def plot_mass(self, plot_DNA=True, fig=None):

		from src.plot_helpers import color_for_key
		import matplotlib.pyplot as plt
		from src.figure_configs import FiguresConfig

		mass_df = self.retrieve_mass()

		H = self.H * (mass_df.total.values/1000.)[:, None]

		rg1_cols = self.get_Hpositions_for_phase('RG1')
		cg1_cols = self.get_Hpositions_for_phase('CG1')
		dg1_cols = self.get_Hpositions_for_phase('DG1')
		s_cols = self.get_Hpositions_for_phase('S')
		g2m_cols = self.get_Hpositions_for_phase('G2M')

		H_cols = np.array([H.shape[1]-1])

		phases = ['H', 'RG1', 'CG1', 'DG1', 'S', 'G2M']
		cols_list = [H_cols, rg1_cols, cg1_cols, dg1_cols, s_cols, g2m_cols]

		if fig is None:
			fig = plt.figure(figsize=(6, 4))

		plt.title("DNA mass over time", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		x = self.timepoints
		prev = np.zeros(len(x))

		label_mapping = {
			'H': 'Halted',
			'RG1': 'Recovery-G1',
			'CG1': 'Mother-G1',
			'DG1': 'Daughter-G1',
			'S': 'S',
			'G2M': 'G2/M',
		}
		for i in range(len(phases)):
			phase = phases[i]
			cols = cols_list[i]
			color = color_for_key(phase)
			mass = H[:, cols].sum(axis=1)

			if plot_DNA:
				if phase == 'G2M': mass = mass*2.
				elif phase == 'S':
					mass = mass * np.linspace(1, 2., len(mass))

				if self.config_type == 'shared':
					if phase in ['CG1', 'DG1']:
						mass = mass/2.

			y = prev+mass

			plt.fill_between(x, prev, y, color=color, label=label_mapping[phase])
			prev = y

		plt.plot(x, y, c='#555', lw=3)

		plt.legend(ncol=2)
		plt.xlim(x[0], x[-1])
		plt.ylim(0, 1.4*y.max())


	def plot_mass_cells_dna(self):
		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=(11, 3.5))

		plt.subplot(1, 2, 1)
		self.plot_mass(plot_DNA=False, fig=fig)
		plt.title("Cell population", fontsize=16, fontweight='demi')
		plt.ylim(0, 7)

		plt.subplot(1, 2, 2)
		self.plot_mass(plot_DNA=True, fig=fig)
		plt.title("Amount of DNA", fontsize=16, fontweight='demi')
		plt.ylim(0, 7)

		plt.suptitle("Replicate 1", fontsize=18, fontweight='demi', y=1.05)

	def plot_H(self, H=None, vmax=None, plot_S=False):

		from src.plot_helpers import color_for_key
		import matplotlib.pyplot as plt
		from src.figure_configs import FiguresConfig

		if H is None:
			H = self.H

		rg1_cols = self.get_Hpositions_for_phase('RG1')
		cg1_cols = self.get_Hpositions_for_phase('CG1')
		dg1_cols = self.get_Hpositions_for_phase('DG1')
		post_g1_cols = self.get_Hpositions_for_phase('postG1')
		s_cols = self.get_Hpositions_for_phase('S')
		g2m_cols = self.get_Hpositions_for_phase('G2M')

		H_cols = np.array([H.shape[1]-1])

		if plot_S:
			phases = ['H', 'RG1', 'CG1', 'DG1', 'S', 'G2M']
			cols_list = [H_cols, rg1_cols, cg1_cols, dg1_cols, s_cols, g2m_cols]
		else:
			phases = ['H', 'RG1', 'CG1', 'DG1', 'postG1']
			cols_list = [H_cols, rg1_cols, cg1_cols, dg1_cols, post_g1_cols]

		plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT_EXTRAWIDE)
		plt.subplot(1, 2, 1)

		vmax = H[:, :-1].max() if vmax is None else vmax
		plt.title("H convolution matrix", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		plt.imshow(H, vmax=vmax, aspect='auto', cmap='Reds',
				  extent=[0, H.shape[1], self.timepoints[-1], 0])

		plt.subplot(1, 2, 2)

		plt.title("Phase proportions over time", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)

		if H is None:
			x = self.timepoints
		else:
			x = np.arange(H.shape[0])

		prev = np.zeros(len(x))

		label_mapping = {
			'H': 'Halted',
			'RG1': 'Recovery G1',
			'CG1': 'Mother G1',
			'DG1': 'Daughter G1',
			'postG1': 'S/G2/M',
			'S': 'S',
			'G2M': 'G2/M',
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

		print("Proportion of the mother branch:")
		print("Recovery: ", recovery_smoothing_tps_length/top_smoothing_tps_length)
		print("Mother: ", top_smoothing_tps_length/top_smoothing_tps_length)
		print("Daughter: ", bottom_smoothing_tps_length/top_smoothing_tps_length)
		print()

def retrieve_phase_ticks(branch, config1, config2, with_labels=False):

	if branch == 'tb':
		t_tps = get_average_timepoints_for_branch(config1, config2, 't')
		b_tps = get_average_timepoints_for_branch(config1, config2, 'b')
		tps = (t_tps+b_tps)/2.
	else:
		tps = get_average_timepoints_for_branch(config1, config2, branch)

	# Use length of G1, S, and G2/M to delineate
	# timing of each phase
	g1_len = len(config1.get_Hpositions_for_phase('CG1'))
	s_len = len(config1.get_Hpositions_for_phase('S'))
	g2m_len = len(config1.get_Hpositions_for_phase('G2M'))

	# Compute locations for tick marks
	g1_mid = g1_len//2
	s_mid = s_len//2
	g2m_mid = g2m_len//2

	g1_start = tps[0]
	mid_g1_tp = tps[g1_mid]
	g1_end = tps[g1_len]
	s_mid = tps[g1_len+s_mid]
	s_end = tps[g1_len+s_len]
	g2m_mid = tps[g1_len+s_len+g2m_mid]
	g2m_end = tps[-1]

	phase_ticks = [mid_g1_tp, s_mid, g2m_mid]
	edge_ticks = [g1_start, g1_end, s_end, g2m_end]

	if with_labels:
		if branch == 'i': labels = ['RG1', 'S', 'G2/M']
		elif branch == 'tb': labels = ['mean G1', 'S', 'G2/M']
		else: raise ValueError()
		return phase_ticks, edge_ticks, labels

	else:
		return phase_ticks, edge_ticks


def read_cloccs_posteriors(posteriors_filepath):
	params = {}
	with open(posteriors_filepath, 'r') as f:
		lines = f.readlines()
		for line in lines[1:-1]:
			line_spl = line.split()
			params[line_spl[0]] = float(line_spl[1])

	return params


def load_timepoints(mode):
	if mode == 'chromatin':
		timepoints1 = GlobalConstants.CHROM_WT1_TIMEPOINTS
		timepoints2 = GlobalConstants.CHROM_WT2_TIMEPOINTS
	else:
		timepoints1 = GlobalConstants.EXPRESSION_WT1_TIMEPOINTS
		timepoints2 = GlobalConstants.EXPRESSION_WT2_TIMEPOINTS
	return timepoints1, timepoints2

def load_from_dic(filepath, replicate=None, config_type='distinct', mode='chromatin'):
	"""Load configs from json file from disk"""
	config = ModelConfig(config_type=config_type)

	timepoints1, timepoints2 = load_timepoints(mode)
	timepoints = timepoints1 if replicate == 1 else timepoints2
	config.load_from_dic(filepath, timepoints)
	config.replicate = replicate
	config.replication_parent_directory = DEFAULT_REPLICATION_PARENT_DIRECTORY

	return config


def load_default_configs(config_type='distinct', mode='chromatin'):

	# Using the CLOCCS configs, the refined model had some issues with fitting. e.g.
	# the narrower sigma0 made gave less room for smoothness during the first cell cycle. But
	# there may have been other details that needed to be resolved as well. Such as the number of 
	# indices to deconvolve, and the selection of alphas. However, with the default CLOCCS parameters
	# the fits for both expression and the chromatin make sense.

	#config1 = load_from_dic(f"models/yl_cell_cycle/refined_rep1.json", 1, config_type, mode)
	#config2 = load_from_dic(f"models/yl_cell_cycle/refined_rep2.json", 2, config_type, mode)
	#return config1, config2

	return load_cloccs_configs(config_type, mode)


def load_cloccs_configs(config_type='distinct', mode='chromatin'):

	# Load configs from disk
	config1 = load_from_dic(f"models/yl_cell_cycle/cloccs_rep1.json", 1, config_type, mode)
	config2 = load_from_dic(f"models/yl_cell_cycle/cloccs_rep2.json", 2, config_type, mode)

	return config1, config2


def retrieve_replication_timing(config1, config2, selected_replication_indices):
	"""todo: deprecated logic, this function essentially retrieves the average timepoints for
	a selected set of indices. Thus, replication_timing is not the appropriate function name.

	todo: rename to retrieve_average_timepoints_for_indices. Likely this is only used for replication timing
	loading at the moment. Since this is a frequent occurrence, a dedicated class or function for loading
	and keeping track of the replication timing and indices is appropriate. It will be relevant for origins, genes,
	and other genomic positions.

	"""
	rep1_timing = config1.timepoints_df.set_index('Hpos').loc[selected_replication_indices]
	rep2_timing = config2.timepoints_df.set_index('Hpos').loc[selected_replication_indices]
	mean_replication_timing = ((rep1_timing + rep2_timing)/2).mean(1) # mean of two replicates and the start and end
	return mean_replication_timing


def load_mean_dg1_mg1_length():
	from src.config import load_default_chrom_configs

	config1, config2 = load_default_chrom_configs()
	cg1_tps = config1.get_timepoints_for_phase('CG1')
	dg1_tps = config1.get_timepoints_for_phase('DG1')
	mean_g1_len = (-cg1_tps[0]+-dg1_tps[0])/2.
	return mean_g1_len


def get_average_timepoints_for_branch(config1, config2, branch):

	if branch == 'tb':
		t_tps = get_average_timepoints_for_branch(config1, config2, 't')
		b_tps = get_average_timepoints_for_branch(config1, config2, 'b')
		tps = (t_tps+b_tps)/2.
		return tps

	tp1 = config1.get_timepoints_for_branch(branch)
	tp2 = config2.get_timepoints_for_branch(branch)
	return (tp1+tp2)/2.

def get_average_timepoints_for_phase(config1, config2, phase):
	tp1 = config1.get_timepoints_for_phase(phase)
	tp2 = config2.get_timepoints_for_phase(phase)
	return (tp1+tp2)/2.


def load_default_expression_configs(config_type='distinct'):
	return load_default_configs(config_type=config_type, mode='expression')


def load_default_chrom_configs(config_type='distinct'):
	config1, config2 = load_default_configs(config_type=config_type, mode='chromatin')

	return config1, config2
