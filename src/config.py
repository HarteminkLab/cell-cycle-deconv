
import pandas as pd
import numpy as np
import os

class Config:
	"""
	A config class to read data and initialize the model.
	"""

	def __init__(self, wt1=None, wt2=None, wt1_timepoints=None, wt2_timepoints=None, 
			model_wt1_file=None, model_wt2_file=None, name=None, model_wt1_lines=None,
			model_wt2_lines=None, replicate=None, alpha=None):

		self.name = name
		self.replicate = replicate
		self.alpha = alpha

		# Replicate 1 configuration
		if wt1 is not None:
			self.wt1_df = wt1
			self.WT1_TIMEPOINTS = wt1.columns.values.astype(int)
			self.model_wt1_file = model_wt1_file

			self.wt2_df = None
			self.WT2_TIMEPOINTS = None
			self.model_wt2_file = None

		# TODO: Assuming we are using the configuration without the data.
		# Currently this is used for deconvolving the chromatin in which
		# the data is handled by the chromatin_grid_compute.py class
		else:
			self.wt1_df = None
			self.WT1_TIMEPOINTS = wt1_timepoints

			self.model_wt2_file = None
			self.wt2_df = None
			self.WT2_TIMEPOINTS = None

		if model_wt1_file is not None:
			self.intervals_wt1 = self.read_model_format(model_wt1_file)
		elif model_wt1_lines is not None:
			self.intervals_wt1 = self.read_model_lines(model_wt1_lines)

		# Replicate 2 configuration
		if wt2 is not None:
			self.WT2_TIMEPOINTS = wt2.columns.values.astype(int)
			self.wt2_df = wt2
			self.model_wt2_file = model_wt2_file
			
			if model_wt2_file is not None:
				self.intervals_wt2 = self.read_model_format(model_wt2_file)
			elif model_wt2_lines is not None:
				self.intervals_wt2 = self.read_model_lines(model_wt2_lines)

		# Boolean flag to indicate whether we have 1 or 2 replicates
		self.has_two_replicates = wt2 is not None

		self.create_helper_structures()

	def all_orfs(self):
		return self.wt1_df.index.values

	def read_model_format(self, modelfile):
		if not os.path.exists(modelfile):
			raise FileNotFoundError(f"The model file {modelfile} does not exist")

		with open(modelfile, 'r') as f:
			lines = f.readlines()
			ret = self.read_model_lines(lines)

		return ret

	def read_model_lines(self, f):

		LENGTHS, DESCRIPTION, I, T, B = '# lengths', '# description', '# i', '# t', '# b' 
		lengths, relations, initial_tps, top_tps, bottom_tps, parseFlag = [], [], [], [], [], -1

		for line in f:
			line = line.strip()
			if not line:
				continue
			if line == LENGTHS:
				parseFlag = 1
			elif line == DESCRIPTION:
				parseFlag = 2
			elif line == I:
				parseFlag = 3
			elif line == T:
				parseFlag = 4
			elif line == B:
				parseFlag = 5
			# lengths
			elif parseFlag == 1:
				value = self.parse_lengths(line)
				lengths.append(value)
			# description
			elif parseFlag == 2:
				relation = line.split(' ')
				relations.append(relation)
			# interval i
			elif parseFlag == 3:
				interval = np.array(line.split(' '), dtype=np.float64)
				initial_tps.append(interval)
			# interval t
			elif parseFlag == 4:
				interval = np.array(line.split(' '), dtype=np.float64)
				top_tps.append(interval)
			# interval b
			elif parseFlag == 5:
				interval = np.array(line.split(' '), dtype=np.float64)
				bottom_tps.append(interval)

		initial_phase_map, top_phase_map, bottom_phase_map = {}, {}, {}
		for i, relation in enumerate(relations):
			notation = relation[0]
			for idx in range(1, len(relation)-1, 2):
				label = relation[idx]
				num = relation[idx+1]
				if label == 'i':
					initial_phase_map[num] = (notation, i)
				elif label == 't':
					top_phase_map[num] = (notation, i)
				elif label == 'b':
					bottom_phase_map[num] = (notation, i)

		return lengths, relations, initial_tps, top_tps, bottom_tps, (initial_phase_map, top_phase_map, bottom_phase_map)

	def parse_lengths(self, line):
		segments = line.split(' ')
		if segments[0] in ('mu0', 'lambda', 'delta', 'sigma0', 'sigmav', 'alpha', 'beta', 'gamma1', 'gamma2', 'halted'):
			value = float(segments[1])
		else:
			raise ValueError(f'Wrong parameter {segments[0]} in line {line}')
		return value


	def create_helper_structures(self):
		"""
		We will create some dataframes and dictionaries that will help with looking up branch/phase subsets.
		"""

		# Assume we can just use wt2's model config (that wt1 has the same defined intervals)
		lengths, relations, initial_tps, top_tps, bottom_tps, \
		(initial_phase_map, top_phase_map, bottom_phase_map) = self.intervals_wt1

		def get_branch_timepoints_by_index(branch, phase_tp_index):

			if branch == 'i':
				ret = initial_tps[phase_tp_index]
			elif branch == 't':
				ret = top_tps[phase_tp_index]
			elif branch == 'b':
				ret = bottom_tps[phase_tp_index]

			return ret

		all_phases = []
		all_branches = []
		all_branch_indices = []
		all_timepoints = []

		for item in relations:
			
			phase = item[0]
			
			intervals = item[1:]

			for interval_idx in range(0, len(intervals), 2):
				branch, phase_tp_index = intervals[interval_idx], int(intervals[interval_idx+1])

				branch, phase_tp_index

				timepoints = get_branch_timepoints_by_index(branch, phase_tp_index)

				all_phases = all_phases + ([phase]*len(timepoints))
				all_branches = all_branches + ([branch]*len(timepoints))
				all_branch_indices = all_branch_indices + ([phase_tp_index]*len(timepoints))
				all_timepoints = all_timepoints + list(timepoints)

		phase_branch_tp_df = pd.DataFrame({'phase': all_phases, 'branch': all_branches, 
					  'tp_index': all_branch_indices, 'timepoint': all_timepoints})
		phase_branch_tp_df = phase_branch_tp_df.set_index(['phase', 'branch'])
		self.phase_branch_tp_df = phase_branch_tp_df

		# --------------------------

		# Construct a dictionary that will allow us to retrieve the indices in the H matrix
		# for the requested cell phase
		phase_columns = {}
		last = 0
		num_columns = 0
		for relation in relations:
			phase = relation[0]
			first_branch = phase_branch_tp_df.loc[phase].index.unique()[0]
			first_branch_tps = phase_branch_tp_df.loc[phase].loc[first_branch]

			current_length = len(first_branch_tps)-1

			phase_columns[relation[0]] = np.arange(last, last+current_length)
			last = last+current_length
			num_columns = last

		self.phase_columns = phase_columns
		self.num_columns = num_columns

	def get_phase_timepoints_for_plotting(self):

		cg1_timepoints = self.get_timepoints_phases_Hpositions_for_branch('t')[0][1].values
		postcg1_timepoints = self.get_timepoints_phases_Hpositions_for_branch('t')[1][1].values
		dg1_timepoints = self.get_timepoints_phases_Hpositions_for_branch('b')[0][1].values
		postdg1_timepoints = self.get_timepoints_phases_Hpositions_for_branch('b')[1][1].values

		# Append end of G1 for contiguous timepoints for plotting
		cg1_timepoints = np.concatenate([cg1_timepoints, postcg1_timepoints[0:1]])
		dg1_timepoints = np.concatenate([dg1_timepoints, postdg1_timepoints[0:1]])

		# Calculate S-phase
		gamma1, gamma2 = self.intervals_wt1[0][7], self.intervals_wt1[0][8]
		lambda_val = self.intervals_wt1[0][1]

		s_start, s_end = lambda_val*gamma1, lambda_val*gamma2

		c_s_timepoints = postcg1_timepoints[postcg1_timepoints < s_end]
		c_g2m_timepoints = postcg1_timepoints[postcg1_timepoints >= s_end]

		# contiguous plotting
		c_s_timepoints = np.concatenate([c_s_timepoints, c_g2m_timepoints[:1]])

		d_s_timepoints = postdg1_timepoints[postdg1_timepoints < s_end]
		d_g2m_timepoints = postdg1_timepoints[postdg1_timepoints >= s_end]

		# contiguous plotting
		d_s_timepoints = np.concatenate([d_s_timepoints, d_g2m_timepoints[:1]])

		return (cg1_timepoints, c_s_timepoints, c_g2m_timepoints), \
			   (dg1_timepoints, d_s_timepoints, d_g2m_timepoints)



	def get_timepoints_for_branch(self, branch):
		"""
		This function will return the timepoints as an array for a branch.

		Basically consolidating the timepoints for  each phase within the branch.

		This is useful for the rescaling function for computing PTR, which takes in
		the timepoints as a single array as input.
		"""

		timepoints = np.array([])
		for phase, timepoints_series, indices in self.get_timepoints_phases_Hpositions_for_branch(branch):    
			current_timepoints = timepoints_series.values
			timepoints = np.concatenate([timepoints, current_timepoints])

		timepoints

		return timepoints


	def get_timepoints_phases_Hpositions_for_branch(self, branch):
		"""
		A bit complicated, but this method is for plotting. 
		
		We will want the phases, the timepoints, and the indices in H (also in f).
		
		There is probably a cleaner way to do this, but we will just use the dataframe
		of all timepoints to do this.
		"""

		# Then we will want the phases and the indices for a branch
		search_df = self.phase_branch_tp_df.reset_index()
		phases_for_branch = search_df[search_df.branch == branch].phase.unique()

		branch_indices = []
		for phase in phases_for_branch:
			timepoints = search_df[(search_df.phase == phase) & 
								   (search_df.branch == branch)].timepoint
			branch_indices.append((phase, timepoints[:-1], self.phase_columns[phase]))

		return branch_indices


	def get_branch_phase_mapping(self):
		"""Get a mapping from branch namem to a list of the phases within the branch. Useful
		for knowing which index of the timepoints list represents which phase"""

		relations = self.intervals_wt1[1]

		branch_phase_mapping = {
			'i': [],
			't': [],
			'b': [],
		}

		for phase_list in relations:
			phase = phase_list[0]
			index_pairs = phase_list[1:]
			
			for index_ind in range(0, len(index_pairs), 2):
				branch = index_pairs[index_ind]
				tp_index = index_pairs[index_ind+1]
				
				cur_map = branch_phase_mapping[branch]
				cur_map.append(phase)

				branch_phase_mapping[branch] = cur_map
		return branch_phase_mapping


	def get_Hpositions_for_branch(self, branch):
		rg1_indices = self.get_timepoints_phases_Hpositions_for_branch('i')[0][2]
		cg1_indices = self.get_timepoints_phases_Hpositions_for_branch('t')[0][2]
		dg1_indices = self.get_timepoints_phases_Hpositions_for_branch('b')[0][2]
		postg1_indices = self.get_timepoints_phases_Hpositions_for_branch('b')[1][2]

		Hpositions_dic = {
			'i': np.concatenate([rg1_indices, postg1_indices]),
			't': np.concatenate([cg1_indices, postg1_indices]),
			'b': np.concatenate([dg1_indices, postg1_indices])
		}
		return Hpositions_dic[branch]

	def get_Hpositions_for_phase(self, phase):
		"""
		TODO: This is strictly for the RG1 model with hard-coded locations for each phase
		in each branch. 

		Refactor this if we start using other models. This method is used for plotting purposes.
		"""

		dg1_indices = self.get_timepoints_phases_Hpositions_for_branch('b')[0][2]
		postg1_indices = self.get_timepoints_phases_Hpositions_for_branch('b')[1][2]
		cg1_indices = self.get_timepoints_phases_Hpositions_for_branch('t')[0][2]
		rg1_indices = self.get_timepoints_phases_Hpositions_for_branch('i')[0][2]

		Hpositions_dic = {
			'DG1': dg1_indices,
			'postG1': postg1_indices,
			'CG1': cg1_indices,
			'RG1': rg1_indices,
		}
		return Hpositions_dic[phase]


	def get_geneset_df(self):
		"""
		Returns the geneset data frame in the same order as the raw gene expression is defined.

		The original matlab data had the orf names and the gene expression in separate files, so here
		we will create a dataframe with the orf names ordering as well as any other gene data
		we may need.

		Note that some of the gene information may note exist (nas).
		"""

		from src.sgd import read_sgd_genes

		genelist_orfs = pd.DataFrame(self.orf_index_map.items())
		genelist_orfs.columns = ['orf_name', 'data_idx']
		genelist_orfs = genelist_orfs.set_index('orf_name')
		genelist_orfs = genelist_orfs.iloc[:-1] # last row is empty for some reason

		genes = read_sgd_genes()
		genes.loc[genes['gene'].isna(), 'gene'] = genes[genes['gene'].isna()].index

		genelist_orfs = genelist_orfs.join(genes).sort_values('data_idx')

		return genelist_orfs


def read_yl_vst_data_rep(replicate):
	wt_data = pd.read_csv(f'datasets/yl_cell_cycle/replicate{replicate}_deseq2_vst_counts.csv')
	wt_data = wt_data.rename(columns={"Unnamed: 0": "orf_name"}).set_index('orf_name')
	wt_data.columns = [int(s.replace('X', '')) for s in wt_data.columns.values]
	return wt_data


def load_yl_replicate1_rg1_alpha_vst_config(alpha=28):
	"""Load the model in which alpha is set to delay between separation and cytokinesis"""
	wt1 = read_yl_vst_data_rep(1)
	model_wt1_file = f'models/yl_cell_cycle/wt1_rg1.{alpha}.label'
	config = Config(wt1=wt1, model_wt1_file=model_wt1_file, name=f'Replicate 1, $\\alpha$={alpha}', 
		replicate=1, alpha=alpha)

	return config

def load_yl_replicate2_rg1_alpha_vst_config(alpha=22):
	"""Load the model in which alpha is set to delay between separation and cytokinesis"""
	wt2 = read_yl_vst_data_rep(2)

	# model file
	model_wt2_file = f'models/yl_cell_cycle/wt2_rg1.{alpha}.label'
	config = Config(wt1=wt2, model_wt1_file=model_wt2_file, name=f'Replicate 2, $\\alpha$={alpha}',
		replicate=2, alpha=alpha)
	return config


def load_yl_rg1_vst_config(replicate):
	if replicate == 1:
		config = load_yl_replicate1_rg1_alpha_vst_config()
	else:
		config = load_yl_replicate2_rg1_alpha_vst_config()
	return config


def load_combined_yl_alpha_vst_gene_expression_config(alphas=[28, 22]):

	WT1 = read_yl_vst_data_rep(1)
	WT2 = read_yl_vst_data_rep(2)

	# model files
	model_wt1_file = f'models/yl_cell_cycle/wt1_rg1.{alphas[0]}.label'
	model_wt2_file = f'models/yl_cell_cycle/wt2_rg1.{alphas[1]}.label'

	config = Config(wt1=WT1, wt2=WT2, model_wt1_file=model_wt1_file, 
		model_wt2_file=model_wt2_file, name=f'Combined, $\\alpha$={alphas[0]},{alphas[1]}')

	return config


def read_xin_published_wt_data(wildtype):    
	# Handle columns and rows, second row has clock time, drop alias columns
	wt1_web_df = pd.read_csv(f'datasets/datasets_from_web_deconvolution.cs.duke.edu/wildtype{wildtype}.tsv', 
		sep='\t')
	wt1_web_df.columns = wt1_web_df.iloc[0]
	wt1_web_df = wt1_web_df.rename(columns={'byClock': 'orf_name'}).set_index('orf_name')
	wt1_web_df = wt1_web_df[wt1_web_df.columns[3:]]
	wt1_web_df = wt1_web_df.iloc[1:]
	return wt1_web_df


def get_yl2019_chromatin_timepoints(replicate):
	"""todo: refactoring to use this function instead of 
	lazy loading the timepoints from the mnase reads"""

	if replicate == 1:
		return np.array([ 0, 10, 20, 30, 40, 50, 60, 70, 
			80, 90, 100, 110, 120, 130, 140, 150])	
	elif replicate == 2:
		return np.array([ 0, 10, 20, 30, 40, 50, 60, 70, 
			80, 90, 100, 110, 120, 130, 140])
