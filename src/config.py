import numpy as np
import os

class Config:
	"""
	A config class to read data and initialize the model.
	"""

	def __init__(self, wt1_tp, wt2_tp, data_wt1_file, data_wt2_file, gene_mapping_file, gene_set_file, model_wt1_file, model_wt2_file):

		self.WT1_TIMEPOINTS = wt1_tp
		self.WT2_TIMEPOINTS = wt2_tp

		# read data files
		self.data_wt1 = np.loadtxt(data_wt1_file, delimiter='\t', dtype=np.float64)
		self.data_wt2 = np.loadtxt(data_wt2_file, delimiter='\t', dtype=np.float64)

		# load gene/orf mappings
		self.gene_orf_map = self.read_gene_orf_map(gene_mapping_file)
		self.orf_index_map = self.read_orf_index_map(gene_set_file)

		# read model data
		self.intervals_wt1 = self.read_model_format(model_wt1_file)
		self.intervals_wt2 = self.read_model_format(model_wt2_file)

	def read_gene_orf_map(self, gene_mapping_file):
		map = {}
		
		with open(gene_mapping_file) as f:
			for line in f:
				(key, val) = line.strip().split('\t')
				map[key] = val
		return map

	def read_orf_index_map(self, gene_set_file):
		map = {}
		with open(gene_set_file) as f:
			map = {orf: index for index, orf in enumerate(f.read().split('\n'))}
		return map
	
	def read_model_format(self, modelfile):
		if not os.path.exists(modelfile):
			raise FileNotFoundError(f"The model file {modelfile} does not exist")

		LENGTHS, DESCRIPTION, I, T, B = '# lengths', '# description', '# i', '# t', '# b' 
		lengths, relations, initial_tps, top_tps, bottom_tps, parseFlag = [], [], [], [], [], -1

		with open(modelfile, 'r') as f:
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
		if segments[0] in ('mu0', 'lambda', 'delta', 'sigma0', 'sigmav', 'alpha', 'beta'):
			value = float(segments[1])
		else:
			raise ValueError(f'Wrong parameter {segments[0]} in line {line}')
		return value



def load_yl_replicate2_gene_expression_config():

	# Time points
	WT1_TP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140]
	WT2_TP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140]

	# dataset files
	DATA_WT1_FILE = 'datasets/yl_cell_cycle/replicate2_gene_expression.txt'
	DATA_WT2_FILE = 'datasets/yl_cell_cycle/replicate2_gene_expression.txt'

	GENE_MAPPING_FILE = 'datasets/yl_cell_cycle/gene_to_orf_name_mapping.txt'
	GENE_SET_FILE = 'datasets/yl_cell_cycle/genes.lst'

	# model files
	MODEL_WT1_FILE = 'models/yl_cell_cycle/wt2_rg1.label'
	MODEL_WT2_FILE = 'models/yl_cell_cycle/wt2_rg1.label'

	config = Config(WT1_TP, WT2_TP, 
		DATA_WT1_FILE, DATA_WT2_FILE,
		GENE_MAPPING_FILE, GENE_SET_FILE,
		MODEL_WT1_FILE, MODEL_WT2_FILE)

	return config


def load_xg_gene_expression_config():

	# Timepoints
	WT1_TP = [x for x in range(30, 255, 16)]
	WT2_TP = [x for x in range(38, 263, 16)]

	# dataset files
	DATA_WT1_FILE = 'datasets/original_budflow/replicate1_gene_expression.txt'
	DATA_WT2_FILE = 'datasets/original_budflow/replicate2_gene_expression.txt'

	GENE_MAPPING_FILE = 'datasets/original_budflow/gene_to_orf_name_mapping.txt'
	GENE_SET_FILE = 'datasets/original_budflow/genes.lst'

	# model files
	MODEL_WT1_FILE = 'models/original_budflow/wt1_budflow/1.1.1.26.label'
	MODEL_WT2_FILE = 'models/original_budflow/wt2_budflow/1.1.1.27.label'

	config = Config(WT1_TP, WT2_TP, 
		DATA_WT1_FILE, DATA_WT2_FILE,
		GENE_MAPPING_FILE, GENE_SET_FILE,
		MODEL_WT1_FILE, MODEL_WT2_FILE)

	return config
