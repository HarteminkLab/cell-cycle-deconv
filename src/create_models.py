
import numpy as np
import pandas as pd


# Via Xin, beta is when the bud first appears in the bud flow model per the Orlando paper
# Because we are using Flow only, we do not have a beta value. So we refer to the previously reported beta value
# This is to describe the timepoint offsets between CG1/DG1 and PostG1
BETA_DEFAULT = 0.

class ModelCreation:
	"""This class is a port of the model creation from the original matlab code. There are some changes, 
	as we had to port to Python, but most of the meat is here. In our case, we are only concerned with 
	creating the RG1 model, so we will be refactoring this file to simplify things."""

	def __init__(self, posteriors_filepath, output_model_path):

		self.Rname = "RG1"
		self.CG1_intervals = "t 0"
		self.PG1_intervals = "i 1 t 1 b 1"

		self.output_model_path = output_model_path
		self.posteriors_filepath = posteriors_filepath
		self.alpha = 0

	def create_model(self, save=True):

		self.params, self.model_dic = self.create_model_rg1_model()
		self.model_cfg = self.get_model_cfg_str()

		if save:
			with open(self.output_model_path, 'w') as f:
				f.write(self.model_cfg)

			print(f"Created model and saved to file: {self.output_model_path}")

	def create_model_rg1_model(self):
		"""

		The RG1 model modeled after the original 1.2.1 model in the deconvolution code

		# ------------------
		# 1.2.1 MODEL (ALL_DIFF_G1, ibt)
		# ------------------ 
		my ($diff_g1_rg1, $diff_g1_cg1, $diff_g1_dg1, $diff_g1_postg1) = (192, 64, 192, 64);
		print OUT <<DIFF_G1;
				"1.2.1":{
					"RG1":[{"i":["-mu0","lambda*beta","$diff_g1_rg1"]}],
					"CG1":[{"t":["-alpha","lambda*beta","$diff_g1_cg1"]}],
					"DG1":[{"b":["-delta-alpha","lambda*beta","$diff_g1_dg1"]}],
					"postG1":[{"t":["lambda*beta","-alpha+lambda","$diff_g1_postg1"]},
						{"b":["lambda*beta","-alpha+lambda","$diff_g1_postg1"]},
						{"i":["lambda*beta","-alpha+lambda","$diff_g1_postg1"]}]
				},
		DIFF_G1
		"""

		params = read_cloccs_posteriors(self.posteriors_filepath)
		mu0, lambd, delta, sigma0, sigmav, halted = (params['mu0'], params['lambda'], 
											 params['delta'], params['sigma0'], params['sigmav'],
											 params['halted'])

		gamma1 = params['gamma1']
		gamma2 = params['gamma2']

		# Time between mother and daughter separation, previously 26/27 from xin.
		alpha = self.alpha

		# Previoulsy, beta was used from the budding index model. However, for the flow cytometry model
		# we do not have beta, but we will instead use the average between gamma1 and gamma2 to estimate
		# the S phase position

		beta = BETA_DEFAULT

		#position_of_s = beta*lambd
		position_of_s = gamma1*lambd

		ret_params = mu0, lambd, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted
		model_dic = {

			"RG1": [
				{"i":[mu0, position_of_s, 49]}],
			"CG1":[
				{"t":[-alpha, position_of_s, 49]}],
			"DG1":[
				{"b":[-delta-alpha, position_of_s, 49]}],
			"postG1":[
				{"t":[position_of_s, lambd-alpha, 79]},
				{"i":[position_of_s, lambd-alpha, 79]},
				{"b":[position_of_s, lambd-alpha, 79]}],
			}

		return ret_params, model_dic


	def get_sub_interval_str(self):
		model = self.model_dic
		subnames = ['R', 'RG1', 'CG1', 'DG1', 'postG1']
		branches = ['i', 't', 'b']

		intervals_str = ""
		for br in branches:
			intervals_str += f"# {br}\n"
			for subname in subnames:

				if subname not in model.keys(): 
					continue

				subintervals = model[subname]
				for subintervals_dict in subintervals:
					k, seq_params = list(subintervals_dict.items())[0]
					if k == br:
						interval = np.linspace(seq_params[0], seq_params[1], seq_params[2]+1)
						intervals_str += ' '.join([f"{x:.4f}" for x in interval]) + '\n'
		return intervals_str


	def get_model_cfg_str(self):

		Rname = self.Rname
		CG1_intervals = self.CG1_intervals
		PG1_intervals = self.PG1_intervals

		mu0, lambd, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted = self.params
		intervals = self.get_sub_interval_str()

		ret_str = """# lengths
mu0 %f
lambda %f
delta %f
sigma0 %f
sigmav %f
alpha %f
beta %f
gamma1 %f
gamma2 %f
halted %f
# description
%s i 0
CG1 %s
DG1 b 0
postG1 %s
%s""" % (-mu0, lambd, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted,
			 Rname, CG1_intervals, PG1_intervals, intervals)

		return ret_str


def read_cloccs_posteriors(posteriors_filepath):
	params = {}
	with open(posteriors_filepath, 'r') as f:
		lines = f.readlines()
		for line in lines[1:-1]:
			line_spl = line.split()
			params[line_spl[0]] = float(line_spl[1])
	return params


def create_wt1_model(output_model_path, alpha):
	posteriors_filepath = 'data/2019_cloccs_fits/yl_2019_replicate1/posteriors.txt'
	model_creator = ModelCreation(posteriors_filepath, output_model_path)
	model_creator.alpha = alpha
	model_creator.create_model()

	return output_model_path


def create_wt2_model(output_model_path, alpha):
	posteriors_filepath = 'data/2019_cloccs_fits/yl_2019_replicate2/posteriors.txt'
	model_creator = ModelCreation(posteriors_filepath, output_model_path)
	model_creator.alpha = alpha
	model_creator.create_model()
	return output_model_path


def main():

	create_wt1_model()
	create_wt2_model()

	# If we want to plot the resulting models, we can use this code
	from src.ModelFile import ModelFile
	model_file = ModelFile()
	model_file.load_model(output_model_path)
	model_file.plot_model()
	
