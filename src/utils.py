
import sys
import os


def parse_bool(s): return s.lower() in ['true', '1', 't', 'y', 'yes']


def print_fl(val='', end='\n', log=True):
	contents = str(val) + end
	sys.stdout.write(contents)
	sys.stdout.flush()


def mkdirs_safe(directories, log=True):
	if type(directories) is not list:
		raise ValueError(f"Invalid type: {type(directories)} "
			"for mkdirs safe, directories should be a list of strings.")
	for directory in directories:
		mkdir_safe(directory, log=log)


def mkdir_safe(directory, log=True):
	if log: print_fl("Creating directory: %s..." % directory, end='')

	if not os.path.exists(directory):

		try:
			os.makedirs(directory)
		except FileExistsError:
			pass

	elif log:
		print_fl("Directory exists. Skipping.", end='')

	if log: print_fl()


def run_cmd(bashCommand, stdout_file=None):
	if stdout_file is not None:
		with open(stdout_file, 'w') as output:
			process = subprocess.Popen(bashCommand.split(), stdout=output) 
			output, error = process.communicate()
	else:
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()

	return output, error


def save_print_df(df, path):
	df.to_csv(path)
	print(f"Saved to {path}")


def load_orf_data(path):
	import pandas as pd
	dat = pd.read_csv(path).set_index('orf_name')
	dat.columns = dat.columns.astype(int)
	return dat

# Saving a dictionary to disk
def save_dict_to_json(dictionary, filepath):
	import json
	with open(filepath, 'w') as file:
		json.dump(dictionary, file, indent=4)
	print(f"Dictionary saved to {filepath}")


# Loading a dictionary from disk
def load_dict_from_json(filepath):
	import json
	with open(filepath, 'r') as file:
		dictionary = json.load(file)
	return dictionary
