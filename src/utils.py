
import sys
import os
import psutil


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


def nearest(value, nearest, func, type_fun=float):
    return type_fun(func(float(value) / nearest) * nearest)


def nearest_span(span, round_to):
    """Round the span to the nearest round_to value"""
    import math
    span = (nearest(span[0], round_to, math.floor, int), 
            nearest(span[1], round_to, math.floor, int))
    return span


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


def print_memory_usage(label=""):
	process = psutil.Process(os.getpid())
	memory_info = process.memory_info()
	memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
	memory_gb = memory_mb / 1024  # Convert to GB
	print(f"Memory usage {label}: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
	sys.stdout.flush()


def round_nearest(x, round_to_value):
	return int(int(round(x/round_to_value)) * round_to_value)
