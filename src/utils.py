
import sys
import os


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
