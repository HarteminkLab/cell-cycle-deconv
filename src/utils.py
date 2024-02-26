
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
        os.makedirs(directory)
    elif log:
        print_fl("Directory exists. Skipping.", end='')

    if log: print_fl()
