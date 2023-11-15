"""This module defines project-level constants."""

import csv
import numpy as np

# Deconvolution argument (copied from calculations in the MATLAB code)
FS_FILE = 'deconv_args/Fs.csv'
H_FILE = 'deconv_args/H.csv'
W1_FILE = 'deconv_args/W1.csv'
W2_FILE = 'deconv_args/W2.csv'
W3_FILE = 'deconv_args/W3.csv'

FS = []
PADDING = 84
with open(FS_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        FS.append(line)

F_INITIAL = np.array(FS[0], dtype=np.int32) - 1
F_TOP = np.array(FS[1], dtype=np.int32) - 1
F_BOTTOM = np.array(FS[2], dtype=np.int32) - 1

H = np.loadtxt(H_FILE, delimiter=',', dtype=np.float64)
W1 = np.loadtxt(W1_FILE, delimiter=',', dtype=np.float64)
W2 = np.loadtxt(W2_FILE, delimiter=',', dtype=np.float64)
W3 = np.loadtxt(W3_FILE, delimiter=',', dtype=np.float64)

F_PADDED_FILE = 'deconv_args/F_padded.csv'
F_PADDED = []
with open(F_PADDED_FILE) as f:
    F_PADDED = np.array(f.read().split('\n'), dtype=np.float64)
