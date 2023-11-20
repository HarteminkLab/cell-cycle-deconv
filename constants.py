"""This module defines project-level constants."""

import csv
import numpy as np

# Deconvolution argument (copied from calculations in the MATLAB code)
FS_FILE = 'deconv_args/Fs.csv'
W1_FILE = 'deconv_args/w1.csv'
W2_FILE = 'deconv_args/w2.csv'

FS = []
PADDING = 84
with open(FS_FILE) as f:
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
        FS.append(line)

F_INITIAL = np.array(FS[0], dtype=np.int32) - 1
F_TOP = np.array(FS[1], dtype=np.int32) - 1
F_BOTTOM = np.array(FS[2], dtype=np.int32) - 1

W1 = np.loadtxt(W1_FILE, delimiter=',', dtype=np.float64)
W2 = np.loadtxt(W2_FILE, delimiter=',', dtype=np.float64)

H_FILE = 'deconv_args/H.csv'
H = np.loadtxt(H_FILE, delimiter=',', dtype=np.float64)

WT1_INITIAL_H1 = np.loadtxt('deconv_args/wt1_initialH1.csv', delimiter=',', dtype=np.float64)
WT1_INITIAL_H2 = np.loadtxt('deconv_args/wt1_initialH2.csv', delimiter=',', dtype=np.float64)
WT1_INITIAL_H3 = np.loadtxt('deconv_args/wt1_initialH3.csv', delimiter=',', dtype=np.float64)

WT1_BOTTOM_H1 = np.loadtxt('deconv_args/wt1_bottomH1.csv', delimiter=',', dtype=np.float64)
WT1_BOTTOM_H2 = np.loadtxt('deconv_args/wt1_bottomH2.csv', delimiter=',', dtype=np.float64)

WT1_TOP_H1 = np.loadtxt('deconv_args/wt1_topH1.csv', delimiter=',', dtype=np.float64)
WT1_TOP_H2 = np.loadtxt('deconv_args/wt1_topH2.csv', delimiter=',', dtype=np.float64)

WT1_H = np.loadtxt('deconv_args/wt1_H.csv', delimiter=',', dtype=np.float64)

WT2_INITIAL_H1 = np.loadtxt('deconv_args/wt2_initialH1.csv', delimiter=',', dtype=np.float64)
WT2_INITIAL_H2 = np.loadtxt('deconv_args/wt2_initialH2.csv', delimiter=',', dtype=np.float64)
WT2_INITIAL_H3 = np.loadtxt('deconv_args/wt2_initialH3.csv', delimiter=',', dtype=np.float64)

WT2_BOTTOM_H1 = np.loadtxt('deconv_args/wt2_bottomH1.csv', delimiter=',', dtype=np.float64)
WT2_BOTTOM_H2 = np.loadtxt('deconv_args/wt2_bottomH2.csv', delimiter=',', dtype=np.float64)

WT2_TOP_H1 = np.loadtxt('deconv_args/wt2_topH1.csv', delimiter=',', dtype=np.float64)
WT2_TOP_H2 = np.loadtxt('deconv_args/wt2_topH2.csv', delimiter=',', dtype=np.float64)

WT2_H = np.loadtxt('deconv_args/wt2_H.csv', delimiter=',', dtype=np.float64)