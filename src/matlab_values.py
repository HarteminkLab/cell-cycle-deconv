"""This module defines project-level constants."""

import csv
import numpy as np

# Deconvolution argument (copied from calculations in the MATLAB code)
F_FILE = 'deconv_args/f.csv'

F_FINAL = []
with open(F_FILE) as f:
    reader = csv.reader(f, delimiter='\n')
    for line in reader:
        F_FINAL.append(line)
F_FINAL = np.array(F_FINAL, dtype=np.float64)

PRED_G = []
with open('deconv_args/pred_g.csv') as f:
    reader = csv.reader(f, delimiter='\n')
    for line in reader:
        PRED_G.append(line)
PRED_G = np.array(PRED_G, dtype=np.float64)
