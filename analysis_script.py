#!/bin/python3
# analysis_script.py
# Corban Swain, 2018

import numpy as np
from scipy.stats import pearsonr, rankdata
import matplotlib.pyplot as plt
import os

data_dirname = 'data'
data_filename = 'signal_data.txt'
data_filepath = os.path.join(data_dirname, data_filename)
all_data = np.genfromtxt(data_filepath, dtype=None, delimiter='\t',
                         names=True, encoding=None)

# selected columns for analysis
cols = ['EGF_24H', 'HRG_24H', 'EGF_P', 'HRG_P']
X = np.column_stack((all_data[c] for c in cols))

# response data
y = np.array([0.10, 0.67, 0.63, 0.87])

# calculate the PCC for each row
pearson_out = [pearsonr(x_slice, y) for x_slice in X]
coeff, p_vals = (np.array(ls) for ls in zip(*pearson_out))

# benjamani hochberg procedure
def arg_bh_test(p, q=0.1):
    i = rankdata(p, method='ordinal')
    m = len(p)
    bh_crit = q * i / m
    # plt.plot(i, bh_crit, '.',  i, p_vals, '.')
    try:
        cutoff_idx = np.max(np.where(p <= bh_crit))
    except ValueError:
        return np.array([])
    signif_idx = np.where(i <= i[cutoff_idx])[0]
    return signif_idx[np.argsort(p_vals[signif_idx])]

q = 0.34
signif_idx = arg_bh_test(p_vals, q)

print('Part 1 - Table')
fmt = '%10s (%11s), %4s | %5.2f (%5.3f)'
for idx in signif_idx:
    print(fmt % (all_data['Protein'][idx],
                 all_data['Residue'][idx],
                 all_data['Time_Point'][idx],
                 coeff[idx], p_vals[idx]))