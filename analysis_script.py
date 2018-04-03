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
coeff, p_vals = (np.array(lst) for lst in zip(*pearson_out))


# benjamani hochberg procedure
def argbenhoch(p, q=0.1, plot=False):
    i = rankdata(p, method='ordinal')
    m = len(p)
    bh_crit = q * i / m
    if plot:
        s = np.argsort(i)
        plt.plot(i[s], bh_crit[s],  i[s], p_vals[s])
        plt.xlabel('Rank')
        plt.ylabel('P Value')
        plt.xlim([0, m])
        plt.ylim([0, 1])

    try:
        cutoff_i = np.max(i[np.where(p <= bh_crit)[0]])
    except ValueError:
        return np.array([])
    signif_idx = np.where(i <= cutoff_i)[0]
    return signif_idx[np.argsort(p_vals[signif_idx])]


# calculate with q value
q = 0.34
signif_idx = argbenhoch(p_vals, q, plot=True)

print('Part 1 - Table')
fmt = '%2s | %10s (%11s), %4s | %5.2f (%5.3f)'
for i, idx in enumerate(signif_idx):
    print(fmt % (i+1, all_data['Protein'][idx],
                 all_data['Residue'][idx],
                 all_data['Time_Point'][idx],
                 coeff[idx], p_vals[idx]))

plt.show()