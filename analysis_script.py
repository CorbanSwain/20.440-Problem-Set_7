#!/bin/python3
# analysis_script.py
# Corban Swain, 2018

import numpy as np
from scipy.stats import pearsonr, zscore
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import os

data_dirname = 'data'
data_filename = 'signal_data.txt'
data_filepath = os.path.join(data_dirname, data_filename)
all_data = np.genfromtxt(data_filepath, dtype=None, delimiter='\t',
                         names=True, encoding=None)

# selected columns for analysis
cols = ['EGF_24H', 'HRG_24H', 'EGF_P', 'HRG_P']
X = np.column_stack(all_data[c] for c in cols)

# response data
Y = np.array([0.10, 0.67, 0.63, 0.87])

# Part 1 - Pearson Correlation Coefficients
# calculate the PCC for each row
pearson_out = [pearsonr(x_slice, Y) for x_slice in X]
coeff, p_vals = (np.array(lst) for lst in zip(*pearson_out))

# calculate R^2
rsq = np.power(coeff, 2)

# choose a cutoff value and select signals with R^2
# above the cutoff
cutoff = 0.95
select = np.where(rsq > cutoff)[0]
select_sorted = select[np.argsort(rsq[select])[::-1]]

# print the identified conditions
print('Part 1 - Highest Pearson Correlation Coeff.')
fmt = '%2s  |  %10s  |  %5.3f'
for i, idx in enumerate(select_sorted):
    print(fmt % (i+1, all_data['Protein'][idx],
                 coeff[idx]))

# Part 2 - Partial Least Squares Regression
Yt = zscore(Y.reshape((-1, 4)).T)
Xt = zscore(X.T)
pls = PLSRegression(n_components=3)
pls.fit(Xt, Yt)
XS, XL, YS, YL = (pls.x_scores_, pls.x_loadings_,
                  pls.y_scores_, pls.y_loadings_)

xy = (np.arange(-100, 100) * 0, np.arange(-100, 100))


def plot_xy_axes():
    """Plots the lines x=0 and y=0 in black."""
    plt.autoscale(False)
    plt.plot(xy[0], xy[1], 'k-', zorder=0, linewidth=0.75)
    plt.plot(xy[1], xy[0], 'k-', zorder=0, linewidth=0.75)

plt.style.use('seaborn-notebook')
fignum = 0
plt.figure(fignum, (20, 7))
ax = plt.subplot(131)
plt.scatter(XS[:, 0], XS[:, 1], label='Signal')
plt.scatter(YS[:, 0], YS[:, 1], label='Response')
for i, col in enumerate(cols):
    pos = (np.mean([XS[i, 0], YS[i, 0]]),
           np.mean([XS[i, 1], YS[i, 1]]))
    a = ax.annotate(col, xy=pos, xycoords='data',
                    xytext=(0, 15), textcoords='offset points',
                    va='center', ha='center',
                    bbox=dict(boxstyle='round', fc='w', lw=0.75))
plot_xy_axes()
ax.legend(loc='lower right', frameon=False)
plt.xlabel("Principal Axis 1")
plt.ylabel("Principal Axis 2")

ax = plt.subplot(132)
plt.scatter(XL[:, 0], XL[:, 1])
plt.scatter(YL[:, 0], YL[:, 1])
plot_xy_axes()
plt.xlabel("Principal Axis 1")
plt.ylabel("Principal Axis 2")


# RE: the "strongly associated subset" in problem 2b, TAs have said that
# instead  of measuring proximity to the response using Euclidean distance,
# instead find the dot product of each signal and the response point and pick
#  the values with the highest magnitude (positive or negative), because
# Euclidean distance will give you values that are orthogonal to the response
#  vector and that doesn't actually show association using these methods
prod = np.dot(XL, YL.T).squeeze()
prod_norm = np.power(prod, 2) / max(np.power(prod, 2))
selects = np.where(prod_norm > 0.90)[0]
selects = selects[np.argsort(prod_norm[selects])[::-1]]
ax = plt.subplot(133)
plt.scatter(XL[selects, 0], XL[selects, 1])
plt.scatter(YL[:, 0], YL[:, 1])
plt.autoscale(False)
plt.scatter(XL[:, 0], XL[:, 1], alpha=0.1, c='k',
            zorder=0.5)
print('\nPart-2: Partial Least Squares Regression')
nudge = {'An A2, 10m': (50, -55),
         'An A2, 30m': (0, -25),
         'EGFR, 0m': (0, 25),
         'Erbin, 5m': (10, -35),
         'Erbin, 10m': (0, 25),
         'Erbin, 30m': (-10, -5),
         'HER2, 5m': (50, 30),
         'HER2, 10m': (0, 15),
         'HER2, 30m': (-25, 0),
         'IGF1R, 5m': (0, 30),
         'ephrin-B2, 10m': (0, 20),
         'EphB1, 5m': (0, 15)}
for i, s in enumerate(selects):
    key = '%s, %s' % (all_data['Protein'][s],
                      all_data['Time_Point'][s])
    description = '(%d) %s-%s' % (i + 1,
                                  all_data['Protein'][s],
                                  all_data['Time_Point'][s])
    n = nudge.get(key, (0, 0))
    pos = (XL[s, 0], XL[s, 1])
    offset = (-25 + n[0], -15 + n[1])
    ax.annotate(description,
                xy=pos,
                xycoords='data',
                xytext=offset,
                textcoords='offset points',
                va='bottom',
                ha='right',
                arrowprops=dict(arrowstyle="->", lw=0.75,
                                connectionstyle="arc3"))
plot_xy_axes()
plt.xlabel("Principal Axis 1")
plt.ylabel("Principal Axis 2")
plt.savefig('figures/plsr.png', transparent=True)
plt.show()

# calulate a prediction using only the first two principle components
prediction = np.dot(XS[:, 0:2], XL[:, 0:2].T)
fignum = fignum + 1
plt.figure(fignum, (8, 8))
for x_real, x_pred, col_name in zip(Xt, prediction, cols):
    plt.scatter(x_real, x_pred, label=col_name, alpha=0.8)
plot_xy_axes()
plt.plot(np.arange(-100, 100), np.arange(-100, 100), ':',
         c='k', alpha=0.5, zorder=0)
plt.xlabel('Observed Value')
plt.ylabel('Predicted Value')
plt.legend(frameon=False)
plt.savefig('figures/plsr_predict.png', transparent=True)
plt.show()