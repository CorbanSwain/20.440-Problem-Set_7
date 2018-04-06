#!/bin/python3
# analysis_script.py
# Corban Swain, 2018

# load in necessary packages
import os
import numpy as np
from scipy.stats import pearsonr, zscore
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# importing data
data_dirname = 'data'
data_filename = 'signal_data.txt'
data_filepath = os.path.join(data_dirname, data_filename)
all_data = np.genfromtxt(data_filepath, dtype=None, delimiter='\t',
                         names=True, encoding=None)

# selected columns for analysis
conds = ['EGF_24H', 'HRG_24H', 'EGF_P', 'HRG_P']
X = np.column_stack(all_data[c] for c in conds)

# response data
Y = np.array([0.10, 0.67, 0.63, 0.87])

# Part 1 - Pearson Correlation Coefficients
# calculate the PCC for each row
pearson_out = [pearsonr(x_slice, Y) for x_slice in X]
corr_coeff, _ = (np.array(lst) for lst in zip(*pearson_out))

# calculate R^2
rsq = np.power(corr_coeff, 2)

# choose a cutoff value and select signals with R^2 above the cutoff
cutoff = 0.95
select = np.where(rsq > cutoff)[0]
select_sorted = select[np.argsort(rsq[select])[::-1]]


# a function to print the signal identity and some coefficient value for a
# selection of indices
def list_coeff(idxs, coeff):
    fmt = '%2s  |  %10s, %3s  |  %5.3f'
    for i, idx in enumerate(idxs):
        print(fmt % (i + 1, all_data['Protein'][idx],
                     all_data['Time_Point'][idx], coeff[idx]))


# print the identified conditions
print('Part 1 - Highest Pearson Correlation Coeff.')
list_coeff(select_sorted, corr_coeff)

# Part 2 - Partial Least Squares Regression (PLSR)
# standardize (by z-score) and reshape each the signal and response data
Xt = zscore(X.T)
Yt = zscore(Y.reshape((-1, 4)).T)

# build a PLSR regression model from the data with two components
pls = PLSRegression(n_components=2)
pls.fit(Xt, Yt)

# extract the scores and loadings from the model
XS, XL, YS, YL = (pls.x_scores_, pls.x_loadings_,
                  pls.y_scores_, pls.y_loadings_)


# function to plot the x and y axes
def plot_xy_axes(a=-100, b=100):
    """Plots the lines x=0 and y=0 in black."""
    v = np.array([a, b])
    zero_v = np.zeros((2,))
    plt.autoscale(False)
    plt.plot(v, zero_v, 'k-', zorder=0, linewidth=0.75)
    plt.plot(zero_v, v, 'k-', zorder=0, linewidth=0.75)


# function to plot a dotted line with a defined slope
def plot_slope(m=1, a=-100, b=100):
    v = np.array([a, b])
    plt.plot(v, v * m, ':',
             c='k', alpha=0.5, zorder=0)


# plot display style
plt.style.use('seaborn-notebook')

# make a new figure
fignum = 0
plt.figure(fignum, (20, 7))

# in the first subplot, plot the PLSR scores for the signal and response
ax = plt.subplot(131)
plt.scatter(XS[:, 0], XS[:, 1], label='Signal')
plt.scatter(YS[:, 0], YS[:, 1], marker='*', s=10 ** 2,
            label='Response')

# then label each of the score groups with the condition they correspond to
for i, col in enumerate(conds):
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

# score the model by dotting each loading vector with the response vector in
# principal component space
prod = np.dot(XL, YL.T).squeeze()

# normalize this inner product by squaring it and dividing by the maximum
prod_norm = np.power(prod, 2) / max(np.power(prod, 2))

# select signals who's normalize product is greater than 0.90
selects = np.where(prod_norm > 0.90)[0]

# sort the selection in order of decreasing inner product magnitude
selects = selects[np.argsort(prod_norm[selects])[::-1]]

# print out the selected signals with their inner product coefficients
list_coeff(selects, prod)

# in the second subplot, plot the PLSR loadings for the signal and response
plt.subplot(132)

# color the loadings by their value for the inner product with the response
plt.scatter(XL[:, 0], XL[:, 1], c=-prod, cmap='coolwarm', linewidths=0.5,
            edgecolors='k')
plt.scatter(YL[:, 0], YL[:, 1], marker='*', s=10 ** 2, c='C1')
plot_xy_axes()
plt.xlabel("Principal Axis 1")
plt.ylabel("Principal Axis 2")

# in the third subplot, plot the PLSR loadings for the significantly
# associated signals ans annotate them with their protein and time-point
# identity.
ax = plt.subplot(133)
plt.scatter(XL[selects, 0], XL[selects, 1])
plt.scatter(YL[:, 0], YL[:, 1], marker='*', s=10 ** 2)
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

# calculate a prediction using the score and loading matrices
x_prediction = np.dot(XS, XL.T)
y_prediction = np.dot(YS, YL.T)

# make a new figure
fignum = fignum + 1
plt.figure(fignum, (17, 8))

# in the first subplot, plot the observed phosphorylation values against the
# predicted phosphorylation values for each signal
plt.subplot(121)
for x_real, x_pred, col_name in zip(Xt, x_prediction, conds):
    plt.scatter(x_real, x_pred, label=col_name, alpha=0.8)
plot_xy_axes()
plt.xlabel('Observed Value')
plt.ylabel('Predicted Value')
plt.legend(frameon=False)

# do the same for the second subplot, except plot the observed and predicted
# response values
plt.subplot(122)
prediction = np.dot(YS[:, 0:2], YL[:, 0:2].T)
for y_real, y_pred, col_name in zip(Yt, y_prediction, conds):
    plt.scatter(y_real, y_pred, label=col_name, alpha=0.8, s=10 ** 2)
plot_xy_axes()
plot_slope()
plt.xlabel('Observed Value')
plt.ylabel('Predicted Value')
plt.savefig('figures/plsr_predict.png', transparent=True)
plt.show()

# Part 3 - LASSO
# choose a list of lambdas with which to build a LASSO model
lambdas = [1, 1e-1, 1e-2, 1e-3]

# begin a new figure
fignum = fignum + 1
plt.figure(fignum, (6, 6))

# loop through each lambda
for lmda in lambdas:
    # build a LASSO model with the data and the given lambda
    lasso = Lasso(alpha=lmda, selection='random',
                  max_iter=1e5, normalize=True)
    lasso.fit(X.T, Y)

    # predict the response from the model
    Y_pred = lasso.predict(X.T)

    # plot the observed response against the predicted response
    plt.scatter(Y, Y_pred, label=('= %.3f' % lmda).rstrip('0').rstrip('.'),
                alpha=0.9, s=10 ** 2)

    # extract the coefficients assigned to each signal by the model
    B = lasso.coef_

    # select all nonzero coefficients
    absB = abs(B)
    nonzero_sel = np.where(absB > 1e-10)[0]

    # sorts the selection from highest coefficient magnitude to lowest then
    # print them out
    nonzero_sel = nonzero_sel[np.argsort(absB[nonzero_sel])[::-1]]
    print('\nLasso with lambda = %1.0E' % lmda)
    list_coeff(nonzero_sel, B)

# finish plot annotations
plot_xy_axes()
plot_slope()
plt.legend(frameon=False, title='Lambda')
plt.xlabel('Observed Value')
plt.ylabel('Predicted Value')
plt.savefig('figures/lasso.png', transparent=True)
plt.show()
