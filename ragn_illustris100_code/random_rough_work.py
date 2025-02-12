import sys
sys.path.append("..")

import functions_illustris as funcs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import curve_fit

a3 = 0.09
a2 = -0.81
a1 = 2.43
a0 = -2.75

path = "../ragn_illustris100_files/df_99.pkl"
df = pd.read_pickle(path)

#inds_high_mass = np.where((df['log_mass'] >= 10) & (df['log_mass'] <= 12))[0]
#df = df.iloc[inds_high_mass].reset_index(drop=True)

inds_high_mass = np.where((df['log_mass'] >= 10) & (df['log_mass'] <= 12))[0]

log_bhm = df['log_bhm']
log_bhmdot = df['log_bhmdot']
log_M_dot = np.log10(funcs.get_eddscaled_mdot(log_bhm, log_bhmdot))

log_bhm_grid = np.linspace(6,10,1000)
log_chi = np.log10(funcs.chi_of_mbh(log_bhm_grid, 0.002, 2))

plt.scatter(log_bhm, log_M_dot, c = 'tab:gray', alpha = 0.1, label = 'All galaxies at z = 0')
plt.scatter(log_bhm[inds_high_mass], log_M_dot[inds_high_mass], c = 'k', label = 'Galaxies consdiered')
plt.plot(log_bhm_grid, log_chi, c = 'k', linewidth = 5, linestyle = '--', label = 'Transition')
plt.xlabel(r"$\log \left(M_{\rm BH} \right)$", fontsize = 25)
plt.ylabel(r"$\log \left(\dot{M} \right)$", fontsize = 25)
plt.xlabel(r"$\log \left(M_{\rm BH}/M_\odot \right)$", fontsize = 25)
plt.legend(fontsize = 20)
plt.show()