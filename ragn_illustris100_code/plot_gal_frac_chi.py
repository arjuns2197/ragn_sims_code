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
from scipy.optimize import curve_fit
from scipy.optimize import minimize
f= 30

a3 = 0.09
a2 = -0.81
a1 = 2.43
a0 = -2.75
lambda_crit = 1e-3

path = "../ragn_illustris100_files/df.pkl"
df = pd.read_pickle(path)

all_parms_opt = np.load("../ragn_illustris100_files/results/opt_chi_parms.npz")['all_parms_opt']
chi_parms_opt = all_parms_opt[:-1]
log_lamc_offset_opt = all_parms_opt[-1]

inds_above_default = funcs.is_above_default_tngline(df['log_bhm'], df[''])


plt.scatter(df['log_bhm'], np.log10(df['ker']))
plt.show()

