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

log_bhm = df['log_bhm']
log_bhmdot = df['log_bhmdot']

log_M_dot = np.log10(funcs.get_eddscaled_mdot(log_bhm, log_bhmdot))
M_dot = 10**log_M_dot

inds_above_default = np.where(funcs.is_above_default_tngline(log_bhm, M_dot) == True)[0]
inds_below_default = np.where(funcs.is_below_default_tngline(log_bhm, M_dot) == True)[0]
inds_below_modified = np.where(funcs.is_below_modified_tngline(log_bhm, M_dot, all_parms_opt) == True)[0]
inds_above_modified = np.where(funcs.is_above_modified_tngline(log_bhm, M_dot, all_parms_opt) == True)[0]
inds_in_between = np.where(funcs.is_between_def_mod(log_bhm, M_dot, all_parms_opt) == True)[0]


N = len(df)

print(N)



