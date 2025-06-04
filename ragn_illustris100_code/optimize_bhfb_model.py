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
f = 30
a3 = 0.09
a2 = -0.81
a1 = 2.43
a0 = -2.75
lambda_crit = 1e-3

log_chi0_start = -5
log_m0_start = 7.
log_beta_start = 0.1
log_chimax_start = -6
log_lamc_offset_start = -25

chi_parms_start = [log_chi0_start, log_m0_start, log_beta_start, log_chimax_start]

path = "../ragn_illustris100_files/df.pkl"
df = pd.read_pickle(path)

print(df.columns)


all_parms_opt = funcs.minimize_chi2error(df)

np.savez("../ragn_illustris100_files/results/opt_chi_parms.npz", all_parms_opt = all_parms_opt)



