import sys
sys.path.append("..")

import functions_illustris300 as funcs
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


df = pd.read_pickle("../ragn_illustris300_files/df_99_tng300.pkl")

log_sfr = df['log_sfr']
log_mass = df['log_mass']
ker = df['ker']
log_bhm = df['log_bhm']
crit_er = df['kin_mode_crit_er']

log_L_kin = df['log_L_kin'][(df['log_L_kin'] > -5) & (df['log_L_kin'] < 45)]


plt.hist(log_L_kin)
plt.show()