import sys
sys.path.append("..")

import funcs_simba as funcs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from scipy.stats import binom
from scipy.optimize import curve_fit

a0 = -2.73
a1 = 2.40
a2 = -0.79
a3 = 0.09

df = pd.read_pickle("../ragn_simba_files/data/df.pkl")
[mass_center,frac,ebars_low,ebars_high,frac_q,ebars_q_low,ebars_q_high,frac_sf,ebars_sf_low,ebars_sf_high] = funcs.get_Fagn_of_lcrit(df,1e-4)
frac_manga = funcs.Fagn_manga(mass_center,a3,a2,a1,a0)


print(funcs.get_rms_fagn(frac_manga, frac))
print(funcs.get_dev_slope(mass_center, frac_manga, frac))
print(funcs.get_rms_sfq(frac_q, frac_sf))
