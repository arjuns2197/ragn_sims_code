import sys
sys.path.append("..")

import yt
import caesar
import matplotlib.pyplot as plt
import numpy as np
import funcs_simba as funcs
import pandas as pd
from readgadget import readsnap
import h5py
from file_read_backwards import FileReadBackwards
from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import betaincinv
from scipy.stats import binom

a3 = 0.09
a2 = -0.81
a1 = 2.43
a0 = -2.75

df = pd.read_pickle("../ragn_simba_files/data/df.pkl")
f = 30

inds_high_mass = np.where((df['mass'] >= 10) & (df['mass'] <= 12))[0]
df = df.iloc[inds_high_mass].reset_index(drop=True)

def lamc_score_slopes(theta):

    log_lam_c = theta[0]
    lam_c = 10**log_lam_c

    fagn_pred = funcs.get_Fagn_of_lcrit(df,lam_c)
    log_masses = fagn_pred[0]
    log_fracs = np.log10(fagn_pred[1])

    log_fracs_manga = funcs.Fagn_manga(log_masses, a3, a2, a1, a0)

    def line_eq(x,m,c):

        y = m*x + c

        return y
    
    inds_low_fracs = np.where(log_fracs < -4)

    log_fracs = np.delete(log_fracs, inds_low_fracs)
    log_fracs_manga = np.delete(log_fracs_manga, inds_low_fracs)
    log_masses = np.delete(log_masses, inds_low_fracs)

    if len(log_fracs) >= 4:
        popt_manga, pcov = curve_fit(line_eq, log_masses, log_fracs_manga)
        popt_simu, pcov = curve_fit(line_eq, log_masses, log_fracs)

        slope_manga = [popt_manga[0]]
        slope_simu = [popt_simu[0]]

        print(slope_manga)
        print(slope_simu)

        mse = root_mean_squared_error(slope_manga, slope_simu)

        return mse

    else:
        return np.inf


def lamc_score_sf_q(theta):

    log_lam_c = theta[0]
    lam_c = 10**log_lam_c

    [mass_center,frac,ebars_low,ebars_high,frac_q,ebars_q_low,ebars_q_high,frac_sf,ebars_sf_low,ebars_sf_high] = funcs.get_Fagn_of_lcrit(df,lam_c)

    frac_q = frac_q + 1e-8
    frac_q = np.log10(frac_q)

    frac_sf = frac_sf + 1e-8
    frac_sf = np.log10(frac_sf)


    rmse_sf_q = root_mean_squared_error(frac_q,frac_sf)

    return rmse_sf_q


def lamc_score_offsets(theta):

    log_lam_c = theta[0]
    lam_c = 10**log_lam_c

    [mass_center,frac,ebars_low,ebars_high,frac_q,ebars_q_low,ebars_q_high,frac_sf,ebars_sf_low,ebars_sf_high] = funcs.get_Fagn_of_lcrit(df,lam_c)

    frac = frac + 1e-8
    frac = np.log10(frac)

    frac_manga = funcs.Fagn_manga(mass_center,a3, a2, a1, a0)

    chisq_offsets = root_mean_squared_error(frac,frac_manga)

    return chisq_offsets



























