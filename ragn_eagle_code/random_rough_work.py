import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import eagleSqlTools as esql
import pandas as pd
import functions_eagle as funcs
from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from scipy.optimize import curve_fit

a3 = 0.09
a2 = -0.81
a1 = 2.43
a0 = -2.75

"""
def get_Nagn_of_lcrit(df,lambda_crit):

    ER = df['ER']

    is_strong_ragn = []
    for i in range(len(df)):

        if df['ER'][i] > lambda_crit:

            if df['ER'][i] < 0.02:

                is_strong_ragn.append(True)
            
            else:
                is_strong_ragn.append(False)
            

        else:
            is_strong_ragn.append(False)


    df['is_strong_ragn'] = is_strong_ragn


    df_mass_bins = funcs.bin_galaxies("log_mass", df)
    grped_df_mass = df_mass_bins.groupby('log_mass_bin')


    ngals = []
    nagn = []
    mass_center = []


    nagn_q = []
    nagn_sf = []

    ngals_q = []
    ngals_sf = []

    for mass, group in grped_df_mass:

        string = str(mass)
        split = string.split(',')
        left = float(split[0][1:])
        right = float(split[1][:-1])
        center = (left + right)/2
        mass_center.append(center)

        n_agn_all = group['is_strong_ragn'].sum()
        nagn.append(n_agn_all)
        size_all = len(group)

        ngals.append(size_all)

        s_q = 0
        n_agn_q = 0

        s_sf = 0
        n_agn_sf = 0

        for i in range(len(group)):
            
            if group['log_ssfr'].iloc[i] < -11:

                s_q = s_q + 1

                if group['is_strong_ragn'].iloc[i] == True:
                    n_agn_q = n_agn_q + 1


            if group['log_ssfr'].iloc[i] > -11:

                s_sf = s_sf + 1

                if group['is_strong_ragn'].iloc[i] == True:
                    n_agn_sf = n_agn_sf + 1


        ngals_q.append(s_q)
        nagn_q.append(n_agn_q)

        ngals_sf.append(s_sf)
        nagn_sf.append(n_agn_sf)
        
    mass_center = np.array(mass_center)
    nagn = np.array(nagn)
    ngals = np.array(ngals)

    nagn_q = np.array(nagn_q)
    ngals_q = np.array(ngals_sf)

    nagn_sf = np.array(nagn_sf)
    ngals_sf = np.array(ngals_sf)


    nagn_manga = ngals*(10**funcs.Fagn_manga(mass_center, a3, a2, a1, a0))
    nagn_q_manga = ngals_q*(10**funcs.Fagn_manga(mass_center, a3, a2, a1, a0))
    nagn_sf_manga = ngals_sf*(10**funcs.Fagn_manga(mass_center, a3, a2, a1, a0))

    return [mass_center,nagn,nagn_manga,nagn_q,nagn_q_manga,nagn_sf,nagn_sf_manga]
"""

df = pd.read_pickle("../ragn_eagle_files/data/df.pkl")

inds_high_mass = np.where((df['log_mass'] >= 10) & (df['log_mass'] <= 12))[0]
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



print(lamc_score_sf_q([-2]))











