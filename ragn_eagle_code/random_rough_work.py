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
f = 30

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


# Log F_AGN values (from board)
log_Fracs_q = np.array([-0.78, -1.88, -1.66, -1.17, -0.84])
log_Fracs_sf = np.array([-0.3, -1.62, -1.58, -1.10, -0.65])

# Errors only for last 3 Q bins and last 4 SF bins
log_ebars_q = np.array([[0.10, 0.05, 0.05],    # lower
                        [0.09, 0.05, 0.05]])   # upper
log_ebars_sf = np.array([[0.4, 0.22, 0.13, 0.22],
                            [0.37, 0.19, 0.12, 0.17]])



