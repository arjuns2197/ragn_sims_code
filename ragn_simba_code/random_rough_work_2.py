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

f = 30


df = pd.read_pickle("../ragn_simba_files/data/df.pkl")

inds_high_mass = np.where((df['mass'] >= 10) & (df['mass'] <= 12))[0]
df = df.iloc[inds_high_mass].reset_index(drop=True)

[mass_center,frac_2,ebars_low_2,ebars_high_2,frac_q_2,ebars_q_low_2,ebars_q_high_2,frac_sf_2,ebars_sf_low_2,ebars_sf_high_2] = funcs.get_Fagn_of_lcrit(df,1e-3)



# Quiescent
plt.scatter(mass_center, frac_q_2, c='tab:red', s=200, alpha=0.5)
plt.plot(mass_center[frac_q_2 > 1e-7], frac_q_2[frac_q_2 > 1e-7], linewidth=1.5, c='tab:red', label="SIMBA Quiescent")
plt.errorbar(mass_center[frac_q_2 > 1e-7], frac_q_2[frac_q_2 > 1e-7],
             yerr=[ebars_q_low_2[frac_q_2 > 1e-7], ebars_q_high_2[frac_q_2 > 1e-7]],
             c='tab:red', capsize=9, capthick=1)

# Star-forming
plt.scatter(mass_center, frac_sf_2, c='royalblue', s=200, alpha=0.5)
plt.plot(mass_center[frac_sf_2 > 1e-7], frac_sf_2[frac_sf_2 > 1e-7], linewidth=1.5, c='royalblue', label="SIMBA Star-forming")
plt.errorbar(mass_center[frac_sf_2 > 1e-7], frac_sf_2[frac_sf_2 > 1e-7],
             yerr=[ebars_sf_low_2[frac_sf_2 > 1e-7], ebars_sf_high_2[frac_sf_2 > 1e-7]],
             c='royalblue', capsize=9, capthick=1)

plt.xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
plt.ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
plt.xlim(10.4, 12.2)
plt.xticks([10.6, 11, 11.4, 12])
plt.yscale('log')
plt.minorticks_off()
plt.ylim(3e-6, 30)
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], labels=[-4, -3, -2, -1, 0])
plt.tick_params(axis='x', labelsize=0.75 * f, length=10, width=3)
plt.tick_params(axis='y', labelsize=0.75 * f, length=10, width=3)

# Call function to plot overlay
funcs.plot_Fagn_mbins_constraint()


plt.legend()
plt.show()
