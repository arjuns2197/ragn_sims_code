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

path = "../ragn_illustris100_files/df.pkl"
df = pd.read_pickle(path)

all_parms_opt = np.load("../ragn_illustris100_files/results/opt_chi_parms.npz")['all_parms_opt']
chi_parms_opt = all_parms_opt[:-1]
log_lamc_offset_opt = all_parms_opt[-1]

##################################
log_chi0_opt, log_M0_opt, log_beta_opt, log_chi_max_opt = chi_parms_opt
log_bhm = df['log_bhm']
log_bhmdot = df['log_bhmdot']
log_ssfr = df['log_ssfr']
log_M_dot = np.log10(funcs.get_eddscaled_mdot(log_bhm, log_bhmdot))
M_dot = 10**log_M_dot

log_bhm_grid = np.linspace(6,10,1000)
log_chi_default = np.log10(funcs.chi_of_mbh(log_bhm_grid, -2.70, 8,0.30, -1))
log_chi_preferably = np.log10(funcs.chi_of_mbh(log_bhm_grid, log_chi0_opt, log_M0_opt, log_beta_opt, log_chi_max_opt))

chi = funcs.chi_of_mbh(df['log_bhm'], log_chi0_opt,log_M0_opt,log_beta_opt,log_chi_max_opt)

is_radio_mode = M_dot < chi
df['is_radio_mode'] = is_radio_mode

epsilon_kin = funcs.get_epsilon_k(df['rho_gas_bh'])
epsilon_r = 0.2
ker_of_chi = M_dot

df['ker_of_chi'] = ker_of_chi
df.loc[~df['is_radio_mode'], 'ker_of_chi'] = 0

condition_ker = np.log10(df['ker_of_chi']) > np.log10(lambda_crit) + log_lamc_offset_opt
df['is_strong_agn'] = np.zeros(len(df),dtype = bool)
df.loc[condition_ker,'is_strong_agn'] = np.ones(len(df['is_strong_agn'][condition_ker]), dtype = bool)

inds_agn = df['is_strong_agn'] 
inds_quenched = df['is_quenched']
inds_sf = ~df['is_quenched']


####################################################################################
#plot separately

# Create a standalone figure
fig1, ax1 = plt.subplots(figsize=(8, 8))

# Scatter plots
ax1.scatter(log_bhm[inds_quenched], np.log10(ker_of_chi[inds_quenched]), s=100, c='tab:red', alpha=0.2, zorder=2, label = 'Star Forming')
ax1.scatter(log_bhm[inds_sf], np.log10(ker_of_chi[inds_sf]), c='royalblue', s=100, alpha=0.2,marker = '*', zorder=1, label = 'Quiescent')

# Lines
ax1.plot(log_bhm_grid, (-3 + log_lamc_offset_opt) * np.ones(len(log_bhm_grid)), c='k', linewidth = 5, label = r"$ \log \left(\lambda_{\rm c}^{\prime} \right)= -3.55$")
ax1.plot(log_bhm_grid, (-3) * np.ones(len(log_bhm_grid)), c='tab:gray', linewidth = 1.5, label = r"$ \log \left(\lambda_{\rm c} \right)= -3.00$")
ax1.plot(log_bhm_grid, log_chi_preferably, c='k', linewidth= 7.5, linestyle=(0, (1, 1)), label=r'$\mathrm{Modified~ Transition~} \chi^{\prime}(M_{\rm BH})$')
ax1.plot(log_bhm_grid, log_chi_default, c='tab:gray', linewidth= 7.5, linestyle=(0, (1, 1)), label=r'$\mathrm{Default~ TNG~Transition~} \chi(M_{\rm BH})$')

# Labels and legend
ax1.set_xlabel(r"$\log_{\rm 10} \left(M_{\rm BH}/M_\odot \right)$", fontsize= f)
ax1.set_ylabel(r"$\log_{\rm 10} \left(\lambda \right)$", fontsize= f)
ax1.tick_params(axis='both', which='major', labelsize=f)
ax1.set_ylim(-10, 0)

# Legend inside the plot (adjust location as needed)
ax1.legend(fontsize= 0.5*f, loc='lower right', frameon=True)

plt.tight_layout()
plt.show()


# Create a standalone figure
fig2, ax2 = plt.subplots(figsize=(8, 8))
#plt.title("Modified TNG100 Model", fontsize = f)

# Compute values
mass_center,frac,ebars_low,ebars_high,frac_q,ebars_q_low,ebars_q_high,frac_sf,ebars_sf_low, ebars_sf_high,size = funcs.get_Fagn_of_chi_to_plot(df, lambda_crit, chi_parms_opt, log_lamc_offset_opt)


ax2.plot(mass_center[frac > 1e-7], frac[frac > 1e-7], linewidth = 6, linestyle = '--',c='tab:gray', label='Star-forming and Quiescent', alpha = 0.75)
#ax2.errorbar(mass_center[frac_4 > 1e-7], frac_4[frac_4 > 1e-7], yerr=ebars_4[frac_4 > 1e-7], c='tab:gray', capsize=15, capthick=1)
ax2.scatter(mass_center, frac_q, c='tab:red', s=150, alpha=0.5)
ax2.plot(mass_center[frac_q > 1e-7], frac_q[frac_q > 1e-7], linewidth=1.5, c='tab:red', label= "Quiescent")
ax2.errorbar(mass_center[frac_q > 1e-7], frac_q[frac_q > 1e-7], yerr=[ebars_q_low[frac_q > 1e-7], ebars_q_high[frac_q > 1e-7]], c='tab:red', capsize=9, capthick=1)
ax2.scatter(mass_center, frac_sf, c='royalblue', s=150, alpha=0.5)
ax2.plot(mass_center[frac_sf > 1e-7], frac_sf[frac_sf > 1e-7], linewidth=1.5, c='royalblue', label="Star-forming")
ax2.errorbar(mass_center[frac_sf > 1e-7], frac_sf[frac_sf > 1e-7], yerr=[ebars_sf_low[frac_sf > 1e-7], ebars_sf_high[frac_sf > 1e-7]], c='royalblue', capsize=9, capthick=1)
ax2.set_xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
ax2.set_ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
ax2.set_xlim(9.8, 12.2)
ax2.set_xticks([10, 10.5, 11, 11.5, 12])
ax2.set_yscale('log')
ax2.minorticks_off()
ax2.set_ylim(1e-5, 1)
ax2.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1])
ax2.set_yticklabels([-5, -4, -3, -2, -1,0])
ax2.tick_params(axis='x', labelsize= 0.75*f, length=10, width=3)
ax2.tick_params(axis='y', labelsize= 0.75*f, length=10, width=3)
funcs.plot_manga_constraint()
plt.legend(loc='lower right', fontsize = 0.7*f)


plt.tight_layout()
plt.show()