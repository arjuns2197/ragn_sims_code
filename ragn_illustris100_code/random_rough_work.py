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

a3 = 0.09
a2 = -0.81
a1 = 2.43
a0 = -2.75

path = "../ragn_illustris100_files/df_99.pkl"
df = pd.read_pickle(path)

#inds_high_mass = np.where((df['log_mass'] >= 10) & (df['log_mass'] <= 12))[0]
#df = df.iloc[inds_high_mass].reset_index(drop=True)

''' 
log_bhm = df['log_bhm']
log_bhmdot = df['log_bhmdot']
log_M_dot = np.log10(funcs.get_eddscaled_mdot(log_bhm, log_bhmdot))

log_bhm_grid = np.linspace(6,10,1000)
log_chi_default = np.log10(funcs.chi_of_mbh(log_bhm_grid, 0.002, 2, 8, 0.1))
log_chi_preferably = np.log10(funcs.chi_of_mbh(log_bhm_grid, 1e-6, 6, 9, 7e-5))


plt.scatter(log_bhm, log_M_dot, c = 'tab:gray', alpha = 0.1, label = 'All galaxies at z = 0')
plt.scatter(log_bhm[inds_high_mass], log_M_dot[inds_high_mass], c = 'k', label = 'Galaxies considered')
plt.plot(log_bhm_grid, log_chi_default, c = 'k', linewidth = 5, linestyle = '--', label = 'Default Transition')
plt.plot(log_bhm_grid, log_chi_preferably, c = 'k', linewidth = 5, linestyle = '-.', label = 'Proposed Transition')
plt.xlabel(r"$\log \left(M_{\rm BH} \right)$", fontsize = 25)
plt.ylabel(r"$\log \left(\dot{M} \right)$", fontsize = 25)
plt.xlabel(r"$\log \left(M_{\rm BH}/M_\odot \right)$", fontsize = 25)
plt.legend(fontsize = 25)
plt.show()
'''

'''
inds_high_mass = np.where((df['log_mass'] >= 10) & (df['log_mass'] <= 12))[0]
log_bhm = df['log_bhm']
log_bhmdot = df['log_bhmdot']
log_ssfr = df['log_ssfr']
log_M_dot = np.log10(funcs.get_eddscaled_mdot(log_bhm, log_bhmdot))

log_bhm_grid = np.linspace(6,10,1000)
log_chi_default = np.log10(funcs.chi_of_mbh(log_bhm_grid, 2,2, 8, 1))
log_chi_preferably = np.log10(funcs.chi_of_mbh(log_bhm_grid, 1e-4, 3, 8.5, 1e-2))

ax = plt.gca()
ax.set_title(r"$\mathrm{Increased ~BH ~vicinity ~gas~ density}; \rho^{\prime} = 100 \times \rho$", fontsize = 30)
#plt.scatter(log_bhm, 2+log_M_dot, c = 'tab:gray', alpha = 0.1, label = 'All galaxies at z = 0')
plt.scatter(log_bhm[inds_high_mass], 2+log_M_dot[inds_high_mass], c = log_ssfr[inds_high_mass], cmap = 'RdBu')
plt.plot(log_bhm_grid, -3*np.ones(len(log_bhm_grid)), c = 'k', label = r'$MaNGA~ \lambda_c$')
plt.plot(log_bhm_grid, log_chi_preferably, c = 'k', linewidth = 5, linestyle = '-.', label = r'$\mathrm{Proposed~ Transition~} \chi^{\prime}(M_{\rm BH})$')
plt.plot(log_bhm_grid, log_chi_default, c = 'k', linewidth = 5, linestyle = ':',
label = r'$\mathrm{Default~ TNG~Transition~} \chi(M_{\rm BH}); \mathrm{adjusted~ for~} \rho^{\prime}$')
plt.xlabel(r"$\log \left(M_{\rm BH} \right)$", fontsize = 25)
plt.ylabel(r"$\log \left(\dot{M} \right)$", fontsize = 25)
plt.xlabel(r"$\log \left(M_{\rm BH}/M_\odot \right)$", fontsize = 25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylim(-6,1)
plt.colorbar().set_label(r"$\log \left(sSFR/ yr^{-1} \right)$", fontsize= 20)
plt.legend(fontsize = 16, loc = 'lower left')
plt.show()


default_epsilonk = df['epsilon_k']

plt.scatter(log_M_dot,np.log10(df['ker']))
#plt.scatter(log_bhm, log_M_dot, c = 'tab:gray', alpha = 0.1, label = 'All galaxies at z = 0')
plt.ylabel(r"$\log \left(\dot{M} \right)$", fontsize = 25)
plt.xlabel(r"$\lambda = P_{\rm Kin}/ L_{\rm Edd} = \epsilon_{kin}\dot{M}/\epsilon_{\rm r}$", fontsize = 25)
plt.show()
'''



