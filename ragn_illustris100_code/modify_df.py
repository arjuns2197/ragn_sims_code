import sys
sys.path.append("..")

import functions_illustris as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os


path = "../ragn_illustris100_files/df_snap99.pkl"
df = pd.read_pickle(path)

df = df.drop(["log_L_kin", "ker", "is_strong_agn_minus6", "is_strong_agn_minus5"], axis = 1)

ssfr = 10**df['log_sfr']/10**df['log_mass']
log_ssfr = np.log10(ssfr)


inds_inf_rho = np.where(df['rho_gas_bh'] == -np.inf)
df['rho_gas_bh'][inds_inf_rho] = np.zeros(len(inds_inf_rho[0]))

rhos = df['rho_gas_bh']


log_L_kin = []
ker = []
is_strong_agn_minus4 = []
epsilon_k = []

for i in range(len(df)):

    mdot_bh_si = (10**df['log_bhmdot'][i])*(1.989e30/3.156e7)

    epsilon = f.get_epsilon_k(rhos[i])
    epsilon_k.append(epsilon)

    L = f.get_edot_k(mdot_bh_si,rhos[i])
    log_L = np.log10(L)
    log_L_kin.append(log_L)

    er = f.get_ker(10**log_L,10**df['log_L_Edd'][i])
    ker.append(er)

    if er > 1e-4:
        is_strong_agn_minus4.append(True)

    else:
        is_strong_agn_minus4.append(False)

    print(i)


df['log_ssfr'] = log_ssfr
df['epsilon_k'] = epsilon_k
df['log_L_kin'] = log_L_kin
df['ker'] = ker
df['is_strong_agn_minus4'] = is_strong_agn_minus4


plt.scatter(df['log_mass'], np.log10(df['ker']))
plt.show()

df.to_pickle("../ragn_illustris100_files/df_99.pkl")