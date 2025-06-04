import sys
sys.path.append("..")

import functions_illustris300 as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os


path_99 = "../ragn_illustris300_files/df_99_tng300.pkl"
df_99 = pd.read_pickle(path_99)

df_99 = df_99.drop(["log_L_kin", "ker"], axis = 1)

ssfr = 10**df_99['log_sfr']/10**df_99['log_mass']
log_ssfr = np.log10(ssfr)

inds_inf_rho = np.where(df_99['rho_gas_bh'] == -np.inf)
df_99['rho_gas_bh'][inds_inf_rho] = np.zeros(len(inds_inf_rho[0]))

rhos = df_99['rho_gas_bh']

log_L_kin = []
ker = []
is_strong_agn_minus4 = []
epsilon_k = []

for i in range(len(df_99)):

    mdot_bh_si = (10**df_99['log_bhmdot'][i])*(1.989e30/3.156e7)

    epsilon = f.get_epsilon_k(rhos[i])
    epsilon_k.append(epsilon)

    L = f.get_edot_k(mdot_bh_si,rhos[i])
    log_L = np.log10(L)
    log_L_kin.append(log_L)

    er = f.get_ker(10**log_L,10**df_99['log_L_Edd'][i])
    ker.append(er)

    if er > 1e-4:
        is_strong_agn_minus4.append(True)

    else:
        is_strong_agn_minus4.append(False)




df_99['log_ssfr'] = log_ssfr
df_99['epsilon_k'] = epsilon_k
df_99['log_L_kin'] = log_L_kin
df_99['ker'] = ker
df_99['is_strong_agn_minus4'] = is_strong_agn_minus4



df_99.to_pickle("../ragn_illustris300_files/df.pkl")
