import sys
sys.path.append("..")

import functions_illustris as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os


path_99 = "../ragn_illustris100_files/df_99.pkl"
df_99 = pd.read_pickle(path_99)

log_L_kin = df_99['log_L_kin'][df_99['log_L_kin'] > -5]

plt.hist(log_L_kin)
plt.show()
############################################################# snapshots 98 and 97
path_9897 = "../ragn_illustris100_files/df_9897.pkl"
df_9897 = pd.read_pickle(path_9897)

df_9897 = df_9897.drop(["log_L_kin", "ker"], axis = 1)

ssfr = 10**df_9897['log_sfr']/10**df_9897['log_mass']
log_ssfr = np.log10(ssfr)

inds_inf_rho = np.where(df_9897['rho_gas_bh'] == -np.inf)
df_9897['rho_gas_bh'][inds_inf_rho] = np.zeros(len(inds_inf_rho[0]))

rhos = df_9897['rho_gas_bh']

log_L_kin = []
ker = []
is_strong_agn_minus4 = []
epsilon_k = []

for i in range(len(df_9897)):

    mdot_bh_si = (10**df_9897['log_bhmdot'][i])*(1.989e30/3.156e7)

    epsilon = f.get_epsilon_k(rhos[i])
    epsilon_k.append(epsilon)

    L = f.get_edot_k(mdot_bh_si,rhos[i])
    log_L = np.log10(L)
    log_L_kin.append(log_L)

    er = f.get_ker(10**log_L,10**df_9897['log_L_Edd'][i])
    ker.append(er)

    if er > 1e-4:
        is_strong_agn_minus4.append(True)

    else:
        is_strong_agn_minus4.append(False)




df_9897['log_ssfr'] = log_ssfr
df_9897['epsilon_k'] = epsilon_k
df_9897['log_L_kin'] = log_L_kin
df_9897['ker'] = ker
df_9897['is_strong_agn_minus4'] = is_strong_agn_minus4

log_L_kin = df_9897['log_L_kin'][df_9897['log_L_kin'] > -5]

plt.hist(log_L_kin)
plt.show()
####################################################################### snapshots 96 through 92


path_9695949392 = "../ragn_illustris100_files/df_9695949392.pkl"
df_9695949392 = pd.read_pickle(path_9695949392)

df_9695949392 = df_9695949392.drop(["log_L_kin", "ker"], axis = 1)

ssfr = 10**df_9695949392['log_sfr']/10**df_9695949392['log_mass']
log_ssfr = np.log10(ssfr)

inds_inf_rho = np.where(df_9695949392['rho_gas_bh'] == -np.inf)
df_9695949392['rho_gas_bh'][inds_inf_rho] = np.zeros(len(inds_inf_rho[0]))

rhos = df_9695949392['rho_gas_bh']

log_L_kin = []
ker = []
is_strong_agn_minus4 = []
epsilon_k = []

for i in range(len(df_9695949392)):

    mdot_bh_si = (10**df_9695949392['log_bhmdot'][i])*(1.989e30/3.156e7)

    epsilon = f.get_epsilon_k(rhos[i])
    epsilon_k.append(epsilon)

    L = f.get_edot_k(mdot_bh_si,rhos[i])
    log_L = np.log10(L)
    log_L_kin.append(log_L)

    er = f.get_ker(10**log_L,10**df_9695949392['log_L_Edd'][i])
    ker.append(er)

    if er > 1e-4:
        is_strong_agn_minus4.append(True)

    else:
        is_strong_agn_minus4.append(False)




df_9695949392['log_ssfr'] = log_ssfr
df_9695949392['epsilon_k'] = epsilon_k
df_9695949392['log_L_kin'] = log_L_kin
df_9695949392['ker'] = ker
df_9695949392['is_strong_agn_minus4'] = is_strong_agn_minus4

log_L_kin = df_9695949392['log_L_kin'][df_9695949392['log_L_kin'] > -5]

plt.hist(log_L_kin)
plt.show()
############################################################################## snapshots 91 and 90

path_9190 = "../ragn_illustris100_files/df_9190.pkl"
df_9190 = pd.read_pickle(path_9190)

df_9190 = df_9190.drop(["log_L_kin", "ker"], axis = 1)

ssfr = 10**df_9190['log_sfr']/10**df_9190['log_mass']
log_ssfr = np.log10(ssfr)

inds_inf_rho = np.where(df_9190['rho_gas_bh'] == -np.inf)
df_9190['rho_gas_bh'][inds_inf_rho] = np.zeros(len(inds_inf_rho[0]))

rhos = df_9190['rho_gas_bh']

log_L_kin = []
ker = []
is_strong_agn_minus4 = []
epsilon_k = []

for i in range(len(df_9190)):

    mdot_bh_si = (10**df_9190['log_bhmdot'][i])*(1.989e30/3.156e7)

    epsilon = f.get_epsilon_k(rhos[i])
    epsilon_k.append(epsilon)

    L = f.get_edot_k(mdot_bh_si,rhos[i])
    log_L = np.log10(L)
    log_L_kin.append(log_L)

    er = f.get_ker(10**log_L,10**df_9190['log_L_Edd'][i])
    ker.append(er)

    if er > 1e-4:
        is_strong_agn_minus4.append(True)

    else:
        is_strong_agn_minus4.append(False)




df_9190['log_ssfr'] = log_ssfr
df_9190['epsilon_k'] = epsilon_k
df_9190['log_L_kin'] = log_L_kin
df_9190['ker'] = ker
df_9190['is_strong_agn_minus4'] = is_strong_agn_minus4


log_L_kin = df_9190['log_L_kin'][df_9190['log_L_kin'] > -5]

plt.hist(log_L_kin)
plt.show()
##################################################################### snapshots 89, 88 and 87
path_898887 = "../ragn_illustris100_files/df_898887.pkl"
df_898887 = pd.read_pickle(path_898887)

df_898887 = df_898887.drop(["log_L_kin", "ker"], axis = 1)

ssfr = 10**df_898887['log_sfr']/10**df_898887['log_mass']
log_ssfr = np.log10(ssfr)

inds_inf_rho = np.where(df_898887['rho_gas_bh'] == -np.inf)
df_898887['rho_gas_bh'][inds_inf_rho] = np.zeros(len(inds_inf_rho[0]))

rhos = df_898887['rho_gas_bh']

log_L_kin = []
ker = []
is_strong_agn_minus4 = []
epsilon_k = []

for i in range(len(df_898887)):

    mdot_bh_si = (10**df_898887['log_bhmdot'][i])*(1.989e30/3.156e7)

    epsilon = f.get_epsilon_k(rhos[i])
    epsilon_k.append(epsilon)

    L = f.get_edot_k(mdot_bh_si,rhos[i])
    log_L = np.log10(L)
    log_L_kin.append(log_L)

    er = f.get_ker(10**log_L,10**df_898887['log_L_Edd'][i])
    ker.append(er)

    if er > 1e-4:
        is_strong_agn_minus4.append(True)

    else:
        is_strong_agn_minus4.append(False)




df_898887['log_ssfr'] = log_ssfr
df_898887['epsilon_k'] = epsilon_k
df_898887['log_L_kin'] = log_L_kin
df_898887['ker'] = ker
df_898887['is_strong_agn_minus4'] = is_strong_agn_minus4


log_L_kin = df_898887['log_L_kin'][df_898887['log_L_kin'] > -5]

plt.hist(log_L_kin)
plt.show()
##################################################################### compile all dfs

compiled_df = pd.concat([df_99, df_9897, df_9695949392, df_9190, df_898887], ignore_index=True)

print(len(df_99)+ len(df_9897) + len(df_9695949392) + len(df_9190) + len(df_898887))
print(len(compiled_df))

is_quenched = compiled_df['log_ssfr'] < -11.5
compiled_df['is_quenched'] = is_quenched


compiled_df.to_pickle("../ragn_illustris100_files/df.pkl")
