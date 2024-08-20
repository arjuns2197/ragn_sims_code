import sys
sys.path.append("..")

import yt
import caesar
import matplotlib.pyplot as plt
import numpy as np
import agn_functions as agn
import pandas as pd

df = pd.read_pickle("../ragn_simba_files/data/df.pkl")
f = 30

inds_all = []

for i in range(len(df)):
    if df['Lr'][i] - 16.146 > 19.5:
        inds_all.append(i)


N_all, L_all = np.histogram(df['Lr'][inds_all] - 16.146, bins = 20)
bin_width = L_all[1] - L_all[0]


inds_agn = []

for i in range(len(df)):
    if df['Lagn'][i] - 16.146 > 19.5:
        inds_agn.append(i)

N_agn, L_agn = np.histogram(df['Lagn'][inds_agn] - 16.146, bins = 20)
bin_width_agn = L_agn[1] - L_agn[0]

inds_sf = []

for i in range(len(df)):
    if df['Lsfr'][i] - 16.146 > 19.5:
        inds_sf.append(i)

N_sf, L_sf = np.histogram(df['Lsfr'][inds_sf] - 16.146, bins = 20)
bin_width_sf = L_sf[1] - L_sf[0]


plt.plot(L_all[:-1],np.log10(N_all/((2.5*bin_width)*9*1e6)), label = 'Simba: All', linewidth = 4, linestyle = '--', c = 'green')
plt.plot(L_agn[:-1],np.log10(N_agn/((2.5*bin_width_agn)*9*1e6)), label = 'Simba: AGN', linewidth = 4, linestyle = '--', c = 'red')
plt.plot(L_sf[:-1],np.log10(N_sf/((2.5*bin_width_sf)*9*1e6)), label = 'Simba: SF', linewidth = 4, linestyle = '--', c = 'c')
plt.xlabel(r"$L_{1.4 \rm ~GHz} ~ \left(\rm W~Hz^{-1}\right)$", fontsize = f)
plt.ylabel(r"$\log \Phi\left(\rm mag^{-1} ~Mpc^{-3}\right)$", fontsize = f)
plt.xticks(fontsize = f)
plt.yticks(fontsize = f)
plt.legend(prop = {'size' : f})
plt.ylim(-7,-1)
plt.xlim(18.5,26.5)
plt.tight_layout()
plt.show()

inds_q = []

for i in range(len(df)):
    if df['Lr'][i] - 16.146 > 19.5:
        if df['sSFR'][i] < -11.5:
            inds_q.append(i)

N_q, L_q = np.histogram(df['Lr'][inds_q] - 16.146, bins = 20)
bin_width_q = L_q[1] - L_q[0]


inds_sfr = []

for i in range(len(df)):
    if df['Lr'][i] - 16.146 > 19.5:
        if df['sSFR'][i] > -11.5:
            inds_sfr.append(i)

N_sfr, L_sfr = np.histogram(df['Lr'][inds_sfr] - 16.146, bins = 20)
bin_width_sfr = L_sfr[1] - L_sfr[0]


plt.plot(L_all[:-1],np.log10(N_all/((2.5*bin_width)*9*1e6)), label = 'Simba: All', linewidth = 4, linestyle = '--', c = 'green')
plt.plot(L_q[:-1],np.log10(N_q/((2.5*bin_width_q)*9*1e6)), label = 'Simba: Q', linewidth = 4, linestyle = '--', c = 'magenta')
plt.plot(L_sfr[:-1],np.log10(N_sfr/((2.5*bin_width_sfr)*9*1e6)), label = 'Simba: SFG', linewidth = 4, linestyle = '--', c = 'blue')
plt.xlabel(r"$L_{1.4 \rm ~GHz} ~ \left(\rm W~Hz^{-1}\right)$", fontsize = f)
plt.ylabel(r"$\log \Phi\left(\rm mag^{-1} ~Mpc^{-3}\right)$", fontsize = f)
plt.xticks(fontsize = f)
plt.yticks(fontsize = f)
plt.legend(prop = {'size' : f})
plt.ylim(-7,-1)
plt.xlim(18.5,26.5)
plt.tight_layout()
plt.show()
