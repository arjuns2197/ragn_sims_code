import sys
sys.path.append("..")

import yt
import caesar
import matplotlib.pyplot as plt
import numpy as np
import agn_functions as agn
import pandas as pd
from readgadget import readsnap
import h5py

df = agn.load_simba_df()
df.to_pickle("../ragn_simba_files/data/df.pkl")
df = pd.read_pickle("../ragn_simba_files/data/df.pkl")


inds_q = []
inds_sf = []

for i in range(len(df)):

    if df['sSFR'][i] > -11.5:
        inds_sf.append(i)
    
    elif df['sSFR'][i] < -11.5:
        inds_q.append(i)

plt.scatter(df['mass'][inds_q], np.log10(df['bhmdot'][inds_q]), c = 'r', alpha = 0.2)
plt.scatter(df['mass'][inds_sf], np.log10(df['bhmdot'][inds_sf]), c = 'b', alpha = 0.2)
plt.show()


plt.scatter(df['bhm'][inds_q], np.log10(df['bhmdot'][inds_q]), c = 'r', alpha = 0.2)
plt.scatter(df['bhm'][inds_sf], np.log10(df['bhmdot'][inds_sf]), c = 'b', alpha = 0.2)
plt.show()

p = (17/12) + 1 ## Dummy Lagn model; Lagn = (M_dot_bhm)^p

Lagn_dummy1 = []
is_agn_dummy1 = []
ER_dummy1 = []
is_strong_agn_dummy1 = []

for i in range(len(df)):

    L_agn_dummy = agn.get_agn_feedback_dummy(df['bhmdot'][i], df['fedd'][i], df['bhm'][i],df['mass'][i],p)
    Lagn_dummy1.append(L_agn_dummy)
    is_agn_dummy1.append(agn.check_is_agn(L_agn_dummy))
    E_ratio_dummy = agn.get_ER(df['bhm'][i],L_agn_dummy, agn.check_is_agn(L_agn_dummy))
    ER_dummy1.append(E_ratio_dummy)
    is_strong_agn_dummy1.append(agn.check_is_sagn(df['sfr'][i], L_agn_dummy, E_ratio_dummy)) 


df = df.drop(["Lagn_dummy","is_agn_dummy","is_strong_agn_dummy","ER_dummy"],axis = 1)


df['Lagn_dummy'] = Lagn_dummy1
df['is_agn_dummy'] = is_agn_dummy1
df['is_strong_agn_dummy'] = is_strong_agn_dummy1
df["ER_dummy"] = ER_dummy1

[frac, prop, size] = agn.get_strong_agn_frac("mass", df)
[frac_dummy1, prop_dummy1, size_dummy1] = agn.get_strong_agn_frac_dummy("mass", df)


p = (17/12) - 0.1 ## Dummy Lagn model; Lagn = (M_dot_bhm)^p

Lagn_dummy2 = []
is_agn_dummy2 = []
ER_dummy2 = []
is_strong_agn_dummy2 = []

for i in range(len(df)):

    L_agn_dummy = agn.get_agn_feedback_dummy(df['bhmdot'][i], df['fedd'][i], df['bhm'][i],df['mass'][i],p)
    Lagn_dummy2.append(L_agn_dummy)
    is_agn_dummy2.append(agn.check_is_agn(L_agn_dummy))
    E_ratio_dummy = agn.get_ER(df['bhm'][i],L_agn_dummy, agn.check_is_agn(L_agn_dummy))
    ER_dummy2.append(E_ratio_dummy)
    is_strong_agn_dummy2.append(agn.check_is_sagn(df['sfr'][i], L_agn_dummy, E_ratio_dummy)) 


df = df.drop(["Lagn_dummy","is_agn_dummy","is_strong_agn_dummy","ER_dummy"],axis = 1)


df['Lagn_dummy'] = Lagn_dummy2
df['is_agn_dummy'] = is_agn_dummy2
df['is_strong_agn_dummy'] = is_strong_agn_dummy2
df["ER_dummy"] = ER_dummy2

[frac, prop, size] = agn.get_strong_agn_frac("mass", df)
[frac_dummy2, prop_dummy2, size_dummy2] = agn.get_strong_agn_frac_dummy("mass", df)



masses = np.linspace(10,12,1000)
F_rmanga = agn.get_Fagn_rmanga(masses)
f = 30

nbins = 8

plt.scatter(prop, np.log10(frac), s = 80, label = r'$L_{\rm AGN} = (\dot{M_{\rm BH}})^{p}~;~ p = 17/12$')
plt.scatter(prop_dummy1, np.log10(frac_dummy1), s = 45, c = 'tab:pink', label = r"$p^{\prime} = p + 1$")
plt.plot(prop_dummy1[5:-2], np.log10(frac_dummy1[5:-2]), c = 'tab:pink', linewidth = 0.5)
plt.scatter(prop_dummy2, np.log10(frac_dummy2), s = 45, c = 'c', label = r"$p^{\prime} = p - 0.1$")
plt.plot(prop_dummy2[5:-2], np.log10(frac_dummy2[5:-2]), c = 'c', linewidth = 0.5)

plt.xlabel("log(M)", fontsize = f)
plt.ylabel(r"$\log\left(F_{\rm AGN}\right)$", fontsize = f)
plt.plot(masses,F_rmanga, label = r'$F_{\rm AGN} ~\rm MaNGA$', linewidth = 4, c = 'k')
plt.xlim(9.8,12.1)

for n in range(nbins):
    plt.annotate(str(size[n]), (prop[n] + 0.05,np.log10(frac[n])+0.05))

plt.legend(prop = {'size' : 20})
plt.show()

