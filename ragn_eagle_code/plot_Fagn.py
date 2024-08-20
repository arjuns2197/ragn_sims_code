import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import functions_eagle as funcs
import pandas as pd


df = pd.read_pickle("../ragn_eagle_files/df.pkl")

ER = funcs.get_ER(np.log10(df['bh_mdot']),df['log_bh_mass'])


df['ER'] = ER



is_strong_ragn = []
for i in range(len(df)):

    if df['ER'][i] > 0.001:

        if df['ER'][i] < 0.01:

            is_strong_ragn.append(True)
        
        else:
            is_strong_ragn.append(False)
        

    else:
        is_strong_ragn.append(False)


df['is_strong_ragn'] = is_strong_ragn

masses = np.linspace(10,12,1000)
F_rmanga = funcs.get_Fagn_rmanga(masses)
f = 30


df_mass_bins = funcs.bin_galaxies("log_mass", df)
grped_df_mass = df_mass_bins.groupby('log_mass_bin')


size = []
frac = []
mass_center = []


frac_q = []
frac_sf = []
ebars_q = []
ebars_sf = []
size_q = []
size_sf = []

for mass, group in grped_df_mass:

    string = str(mass)
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    mass_center.append(center)

    n_agn_all = group['is_strong_ragn'].sum()
    size_all = len(group)
    f_all = n_agn_all/size_all
    size.append(size_all)
    frac.append(f_all)

    s_q = 0
    n_agn_q = 0

    s_sf = 0
    n_agn_sf = 0

    for i in range(len(group)):
        
        if group['log_ssfr'].iloc[i] < -11.5:

            s_q = s_q + 1

            if group['is_strong_ragn'].iloc[i] == True:
                n_agn_q = n_agn_q + 1


        if group['log_ssfr'].iloc[i] > -11.5:

            s_sf = s_sf + 1

            if group['is_strong_ragn'].iloc[i] == True:
                n_agn_sf = n_agn_sf + 1
    

    print(mass_center)

    size_q.append(s_q)
    frac_q.append(n_agn_q/s_q)
    ebars_q.append(2*np.sqrt(n_agn_q/(s_q)**2))


    size_sf.append(s_sf)
    frac_sf.append(n_agn_sf/s_sf)
    ebars_sf.append(2*np.sqrt(n_agn_sf/(s_sf)**2))
    

plt.scatter(mass_center, frac_q, c = 'r')
plt.plot(mass_center, frac_q, linewidth = 3, c = 'r',label = r"$Q~ (\log\left(sSFR\right) < -11.5)$")
plt.errorbar(mass_center, frac_q, yerr = ebars_q, c = 'r')

plt.scatter(mass_center, frac_sf, c = 'b')
plt.plot(mass_center, frac_sf, linewidth = 3, c = 'b',label = r"$SFG~ (\log\left(sSFR\right) > -11.5)$")
plt.errorbar(mass_center, frac_sf, yerr = ebars_sf, c = 'b')


plt.plot(masses,10**F_rmanga, label = r'$F_{\rm AGN} ~\rm MaNGA$', linewidth = 4, c = 'k')
plt.scatter(mass_center, frac, s = 75, c = 'tab:gray', label = 'Both Q and SFG')
plt.xlabel("log(M)", fontsize = f)
plt.ylabel(r"$F_{\rm AGN}$", fontsize = f)
plt.xlim(9.8,12.1)
plt.xticks(fontsize = 0.5*f)
plt.yticks(fontsize = 0.5*f)
plt.yscale('log')
plt.legend(prop = {'size' : 15})
plt.show()