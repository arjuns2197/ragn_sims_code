import sys
sys.path.append("..")


import caesar
import matplotlib.pyplot as plt
import numpy as np
import agn_functions as agn
import pandas as pd



df = pd.read_pickle("../ragn_simba_files/data/df.pkl")

masses = np.linspace(10,12,1000)
F_rmanga = agn.get_Fagn_rmanga(masses)
f = 30


df_mass_bins = agn.bin_galaxies("mass", df)
grped_df_mass = df_mass_bins.groupby('mass_bin')


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

    n_agn_all = group['is_strong_agn_fedd'].sum()
    size_all = len(group)
    f_all = n_agn_all/size_all
    size.append(size_all)
    frac.append(f_all)

    s_q = 0
    n_agn_q = 0

    s_sf = 0
    n_agn_sf = 0

    for i in range(len(group)):
        
        if group['sSFR'].iloc[i] < -11.5:

            s_q = s_q + 1

            if group['is_strong_agn_fedd'].iloc[i] == True:
                n_agn_q = n_agn_q + 1


        if group['sSFR'].iloc[i] > -11.5:

            s_sf = s_sf + 1

            if group['is_strong_agn_fedd'].iloc[i] == True:
                n_agn_sf = n_agn_sf + 1
    

    print(mass_center)

    size_q.append(s_q)
    frac_q.append(n_agn_q/s_q)
    ebars_q.append(2*np.sqrt(n_agn_q/(s_q)**2))


    size_sf.append(s_sf)
    frac_sf.append(n_agn_sf/s_sf)
    ebars_sf.append(2*np.sqrt(n_agn_sf/(s_sf)**2))
    

plt.title("Comparision with SIMBA Simulation", fontsize = f)
plt.scatter(mass_center, frac_q, c = 'r', s = 300)
plt.plot(mass_center, frac_q, linewidth = 1.5, c = 'r',label = r"$Q~:~ \log\left(sSFR/ yr^{-1}\right) < -11.5$")
plt.errorbar(mass_center, frac_q, yerr = ebars_q, c = 'r', capsize = 4)

plt.scatter(mass_center, frac_sf, c = 'b', s = 300)
plt.plot(mass_center, frac_sf, linewidth = 1.5, c = 'b',label = r"$SFG~:~ (\log\left(sSFR/ yr^{-1}\right) > -11.5)$")
plt.errorbar(mass_center, frac_sf, yerr = ebars_sf, c = 'b', capsize = 4)

# for n in range(len(size_q)):
#     plt.annotate(str(size_q[n]), (mass_center[n] + 0.0000008,frac_q[n]+0.0000008))
#     plt.annotate(str(size_sf[n]), (mass_center[n] + 0.0000008,frac_sf[n]+0.0000008))
#     plt.annotate(str(size[n]), (mass_center[n] + 0.0000008,frac[n]+0.0000008))


plt.plot(masses,10**F_rmanga, label = r'$F_{\rm AGN} ~\rm MaNGA$', linewidth = 4, c = 'k')
plt.scatter(mass_center, frac, s = 500, c = 'tab:gray', label = 'Both Q and SFG')
plt.xlabel(r"$\log\left(M_\star/ M_{\odot}\right)$", fontsize = 1.5*f)
plt.ylabel(r"$F_{\rm AGN}$", fontsize = 1.5*f)
plt.xlim(9.8,12.1)
plt.yscale('log')
plt.minorticks_off()
plt.legend(prop = {'size' : 20})
plt.yticks([1e-4,1e-3,1e-2,1e-1])
plt.tick_params(axis = 'x', labelsize = 1.5*f, length = 10, width = 3)
plt.tick_params(axis = 'y', labelsize = 1.5*f, length = 10, width = 3)
plt.tight_layout()
plt.show()