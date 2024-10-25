import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import functions_eagle as funcs
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
f = 30
#df = funcs.download_eagle_data()
#df.to_pickle("../ragn_eagle_files/data/df.pkl")

df = pd.read_pickle("../ragn_eagle_files/data/df.pkl")

[mass_center,frac_4,frac_q_4,ebars_q_4,frac_sf_4,ebars_sf_4] = funcs.get_Fagn_of_lcrit(df,1e-4)
[mass_center,frac_3,frac_q_3,ebars_q_3,frac_sf_3,ebars_sf_3] = funcs.get_Fagn_of_lcrit(df,1e-3)
[mass_center,frac_5,frac_q_5,ebars_q_5,frac_sf_5,ebars_sf_5] = funcs.get_Fagn_of_lcrit(df,1e-5)


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
plt.suptitle("Comparison with EAGLE Simulation", fontsize=f)

# First subplot (axs[0, 0])
funcs.plot_manga_constraint()
axs[0, 0].scatter(mass_center, frac_4, s=200, c='tab:gray', label='Both Q and SFG', alpha=0.5)
axs[0, 0].scatter(mass_center, frac_q_4, c='r', s=200, alpha=0.5)
axs[0, 0].plot(mass_center[frac_q_4 > 1e-7], frac_q_4[frac_q_4 > 1e-7], linewidth=1.5, c='r', label=r"$Q~:~ \log\left(sSFR/ yr^{-1}\right) < -11$")
axs[0, 0].errorbar(mass_center[frac_q_4 > 1e-7], frac_q_4[frac_q_4 > 1e-7], yerr=ebars_q_4[frac_q_4 > 1e-7], c='r', capsize=9, capthick=1)
axs[0, 0].scatter(mass_center, frac_sf_4, c='b', s=200, alpha=0.5)
axs[0, 0].plot(mass_center[frac_sf_4 > 1e-7], frac_sf_4[frac_sf_4 > 1e-7], linewidth=1.5, c='b', label=r"$SFG~:~ \log\left(sSFR/ yr^{-1}\right) > -11$")
axs[0, 0].errorbar(mass_center[frac_sf_4 > 1e-7], frac_sf_4[frac_sf_4 > 1e-7], yerr=ebars_sf_4[frac_sf_4 > 1e-7], c='b', capsize=9, capthick=1)
axs[0, 0].set_xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
axs[0, 0].set_ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
axs[0, 0].set_xlim(9.8, 12.2)
axs[0, 0].set_xticks([10, 10.5, 11, 11.5, 12])
axs[0, 0].set_yscale('log')
axs[0, 0].minorticks_off()
axs[0, 0].set_ylim(1e-5, 1)
axs[0, 0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
axs[0, 0].set_yticklabels([-4, -3, -2, -1])
axs[0, 0].tick_params(axis='x', labelsize=f, length=10, width=3)
axs[0, 0].tick_params(axis='y', labelsize=f, length=10, width=3)

# Second subplot (axs[0, 1])

axs[0, 1].minorticks_off()
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 1].legend(handles, labels, loc='center', fontsize = 0.75*f, frameon = False)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)

# Third subplot (axs[1, 0])
funcs.plot_manga_constraint()
axs[1, 0].scatter(mass_center, frac_3, s=200, c='tab:gray', alpha=0.5)
axs[1, 0].scatter(mass_center, frac_q_3, c='r', s=200, alpha=0.5)
axs[1, 0].plot(mass_center[frac_q_3 > 1e-7], frac_q_3[frac_q_3 > 1e-7], linewidth=1.5, c='r')
axs[1, 0].errorbar(mass_center[frac_q_3 > 1e-7], frac_q_3[frac_q_3 > 1e-7], yerr=ebars_q_3[frac_q_3 > 1e-7], c='r', capsize=9, capthick=1)
axs[1, 0].scatter(mass_center, frac_sf_3, c='b', s=200, alpha=0.5)
axs[1, 0].plot(mass_center[frac_sf_3 > 1e-7], frac_sf_3[frac_sf_3 > 1e-7], linewidth=1.5, c='b')
axs[1, 0].errorbar(mass_center[frac_sf_3 > 1e-7], frac_sf_3[frac_sf_3 > 1e-7], yerr=ebars_sf_3[frac_sf_3 > 1e-7], c='b', capsize=9, capthick=1)
axs[1, 0].set_xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
axs[1, 0].set_ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
axs[1, 0].set_xlim(9.8, 12.2)
axs[1, 0].set_xticks([10, 10.5, 11, 11.5, 12])
axs[1, 0].set_yscale('log')
axs[1, 0].minorticks_off()
axs[1, 0].set_ylim(1e-5, 1)
axs[1, 0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
axs[1, 0].set_yticklabels([-4, -3, -2, -1])
axs[1, 0].tick_params(axis='x', labelsize=f, length=10, width=3)
axs[1, 0].tick_params(axis='y', labelsize=f, length=10, width=3)

# Fourth subplot (axs[1, 1])
funcs.plot_manga_constraint()
axs[1, 1].scatter(mass_center, frac_5, s=200, c='tab:gray', alpha=0.5)
axs[1, 1].scatter(mass_center, frac_q_5, c='r', s=200, alpha=0.5)
axs[1, 1].plot(mass_center[frac_q_5 > 1e-7], frac_q_5[frac_q_5 > 1e-7], linewidth=1.5, c='r')
axs[1, 1].errorbar(mass_center[frac_q_5 > 1e-7], frac_q_5[frac_q_5 > 1e-7], yerr=ebars_q_5[frac_q_5 > 1e-7], c='r', capsize=9, capthick=1)
axs[1, 1].scatter(mass_center, frac_sf_5, c='b', s=200, alpha=0.5)
axs[1, 1].plot(mass_center[frac_sf_5 > 1e-7], frac_sf_5[frac_sf_5 > 1e-7], linewidth=1.5, c='b')
axs[1, 1].errorbar(mass_center[frac_sf_5 > 1e-7], frac_sf_5[frac_sf_5 > 1e-7], yerr=ebars_sf_5[frac_sf_5 > 1e-7], c='b', capsize=9, capthick=1)
axs[1, 1].set_xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
axs[1, 1].set_ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
axs[1, 1].set_xlim(9.8, 12.2)
axs[1, 1].set_xticks([10, 10.5, 11, 11.5, 12])
axs[1, 1].set_yscale('log')
axs[1, 1].minorticks_off()
axs[1, 1].set_ylim(1e-5, 1)
axs[1, 1].set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
axs[1, 1].set_yticklabels([-4, -3, -2, -1])
axs[1, 1].tick_params(axis='x', labelsize=f, length=10, width=3)
axs[1, 1].tick_params(axis='y', labelsize=f, length=10, width=3)

# Adjust layout and add legend
fig.subplots_adjust(wspace=0, hspace=0)


plt.show()
