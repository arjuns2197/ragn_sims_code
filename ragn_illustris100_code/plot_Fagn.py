import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import functions_illustris as funcs
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
f = 30
red_hex = '#d62728'
blue_hex = '#1f77b4'

df = pd.read_pickle("../ragn_illustris100_files/df.pkl")

inds_high_mass = np.where((df['log_mass'] >= 10) & (df['log_mass'] <= 12))[0]
df = df.iloc[inds_high_mass].reset_index(drop=True)

[mass_center,frac_2,ebars_low_2,ebars_high_2,frac_q_2,ebars_q_low_2,ebars_q_high_2,frac_sf_2,ebars_sf_low_2,ebars_sf_high_2] = funcs.get_Fagn_of_lcrit(df,1e-2)
[mass_center,frac_3,ebars_low_3,ebars_high_3,frac_q_3,ebars_q_low_3,ebars_q_high_3,frac_sf_3,ebars_sf_low_3,ebars_sf_high_3] = funcs.get_Fagn_of_lcrit(df,1e-3)
[mass_center,frac_4,ebars_low_4,ebars_high_4,frac_q_4,ebars_q_low_4,ebars_q_high_4,frac_sf_4,ebars_sf_low_4,ebars_sf_high_4] = funcs.get_Fagn_of_lcrit(df,1e-4)


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
plt.suptitle("Comparison with TNG100", fontsize=f)

# First subplot (axs[0, 0])

#axs[0, 0].plot(mass_center[frac_2 > 1e-7], frac_2[frac_2 > 1e-7], linewidth = 4, linestyle = '--',c='tab:gray', label='TNG100 Star-forming and Quiescent', alpha = 0.75)
#axs[0, 0].errorbar(mass_center[frac_3 > 1e-7], frac_3[frac_3 > 1e-7], yerr=ebars_3[frac_3 > 1e-7], c='tab:gray', capsize=15, capthick=1)
axs[0, 0].scatter(mass_center, frac_q_2, c= red_hex, s=200, alpha=0.5)
axs[0, 0].plot(mass_center[frac_q_2 > 1e-7], frac_q_2[frac_q_2 > 1e-7], linewidth=1.5, c=red_hex, label= "TNG100 Quiescent")
axs[0, 0].errorbar(mass_center[frac_q_2 > 1e-7], frac_q_2[frac_q_2 > 1e-7], yerr=[ebars_q_low_2[frac_q_2 > 1e-7], ebars_q_high_2[frac_q_2 > 1e-7]], c= red_hex, capsize=9, capthick=1)
axs[0, 0].scatter(mass_center, frac_sf_2, c= blue_hex, s=200, alpha=0.5)
axs[0, 0].plot(mass_center[frac_sf_2 > 1e-7], frac_sf_2[frac_sf_2 > 1e-7], linewidth=1.5, c= blue_hex, label="TNG100 Star-forming")
axs[0, 0].errorbar(mass_center[frac_sf_2 > 1e-7], frac_sf_2[frac_sf_2 > 1e-7], yerr=[ebars_sf_low_2[frac_sf_2 > 1e-7], ebars_sf_high_2[frac_sf_2 > 1e-7]] , c = blue_hex, capsize=9, capthick=1)
axs[0, 0].set_xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize= f)
axs[0, 0].set_ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
axs[0, 0].set_xlim(9.8, 12.2)
axs[0, 0].set_xticks([10, 10.4, 10.8,11.2, 11.6, 12])
axs[0, 0].set_yscale('log')
axs[0, 0].minorticks_off()
axs[0, 0].set_ylim(3e-6, 30)
axs[0, 0].set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1])
axs[0, 0].set_yticklabels([-5 ,-4, -3, -2, -1, 0])
axs[0, 0].tick_params(axis='x', labelsize= 0.75*f, length=10, width=3)
axs[0, 0].tick_params(axis='y', labelsize= 0.75*f, length=10, width=3)
axs[0,0].annotate(r"$\Delta \log_{10}\left(\lambda_c \right) = 1$", (10,3),fontsize = f)

plt.sca(axs[0, 0])
funcs.plot_Fagn_mbins_constraint()


# Second subplot (axs[0, 1])

axs[0, 1].minorticks_off()
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 1].legend(handles, labels, loc='center', fontsize = 0.75*f, frameon = False)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)

# Third subplot (axs[1, 0])

#axs[1, 0].plot(mass_center[frac_3 > 1e-7], frac_3[frac_3 > 1e-7], linewidth = 4, linestyle = '--',c='tab:gray', label='TNG100 Star-forming and Quiescent', alpha = 0.75)
#axs[1, 0].errorbar(mass_center[frac_4 > 1e-7], frac_4[frac_4 > 1e-7], yerr=ebars_4[frac_4 > 1e-7], c='tab:gray', capsize=15, capthick=1)
axs[1, 0].scatter(mass_center, frac_q_3, c= red_hex, s=200, alpha=0.5)
axs[1, 0].plot(mass_center[frac_q_3 > 1e-7], frac_q_3[frac_q_3 > 1e-7], linewidth=1.5, c= red_hex)
axs[1, 0].errorbar(mass_center[frac_q_3 > 1e-7], frac_q_3[frac_q_3 > 1e-7], yerr=[ebars_q_low_3[frac_q_3 > 1e-7], ebars_q_high_3[frac_q_3 > 1e-7]], c= red_hex, capsize=9, capthick=1)
axs[1, 0].scatter(mass_center, frac_sf_3, c= blue_hex, s=200, alpha=0.5)
axs[1, 0].plot(mass_center[frac_sf_3 > 1e-7], frac_sf_3[frac_sf_3 > 1e-7], linewidth=1.5, c= blue_hex)
axs[1, 0].errorbar(mass_center[frac_sf_3 > 1e-7], frac_sf_3[frac_sf_3 > 1e-7], yerr=[ebars_sf_low_3[frac_sf_3 > 1e-7], ebars_sf_high_3[frac_sf_3 > 1e-7]], c= blue_hex, capsize=9, capthick=1)
axs[1, 0].set_xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
axs[1, 0].set_ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
axs[1, 0].set_xlim(9.8, 12.2)
axs[1, 0].set_xticks([10, 10.4, 10.8,11.2, 11.6, 12])
axs[1, 0].set_yscale('log')
axs[1, 0].minorticks_off()
axs[1, 0].set_ylim(3e-6, 30)
axs[1, 0].set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1])
axs[1, 0].set_yticklabels([-5, -4, -3, -2, -1,0])
axs[1, 0].tick_params(axis='x', labelsize= 0.75*f, length=10, width=3)
axs[1, 0].tick_params(axis='y', labelsize= 0.75*f, length=10, width=3)
axs[1,0].annotate(r"$\Delta \log_{10}\left(\lambda_c \right) = 0$", (10,3),fontsize = f)

plt.sca(axs[1, 0])
funcs.plot_Fagn_mbins_constraint()

# Fourth subplot (axs[1, 1])

#axs[1, 1].plot(mass_center[frac_4 > 1e-7], frac_4[frac_4 > 1e-7], linewidth = 4, linestyle = '--',c='tab:gray', label='TNG100 Star-forming and Quiescent', alpha = 0.75)
#axs[1, 1].errorbar(mass_center[frac_5 > 1e-7], frac_5[frac_5 > 1e-7], yerr=ebars_5[frac_5 > 1e-7], c='tab:gray', capsize=15, capthick=1)
axs[1, 1].scatter(mass_center, frac_q_4, c=red_hex, s=200, alpha=0.5)
axs[1, 1].plot(mass_center[frac_q_4 > 1e-7], frac_q_4[frac_q_4 > 1e-7], linewidth=1.5, c= red_hex)
axs[1, 1].errorbar(mass_center[frac_q_4 > 1e-7], frac_q_4[frac_q_4 > 1e-7], yerr=[ebars_q_low_4[frac_q_4 > 1e-7], ebars_q_high_4[frac_q_4 > 1e-7]], c= red_hex, capsize=9, capthick=1)
axs[1, 1].scatter(mass_center, frac_sf_4, c= blue_hex, s=200, alpha=0.5)
axs[1, 1].plot(mass_center[frac_sf_4 > 1e-7], frac_sf_4[frac_sf_4 > 1e-7], linewidth=1.5, c=blue_hex)
axs[1, 1].errorbar(mass_center[frac_sf_4 > 1e-7], frac_sf_4[frac_sf_4 > 1e-7], yerr=[ebars_sf_low_4[frac_sf_4 > 1e-7], ebars_sf_high_4[frac_sf_4 > 1e-7]], c=blue_hex, capsize=9, capthick=1)
axs[1, 1].set_xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
axs[1, 1].set_ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
axs[1, 1].set_xlim(9.8, 12.2)
axs[1, 1].set_xticks([10, 10.4, 10.8,11.2, 11.6, 12])
axs[1, 1].set_yscale('log')
axs[1, 1].minorticks_off()
axs[1, 1].set_ylim(3e-6, 30)
axs[1, 1].set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1])
axs[1, 1].set_yticklabels([-5, -4, -3, -2, -1,0])
axs[1, 1].yaxis.set_label_position("right")
axs[1, 1].yaxis.tick_right()
axs[1, 1].tick_params(axis='x', labelsize= 0.75*f, length=10, width=3)
axs[1, 1].tick_params(axis='y', labelsize= 0.75*f, length=10, width=3)
axs[1,1].annotate(r"$\Delta \log_{10}\left(\lambda_c \right) = -1$", (10,3),fontsize = f)

plt.sca(axs[1, 1])
funcs.plot_Fagn_mbins_constraint()

# Adjust layout and add legend
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()
