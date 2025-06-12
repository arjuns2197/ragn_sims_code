import sys
sys.path.append("..")


import caesar
import matplotlib.pyplot as plt
import numpy as np
import funcs_simba as funcs
import pandas as pd
f = 45


df = pd.read_pickle("../ragn_simba_files/data/df.pkl")

inds_high_mass = np.where((df['mass'] >= 10) & (df['mass'] <= 12))[0]
df = df.iloc[inds_high_mass].reset_index(drop=True)

[mass_center,frac_2,ebars_low_2,ebars_high_2,frac_q_2,ebars_q_low_2,ebars_q_high_2,frac_sf_2,ebars_sf_low_2,ebars_sf_high_2] = funcs.get_Fagn_of_lcrit(df,1e-3)
red_hex = '#d62728'
blue_hex = '#1f77b4'

plt.title("SIMBA " + r"$F_{\rm AGN}$" + " Comparison", fontsize = f)
# Quiescent
plt.scatter(mass_center, frac_q_2, c=red_hex, s=400, alpha=0.5, label="SIMBA Quiescent")
plt.plot(mass_center[frac_q_2 > 1e-7], frac_q_2[frac_q_2 > 1e-7], linewidth=2.5, c=red_hex)
plt.errorbar(mass_center[frac_q_2 > 1e-7], frac_q_2[frac_q_2 > 1e-7],
             yerr=[ebars_q_low_2[frac_q_2 > 1e-7], ebars_q_high_2[frac_q_2 > 1e-7]],
             c=red_hex, capsize=9, capthick=1)

# Star-forming
plt.scatter(mass_center, frac_sf_2, c=blue_hex, s=400, alpha=0.5, label="SIMBA Star-forming")
plt.plot(mass_center[frac_sf_2 > 1e-7], frac_sf_2[frac_sf_2 > 1e-7], linewidth=2.5, c=blue_hex)
plt.errorbar(mass_center[frac_sf_2 > 1e-7], frac_sf_2[frac_sf_2 > 1e-7],
             yerr=[ebars_sf_low_2[frac_sf_2 > 1e-7], ebars_sf_high_2[frac_sf_2 > 1e-7]],
             c=blue_hex, capsize=9, capthick=1)

plt.xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
plt.ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
plt.xlim(9.8, 12.2)
plt.xticks([10.2,10.6, 11, 11.4, 11.8])
plt.yscale('log')
plt.minorticks_off()
plt.ylim(3e-6, 3)
plt.yticks([1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1], labels=[-5,-4, -3, -2, -1, 0])
plt.tick_params(axis='x', labelsize=0.75 * f, length=10, width=3)
plt.tick_params(axis='y', labelsize=0.75 * f, length=10, width=3)

# Call function to plot overlay
funcs.plot_Fagn_mbins_constraint()

plt.tight_layout()
plt.legend(loc = 'upper left', fontsize = 0.45*f)
plt.show()


