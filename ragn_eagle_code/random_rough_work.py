import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import eagleSqlTools as esql
import pandas as pd
import functions_eagle as funcs


plt.scatter(mass_center, frac_q_5, c = 'r', s = 200, alpha = 0.5)
plt.plot(mass_center[frac_q_5 > 1e-7], frac_q_5[frac_q_5 > 1e-7], linewidth = 1.5, c = 'r',label = r"$Q~:~ \log\left(sSFR/ yr^{-1}\right) < -11$")
plt.errorbar(mass_center[frac_q_5 > 1e-7], frac_q_5[frac_q_5 > 1e-7], yerr = ebars_q_5[frac_q_5 > 1e-7], c = 'r', capsize = 9, capthick=1)


plt.scatter(mass_center, frac_sf_5, c = 'b', s = 200, alpha = 0.5)
plt.plot(mass_center[frac_sf_5 > 1e-7], frac_sf_5[frac_sf_5 > 1e-7], linewidth = 1.5, c = 'b',label = r"$SFG~:~ \log\left(sSFR/ yr^{-1}\right) > -11$")
plt.errorbar(mass_center[frac_sf_5 > 1e-7], frac_sf_5[frac_sf_5 > 1e-7], yerr = ebars_sf_5[frac_sf_5 > 1e-7], c = 'b', capsize = 9, capthick=1)