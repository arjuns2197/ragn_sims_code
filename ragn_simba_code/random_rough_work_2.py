import sys
sys.path.append("..")

import yt
import caesar
import matplotlib.pyplot as plt
import numpy as np
import funcs_simba as funcs
import pandas as pd
from readgadget import readsnap
import h5py
from file_read_backwards import FileReadBackwards
from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import betaincinv
from scipy.stats import binom

f = 30

plt.title(r"$F_{\rm AGN}$" + " Constraint", fontsize = f)
plt.xlabel(r"$\log_{10}\left(M_\star/ M_{\odot}\right)$", fontsize=f)
plt.ylabel(r"$\log_{10}F_{\rm AGN}$", fontsize=f)
plt.xlim(9.8, 12.2)
plt.xticks([10,10.4,10.8, 11.2, 11.6, 12])
plt.yscale('log')
plt.minorticks_off()
plt.ylim(1e-5, 1)
plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1, 1], labels=[-5,-4,-3, -2, -1, 0])
plt.tick_params(axis='x', labelsize=0.75 * f, length=10, width=3)
plt.tick_params(axis='y', labelsize=0.75 * f, length=10, width=3)

# Call function to plot overlay
funcs.plot_Fagn_mbins_constraint()
funcs.plot_manga_constraint()


plt.tight_layout()
plt.legend(loc = 'lower right', fontsize = 0.75*f)
plt.show()