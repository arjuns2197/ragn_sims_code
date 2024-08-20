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
inds_agn = []

for i in range(len(df)):
    if df['is_agn'][i] == True:
        inds_agn.append(i)


plt.scatter(df['Lr'][inds_agn] - 16.146, df['mass'][inds_agn], c = 'red', marker = 'x')
plt.xlim(18.5,26.5)
plt.ylim(9,13)
plt.xlabel(r"$L_{1.4 \rm ~GHz} ~ \left(\rm W~Hz^{-1}\right)$", fontsize = f)
plt.ylabel(r"$\log\left(M_\star / M_\odot\right)$", fontsize = f)
plt.yticks([10,11,12],fontsize = 0.8*f)
plt.xticks([19,21,23,25],fontsize = 0.8*f)
plt.tight_layout()
plt.show()

plt.scatter(df['Lr'][inds_agn] - 16.146, df['bhm'][inds_agn], c = 'red', marker = 'x')
plt.xlim(18.5,26.5)
plt.ylim(7,11)
plt.xlabel(r"$L_{1.4 \rm ~GHz} ~ \left(\rm W~Hz^{-1}\right)$", fontsize = f)
plt.ylabel(r"$\log\left(M_{\rm BH} / M_\odot\right)$", fontsize = f)
plt.yticks([8,9,10,11],fontsize = 0.8*f)
plt.xticks([19,21,23,25],fontsize = 0.8*f)
plt.tight_layout()
plt.show()

plt.scatter(df['Lr'][inds_agn] - 16.146, df['sSFR'][inds_agn] + 9, c = 'red', marker = 'x')
plt.xlim(18.5,26.5)
plt.ylim(-5.5,0)
plt.xlabel(r"$L_{1.4 \rm ~GHz} ~ \left(\rm W~Hz^{-1}\right)$", fontsize = f)
plt.ylabel(r"$\log\left (sSFR / \rm Gyr^{-1}\right)$", fontsize = f)
plt.yticks([-5,-3,-1],fontsize = 0.8*f)
plt.xticks([19,21,23,25],fontsize = 0.8*f)
plt.tight_layout()
plt.show()

