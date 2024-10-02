import sys
sys.path.append("..")

import functions_illustris as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os


path = "../ragn_illustris100_files/df_99.pkl"
df = pd.read_pickle(path)


bhm = df['log_bhm']
rho_gas = np.log10(df['rho_gas_bh'])

plt.scatter(bhm, rho_gas)
plt.show()

