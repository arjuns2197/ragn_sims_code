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
epsilon = df['epsilon_k']

plt.scatter(bhm, epsilon)
plt.show()




    

    

