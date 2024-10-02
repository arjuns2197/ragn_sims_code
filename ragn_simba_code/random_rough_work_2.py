import sys
sys.path.append("..")

import yt
import caesar
import matplotlib.pyplot as plt
import numpy as np
import agn_functions as agn
import pandas as pd
from readgadget import readsnap
import h5py
from file_read_backwards import FileReadBackwards
f = 30

df = agn.load_simba_df()
df.to_pickle("../ragn_simba_files/data/df.pkl")



df = pd.read_pickle("../ragn_simba_files/data/df.pkl")



plt.scatter(df['mass'],np.log10(df['ker']))
plt.show()




























































