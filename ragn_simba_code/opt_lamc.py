import sys
sys.path.append("..")


import caesar
import matplotlib.pyplot as plt
import numpy as np
import funcs_simba as funcs
import pandas as pd
f = 30


df = pd.read_pickle("../ragn_simba_files/data/df.pkl")

[mass_center,frac_3,ebars_3,frac_q_3,ebars_q_3,frac_sf_3,ebars_sf_3] = funcs.get_Fagn_of_lcrit(df,1e-3)



