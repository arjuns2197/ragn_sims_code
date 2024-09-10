import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import eagleSqlTools as esql
import pandas as pd
import functions_eagle as funcs

data = funcs.download_eagleagn_data()
data.to_pickle("../ragn_eagle_files/data_agn_variant/df.pkl")

df = pd.read_pickle("../ragn_eagle_files/data_agn_variant/df.pkl")


ER = df['ER'] 

print(ER.max())

plt.scatter(df['log_mass'], ER)
plt.show()
