import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import eagleSqlTools as esql
import pandas as pd
import functions_eagle as funcs

#data = funcs.download_eagle_data()
#data.to_pickle("../ragn_eagle_files/df.pkl")

df = pd.read_pickle("../ragn_eagle_files/df.pkl")

ER = funcs.get_ER(np.log10(df['bh_mdot']),df['log_bh_mass'])
df['ER'] = ER

print(ER.max())

plt.scatter(df['log_mass'], ER)
plt.show()
