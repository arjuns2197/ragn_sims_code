import sys
sys.path.append("..")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os


sfr = np.array([])
mass = np.array([])


path = "../ragn_illustris100_files"
directory = os.fsencode(path)

for folder_path in os.listdir(directory):

    full_folder_path = os.path.join(directory,folder_path)

    for file_path in os.listdir(full_folder_path):

        full_file_path = os.path.join(full_folder_path,file_path)
        print(full_file_path)
        file_size = os.path.getsize(full_file_path)

        if file_size/1e6 < 0.3:

            continue

        else:

            file = h5py.File(full_file_path)
            

            sfr_this_file = np.array(file['Subhalo/SubhaloSFRinRad'])
            sfr = np.append(sfr, np.log10(sfr_this_file))
            
            mass_this_file = np.array(file['Subhalo/SubhaloMassInRadType'][:,4])*1e10/0.677
            mass = np.append(mass, np.log10(mass_this_file))
  



plt.scatter(mass, sfr)
plt.show()








      




   








    

