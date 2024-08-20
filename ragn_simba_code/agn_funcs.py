import sys
sys.path.append("..")

import caesar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from readgadget import readsnap
log_lambda_max = 1


def load_simba_df():

    sfr = []
    bhm_dot = []
    bhm = []
    fedd = []
    Lradio = []
    sigma = []
    ssfr = []
    stellar_mass = []



    for num in range(143,152):
        num = str(num)
        path = "../ragn_simba_files/data/m100n1024_" + num + ".hdf5"
        data = caesar.load(path)

        for gal in data.galaxies:
            
            sfr1 = gal.sfr.to_value(units = None, equivalence = None)
            sfr1 = np.log10(sfr1)
            sfr.append(sfr1)

            bhm_dot.append(gal.bhmdot.to_value(units = None, equivalence = None))

            bhm1 = gal.masses['bh'].to_value(units = None, equivalence = None)
            bhm1 = np.log10(bhm1)
            bhm.append(bhm1)

            sm = gal.masses['stellar'].to_value(units = None, equivalence = None)
            log_sm = np.log10(sm)
            stellar_mass.append(log_sm)

            fedd_value = gal.bh_fedd.to_value(units = None, equivalence = None)
            fedd.append(fedd_value)

            sigma_value = gal.velocity_dispersions['stellar'].to_value(units = None, equivalence = None)
            sigma_value = np.log10(sigma_value)
            sigma.append(sigma_value)

            ssfr_value = gal.sfr.to_value(units = None, equivalence = None)/gal.masses['stellar'].to_value(units = None, equivalence = None)

            if ssfr_value == 0:
                ssfr.append(-1*np.inf)

            else:
                ssfr.append(np.log10(ssfr_value))
            




    dict = {"sfr" : sfr, 'bhmdot' : bhm_dot, "bhm" : bhm, "fedd" : fedd, "Lr" : Lradio,
    "sigma" : sigma, "sSFR" : ssfr, "mass" : stellar_mass}


    df = pd.DataFrame(dict)

    return df