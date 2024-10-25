import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py


def get_radio_cavag(log_pcav):

  pcav = 10**log_pcav
  pcav = pcav/1e42

  log_pcav = np.log10(pcav)

  log_pradio = (log_pcav - 1.85)/0.35

  log_pradio = log_pradio + 40

  return log_pradio


def get_rho_sf_thresh():

  rho_sf_thresh = 3.213e6 # Msun/kpc^3 or 0.13 protons per cm^3

  return rho_sf_thresh


def get_Fagn_rmanga(log_M):
    
  a0 = -2.73
  a1 = 2.40
  a2 = -0.79
  a3 = 0.09
  
  F = a3*(log_M -10.5)**3 + a2*(log_M - 10.5)**2 + a1*(log_M - 10.5) + a0
  
  return F


def bin_galaxies(property1,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  if property1 == 'sSFR':
    nbins = [-np.inf,-11.5,-8]

  else:
    nbins = [10,10.25,10.5,10.75,11,11.25,11.5,11.75,12]    

  sorted_byprop1 = df.sort_values(by = [property1])
  binned_byprop1 = pd.cut(sorted_byprop1[property1], bins= nbins)
  df[property1 + "_bin"] = binned_byprop1

  return df


def get_epsilon_k(rho):

  rho_sf_thresh =  get_rho_sf_thresh() # Msun/kpc^3

  epsilon = min(rho/(0.05*rho_sf_thresh), 0.2)

  return epsilon


def get_edot_k(mdot_bh_si, rho):

  c = 2.998e8

  epsilon = get_epsilon_k(rho)
  edot = (epsilon*mdot_bh_si*c**2)*1e7  # erg/s
  

  return edot


def get_ker(L_k, Ledd):
  
  if Ledd == 0:
    ker = 0

  else:
    ker = L_k/Ledd

  return ker



