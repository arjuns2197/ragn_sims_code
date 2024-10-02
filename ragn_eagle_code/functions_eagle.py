import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import eagleSqlTools as esql
import pandas as pd


def download_eagle_data():

    """
    Returns -> A pandas dataframe containing properties of the galaxies in the EAGLE simulation. 
    The galaxy properties queried are:

    -galaxy_id
    -z
    -stellar mass
    -black hole mass
    -black hole accretion rate
    -star formation rate

    Takes -> Nothing.
    """

    username = "qdk114"
    password = "ry500uqu"

    con = esql.connect(username, password= password)

    query = "SELECT\
            GalaxyID as galaxy_id,\
            Redshift as z,\
            MassType_Star as mass,\
            BlackHoleMass as bh_mass,\
            BlackHoleMassAccretionRate as bh_mdot,\
            StarFormationRate as sfr\
            FROM\
            RefL0100N1504_SubHalo\
            WHERE\
            SnapNum IN (27,28)"
    
    data = esql.execute_query(con, query)

    df = pd.DataFrame(data)

    df['log_mass'] = np.log10(df['mass'])
    df['log_bh_mass'] = np.log10(df['bh_mass'])
    df['log_sfr'] = np.log10(df['sfr'])
    df['log_ssfr'] = np.log10(df['sfr']/df['mass'])
    df['log_bh_mdot'] = np.log10(df['bh_mdot'])

    ER = get_ER(df['log_bh_mdot'], df['log_bh_mass'])
    df['ER'] = ER

    df = df.drop(["mass", "bh_mass", "sfr", "bh_mdot"],axis= 'columns')

    return df


def download_eagleagn_data():

  """
  Returns -> A pandas dataframe containing properties of the galaxies in the EAGLE simulation with 
  increased AGN feedback. 
  The galaxy properties queried are:

  -galaxy_id
  -z
  -stellar mass
  -black hole mass
  -black hole accretion rate
  -star formation rate

  Takes -> Nothing.
  """

  username = "qdk114"
  password = "ry500uqu"

  con = esql.connect(username, password= password)

  query = "SELECT\
          GalaxyID as galaxy_id,\
          Redshift as z,\
          MassType_Star as mass,\
          BlackHoleMass as bh_mass,\
          BlackHoleMassAccretionRate as bh_mdot,\
          StarFormationRate as sfr\
          FROM\
          AGNdT9L0050N0752_SubHalo\
          WHERE\
          SnapNum IN (27,28)"
  
  data = esql.execute_query(con, query)

  df = pd.DataFrame(data)

  df['log_mass'] = np.log10(df['mass'])
  df['log_bh_mass'] = np.log10(df['bh_mass'])
  df['log_sfr'] = np.log10(df['sfr'])
  df['log_ssfr'] = np.log10(df['sfr']/df['mass'])
  df['log_bh_mdot'] = np.log10(df['bh_mdot'])

  ER = get_ER(df['log_bh_mdot'], df['log_bh_mass'])
  df['ER'] = ER

  df = df.drop(["mass", "bh_mass", "sfr", "bh_mdot"],axis= 'columns')

  return df


def get_ER(log_mdot,log_bh_mass):

    """
    Returns -> The Eddington ratio of a black hole (Mdot/Mdot_Edd), given its
    mass and accretion rate

    Takes -> The logarithm of the black hole mass in Msun
    """

    eta = 0.1*0.15
    c_sq = (3e8)**2 
    L_edd = 1.26e31*(10**log_bh_mass)

    mdot = (10**log_mdot)*(1.989e30/3.156e7)  # In kg/s
    er = (eta*c_sq*mdot)/L_edd

    return er


def bin_galaxies(property1,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  if property1 == 'log_ssfr':
    nbins = [-np.inf,-11.5,-8]

  else:
    nbins = [10,10.25,10.5,10.75,11,11.25,11.5,11.75,12]    

  sorted_byprop1 = df.sort_values(by = [property1])
  binned_byprop1 = pd.cut(sorted_byprop1[property1], bins= nbins)
  df[property1 + "_bin"] = binned_byprop1

  return df


def get_Fagn_rmanga(log_M):
    
    a0 = -2.73
    a1 = 2.40
    a2 = -0.79
    a3 = 0.09
    
    F = a3*(log_M -10.5)**3 + a2*(log_M - 10.5)**2 + a1*(log_M - 10.5) + a0
    
    return F