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



def Fagn_manga(log_M, a3, a2, a1, a0):
  
  F = a3*(log_M -10.5)**3 + a2*(log_M - 10.5)**2 + a1*(log_M - 10.5) + a0
  
  return F


def plot_manga_constraint():

    a3_mean = 0.09
    a2_mean = -0.81
    a1_mean = 2.43
    a0_mean = -2.75

    mu = np.array([a3_mean, a2_mean, a1_mean, a0_mean])

    cov_mat = np.array([[0.0002, -0.0014, 0.0029, -0.0015],
                        [-0.0014, 0.0106, -0.0223, 0.012],
                        [0.003, -0.0223, 0.0486, -0.0276],
                        [-0.0015, 0.012, -0.0276, 0.0175]])
    

    m = 100
    n = 10000
    coeffs_array = np.random.multivariate_normal(mu, cov= cov_mat, size = n)

    log_Ms = np.linspace(10,12,m)
    log_Fs_all = []

    for i in coeffs_array:
       
       a3,a2,a1,a0 = i

       Fs = Fagn_manga(log_Ms,a3,a2,a1,a0)
       log_Fs_all.append(Fs)
    
    log_Fs_all = np.array(log_Fs_all)
    log_Fs_ul = np.percentile(log_Fs_all, 84, axis = 0)
    log_Fs_ll = np.percentile(log_Fs_all, 16, axis = 0)
    log_Fs = np.percentile(log_Fs_all, 50, axis = 0)


    plt.plot(log_Ms, 10**log_Fs, label = r'$F_{\rm AGN} ~\rm MaNGA$', linewidth = 2.5, c = 'k')
    plt.fill_between(log_Ms, 10**log_Fs_ll, 10**log_Fs_ul, color = 'tab:gray', alpha = 0.5)



def get_Fagn_of_lcrit(df,lambda_crit):

    ER = df['ER']


    is_strong_ragn = []
    for i in range(len(df)):

        if df['ER'][i] > lambda_crit:

            if df['ER'][i] < 0.02:

                is_strong_ragn.append(True)
            
            else:
                is_strong_ragn.append(False)
            

        else:
            is_strong_ragn.append(False)


    df['is_strong_ragn'] = is_strong_ragn


    df_mass_bins = bin_galaxies("log_mass", df)
    grped_df_mass = df_mass_bins.groupby('log_mass_bin')


    size = []
    frac = []
    mass_center = []


    frac_q = []
    frac_sf = []
    ebars_q = []
    ebars_sf = []
    size_q = []
    size_sf = []

    for mass, group in grped_df_mass:

        string = str(mass)
        split = string.split(',')
        left = float(split[0][1:])
        right = float(split[1][:-1])
        center = (left + right)/2
        mass_center.append(center)

        n_agn_all = group['is_strong_ragn'].sum()
        size_all = len(group)
        f_all = n_agn_all/size_all
        size.append(size_all)
        frac.append(f_all)

        s_q = 0
        n_agn_q = 0

        s_sf = 0
        n_agn_sf = 0

        for i in range(len(group)):
            
            if group['log_ssfr'].iloc[i] < -11:

                s_q = s_q + 1

                if group['is_strong_ragn'].iloc[i] == True:
                    n_agn_q = n_agn_q + 1


            if group['log_ssfr'].iloc[i] > -11:

                s_sf = s_sf + 1

                if group['is_strong_ragn'].iloc[i] == True:
                    n_agn_sf = n_agn_sf + 1
        

        print(mass_center)

        size_q.append(s_q)
        frac_q.append(n_agn_q/s_q)
        ebars_q.append(2*np.sqrt(n_agn_q/(s_q)**2))


        size_sf.append(s_sf)
        frac_sf.append(n_agn_sf/s_sf)
        ebars_sf.append(2*np.sqrt(n_agn_sf/(s_sf)**2))
        
    mass_center = np.array(mass_center)
    size_q = np.array(size_q)
    frac_q = np.array(frac_q)
    ebars_q = np.array(ebars_q)

    size_sf = np.array(size_sf)
    frac_sf = np.array(frac_sf)
    ebars_sf = np.array(ebars_sf)

    return [mass_center,frac,frac_q,ebars_q,frac_sf,ebars_sf]
