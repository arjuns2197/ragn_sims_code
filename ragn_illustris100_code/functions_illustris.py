########################################################################################
# MAKE SURE TO USE NUMPY VERSION 2.2.3 TO BE ABLE TO READ THE PICKLED DATAFRAME FILES
########################################################################################

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from scipy.stats import binom
from scipy.optimize import curve_fit


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

def bin_galaxies_4mbins(property1,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  if property1 == 'sSFR':
    nbins = [-np.inf,-11.5,-8]

  else:
    nbins = [10.4,10.8,11.2,11.6,12]    

  sorted_byprop1 = df.sort_values(by = [property1])
  binned_byprop1 = pd.cut(sorted_byprop1[property1], bins= nbins)
  df[property1 + "_bin"] = binned_byprop1

  return df

def get_epsilon_k(rho, fth = 0.005, epsilonk_max = 0.2):

  rho_sf_thresh =  get_rho_sf_thresh() # Msun/kpc^3

  epsilon = np.minimum(rho/(fth*rho_sf_thresh), epsilonk_max)

  return epsilon


def get_edot_k(mdot_bh_si, rho, fth = 0.005, epsilonk_max = 0.2):

  c = 2.998e8

  epsilon = get_epsilon_k(rho, fth, epsilonk_max)
  edot = (epsilon*mdot_bh_si*c**2)*1e7  # erg/s
  

  return edot


def get_ker(L_k, Ledd):
  
  if Ledd == 0:
    ker = 0

  else:
    ker = L_k/Ledd

  return ker


def get_binom_ebars(frac, size):


    ppf_low = binom.ppf(0.16,size, frac)
    ppf_median = binom.ppf(0.5,size, frac)
    ppf_high = binom.ppf(0.84,size, frac)

    frac_low = ppf_low/size
    frac_high = ppf_high/size

    ebar_low = frac - frac_low
    ebar_high = frac_high - frac

    return [ebar_low, ebar_high]


def get_Fagn_of_lcrit(df,lambda_crit):

  is_strong_ragn = []
  for i in range(len(df)):

    if df['ker'][i] > lambda_crit:
      is_strong_ragn.append(True)

    else:
      is_strong_ragn.append(False)


  df['is_strong_agn'] = is_strong_ragn


  df_mass_bins = bin_galaxies_4mbins("log_mass", df)
  grped_df_mass = df_mass_bins.groupby('log_mass_bin')

  size = []
  frac = []
  ebars_low = []
  ebars_high = []
  mass_center = []

  frac_q = []
  frac_sf = []
  ebars_q_low = []
  ebars_q_high = []

  size_q = []
  size_sf = []
  ebars_sf_low = []
  ebars_sf_high = []


  for mass, group in grped_df_mass:

    string = str(mass)
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    mass_center.append(center)

    n_agn_all = group['is_strong_agn'].sum()
    size_all = len(group)
    f_all = n_agn_all/size_all
    size.append(size_all)
    frac.append(f_all)

    [ebar_low, ebar_high] = get_binom_ebars(f_all,size_all)
    ebars_low.append(ebar_low)
    ebars_high.append(ebar_high)

    s_q = 0
    n_agn_q = 0

    s_sf = 0
    n_agn_sf = 0

    for i in range(len(group)):
        
      if group['log_ssfr'].iloc[i] < -11.5:

        s_q = s_q + 1

        if group['is_strong_agn'].iloc[i] == True:
          n_agn_q = n_agn_q + 1


      if group['log_ssfr'].iloc[i] > -11.5:

        s_sf = s_sf + 1

        if group['is_strong_agn'].iloc[i] == True:
          n_agn_sf = n_agn_sf + 1

    size_q.append(s_q)
    frac_q.append(n_agn_q/s_q)

    [ebar_q_low, ebar_q_high] = get_binom_ebars(n_agn_q/s_q,s_q)
    ebars_q_low.append(ebar_q_low)
    ebars_q_high.append(ebar_q_high)
        


    size_sf.append(s_sf)
    frac_sf.append(n_agn_sf/s_sf)

    [ebar_sf_low, ebar_sf_high] = get_binom_ebars(n_agn_sf/s_sf,s_sf)
    ebars_sf_low.append(ebar_sf_low)
    ebars_sf_high.append(ebar_sf_high)

  mass_center = np.array(mass_center)
  frac = np.array(frac)
  ebars_low = np.array(ebars_low)
  ebars_high = np.array(ebars_high)


  size_q = np.array(size_q)
  frac_q = np.array(frac_q)
  ebars_q_low = np.array(ebars_q_low)
  ebars_q_high = np.array(ebars_q_high)

  size_sf = np.array(size_sf)
  frac_sf = np.array(frac_sf)
  ebars_sf_low = np.array(ebars_sf_low)
  ebars_sf_high = np.array(ebars_sf_high)

  return [mass_center,frac,ebars_low,ebars_high,frac_q,ebars_q_low,ebars_q_high,frac_sf,ebars_sf_low, ebars_sf_high]


def Fagn_manga(log_M, a3, a2, a1, a0):
  
  F = a3*(log_M -10.5)**3 + a2*(log_M - 10.5)**2 + a1*(log_M - 10.5) + a0
  
  return F

def ddm_Fagn_manga(log_M, a3, a2, a1):
  
  dFdm = 3*a3*(log_M -10.5)**2 + 2*a2*(log_M - 10.5) + a1
  
  return dFdm   


def optimize_lamc(df):

  def opt_lamc(theta):

    log_lam_c = theta[0]
    lam_c = 10**log_lam_c

    if log_lam_c >= -2:
      print("1111")
      return 1e5

    else:
      fagn_pred = get_Fagn_of_lcrit(df,lam_c)
      log_masses = fagn_pred[0]
      log_fracs = np.log10(fagn_pred[1] + 1e-8)
      log_fracs_q = np.log10(fagn_pred[3] + 1e-8)
      log_fracs_sf = np.log10(fagn_pred[5] + 1e-8)

      print(log_fracs_sf)

      log_ms = np.linspace(10,12,1000)
      interp_fagn = UnivariateSpline(log_masses,log_fracs, k =4)
      fagn_pred = interp_fagn(log_ms)

      interp_fagn_q = UnivariateSpline(log_masses,log_fracs_q, k =4)
      fagn_pred_q = interp_fagn_q(log_ms)

      interp_fagn_sf = UnivariateSpline(log_masses,log_fracs_sf, k =4)
      fagn_pred_sf = interp_fagn_sf(log_ms)

      print(interp_fagn_sf(log_masses))

      ddm_interp = interp_fagn.derivative(1)
      ddm_pred = ddm_interp(log_ms)

      fagn_manga = Fagn_manga(log_ms,0.09,-0.81,2.43,-2.75)
      ddm_manga = ddm_Fagn_manga(log_ms,0.09,-0.81,2.43)

      # plt.scatter(log_ms,fagn_pred, c = 'k')
      # plt.scatter(log_ms, fagn_pred_q, c = 'r')
      # plt.scatter(log_ms, fagn_pred_sf, c = 'b')
      # plt.plot(log_ms,fagn_manga)
      # plt.annotate(str(log_lam_c),(11,-4))
      # plt.show()

      if np.isnan(ddm_pred).all() == True:
        print("2222")
        return 1e5

      if np.isnan(fagn_pred).all() == True:
        print("3333")
        return 1e5
                  
      if np.isnan(fagn_pred_sf).all() == True:
        print("4444")
        return 1e5

      if np.isnan(fagn_pred_q).all() == True:
        print("5555")
        return 1e5

      rmse_slopes = root_mean_squared_error(ddm_manga, ddm_pred)
      rmse = root_mean_squared_error(fagn_manga,fagn_pred)
      rmse_q_sf = root_mean_squared_error(fagn_pred_sf, fagn_pred_q)

      fagn_manga_at_m11 = [Fagn_manga(11,0.09,-0.81,2.43,-2.75)]
      fagn_pred_at_m11 = [interp_fagn(11)]

      rmse_at_m11 = root_mean_squared_error(fagn_manga_at_m11,fagn_pred_at_m11)

      if rmse > 4:
        print("6666")
        return 1e5
    
      if rmse_slopes > 4:
        print('7777')
        return 1e5

      if rmse_q_sf > 4:
        print("8888")
        return 1e5
      
      if rmse_at_m11 > 4:
        print("9999")
        return 1e5
    
      #print("Total Score:", rmse + rmse_slopes + rmse_q_sf)
      print("Metric Score:", rmse_slopes**2 + rmse_q_sf**2 + rmse_at_m11**2)

      return rmse_slopes**2 + rmse_q_sf**2 + rmse_at_m11**2



  res = minimize(opt_lamc, x0 = -4.35, method = "Nelder-Mead")
  optimal_lamc = res.x[0]

  return optimal_lamc


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


    plt.plot(log_Ms, 10**log_Fs, label = 'MaNGA All', linewidth = 4, c = 'k')
    plt.fill_between(log_Ms, 10**log_Fs_ll, 10**log_Fs_ul, color = 'tab:gray', alpha = 0.5)


def get_eddscaled_mdot(log_bhm, log_bhmdot):

  bhmdot_in_si = 10**log_bhmdot*(1.989e30/3.156e7)
  bhm_in_si = 10**log_bhm*(1.989e30)
  G = 6.67e-11
  mp = 1.673e-27
  epsilon_r = 0.2
  sigma_t = 6.65e-29
  c = 3e8

  numer = 4*np.pi*G*bhm_in_si*mp
  denom = epsilon_r*sigma_t*c

  mdot_edd_si = numer/denom

  edd_scaled_mdot = bhmdot_in_si/mdot_edd_si

  return edd_scaled_mdot


def chi_of_mbh(log_bhm, log_chi_0,log_bhm0,log_beta,log_chi_max):

  
  chi_0 = 10**log_chi_0
  chi_max = 10**log_chi_max
  beta = 10**log_beta

  term1 = chi_0*(10**log_bhm/10**log_bhm0)**beta
  term2 = chi_max

  chi = np.minimum(term1, term2)

  return chi


def get_Fagn_of_chi_to_optimize(df,lambda_crit, chi_parms, log_lamc_offset):

  log_chi0, log_M0, log_beta, log_chi_max = chi_parms

  chi = chi_of_mbh(df['log_bhm'], log_chi0,log_M0,log_beta,log_chi_max)
  M_dot = get_eddscaled_mdot(df['log_bhm'],df['log_bhmdot']) 

  is_radio_mode = M_dot < chi
  df['is_radio_mode'] = is_radio_mode
  
  epsilon_kin = get_epsilon_k(df['rho_gas_bh'])
  epsilon_r = 0.2
  ker_of_chi = epsilon_kin*M_dot/epsilon_r

  df['ker_of_chi'] = ker_of_chi
  df.loc[~df['is_radio_mode'], 'ker_of_chi'] = 0

  condition_ker = np.log10(df['ker_of_chi']) > np.log10(lambda_crit) + log_lamc_offset
  df['is_strong_agn'] = np.zeros(len(df),dtype = bool)
  df.loc[condition_ker,'is_strong_agn'] = np.ones(len(df['is_strong_agn'][condition_ker]), dtype = bool)

  df_mass_bins = bin_galaxies("log_mass", df)
  grped_df_mass = df_mass_bins.groupby('log_mass_bin', observed = False)

  size = []
  frac = []

  mass_center = []

  for mass, group in grped_df_mass:

    string = str(mass)
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    mass_center.append(center)

    n_agn_all = group['is_strong_agn'].sum()
    size_all = len(group)
    f_all = n_agn_all/size_all
    size.append(size_all)
    frac.append(f_all)

  mass_center = np.array(mass_center)
  frac = np.array(frac)
  size = np.array(size)

  return [mass_center, frac, size]



def get_Fagn_of_chi_to_plot(df,lambda_crit, chi_parms, log_lamc_offset):

  log_chi0, log_M0, log_beta, log_chi_max = chi_parms

  chi = chi_of_mbh(df['log_bhm'], log_chi0,log_M0,log_beta,log_chi_max)
  M_dot = get_eddscaled_mdot(df['log_bhm'],df['log_bhmdot']) 

  is_radio_mode = M_dot < chi
  df['is_radio_mode'] = is_radio_mode
  
  epsilon_kin = get_epsilon_k(df['rho_gas_bh'])
  epsilon_r = 0.2
  ker_of_chi = epsilon_kin*M_dot/epsilon_r

  df['ker_of_chi'] = ker_of_chi
  df.loc[~df['is_radio_mode'], 'ker_of_chi'] = 0

  condition_ker = np.log10(df['ker_of_chi']) > np.log10(lambda_crit) + log_lamc_offset
  df['is_strong_agn'] = np.zeros(len(df),dtype = bool)
  df.loc[condition_ker,'is_strong_agn'] = np.ones(len(df['is_strong_agn'][condition_ker]), dtype = bool)

  df_mass_bins = bin_galaxies("log_mass", df)
  grped_df_mass = df_mass_bins.groupby('log_mass_bin', observed = False)

  size = []
  frac = []
  ebars_low = []
  ebars_high = []
  mass_center = []

  frac_q = []
  frac_sf = []
  ebars_q_low = []
  ebars_q_high = []

  size_q = []
  size_sf = []
  ebars_sf_low = []
  ebars_sf_high = []


  for mass, group in grped_df_mass:

    string = str(mass)
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    mass_center.append(center)

    n_agn_all = group['is_strong_agn'].sum()
    size_all = len(group)
    f_all = n_agn_all/size_all
    size.append(size_all)
    frac.append(f_all)

    [ebar_low, ebar_high] = get_binom_ebars(f_all,size_all)
    ebars_low.append(ebar_low)
    ebars_high.append(ebar_high)

    s_q = 0
    n_agn_q = 0

    s_sf = 0
    n_agn_sf = 0

    for i in range(len(group)):
        
      if group['log_ssfr'].iloc[i] < -11.5:

        s_q = s_q + 1

        if group['is_strong_agn'].iloc[i] == True:
          n_agn_q = n_agn_q + 1


      if group['log_ssfr'].iloc[i] > -11.5:

        s_sf = s_sf + 1

        if group['is_strong_agn'].iloc[i] == True:
          n_agn_sf = n_agn_sf + 1

    size_q.append(s_q)
    frac_q.append(n_agn_q/s_q)

    [ebar_q_low, ebar_q_high] = get_binom_ebars(n_agn_q/s_q,s_q)
    ebars_q_low.append(ebar_q_low)
    ebars_q_high.append(ebar_q_high)
        

    size_sf.append(s_sf)
    frac_sf.append(n_agn_sf/s_sf)

    [ebar_sf_low, ebar_sf_high] = get_binom_ebars(n_agn_sf/s_sf,s_sf)
    ebars_sf_low.append(ebar_sf_low)
    ebars_sf_high.append(ebar_sf_high)

  mass_center = np.array(mass_center)
  frac = np.array(frac)
  size = np.array(size)
  ebars_low = np.array(ebars_low)
  ebars_high = np.array(ebars_high)

  size_q = np.array(size_q)
  frac_q = np.array(frac_q)
  ebars_q_low = np.array(ebars_q_low)
  ebars_q_high = np.array(ebars_q_high)

  size_sf = np.array(size_sf)
  frac_sf = np.array(frac_sf)
  ebars_sf_low = np.array(ebars_sf_low)
  ebars_sf_high = np.array(ebars_sf_high)

  return [mass_center,frac,ebars_low,ebars_high,frac_q,ebars_q_low,ebars_q_high,frac_sf,ebars_sf_low, ebars_sf_high,size]



def get_rms_fagn(frac_manga, frac):

  frac = frac + 1e-8
  frac = np.log10(frac)

  rms = root_mean_squared_error(frac_manga, frac)

  return rms


def get_dev_slope(masses,frac_manga, frac):

  def line(x,m,c):

    y = m*x + c

    return y

  frac = frac + 1e-6
  frac = np.log10(frac)

  parms_pred, cov = curve_fit(line, masses, frac)
  parms_manga, cov = curve_fit(line, masses, frac_manga)

  dev = np.abs(parms_manga[0] - parms_pred[0])
  print(parms_manga[0])
  print(parms_pred[0])

  return dev

def get_rms_sfq(frac_q, frac_sf):

  frac_q = frac_q + 1e-6
  frac_q = np.log10(frac_q)
  frac_sf = frac_sf + 1e-6
  frac_sf = np.log10(frac_sf)

  rms = root_mean_squared_error(frac_q, frac_sf)

  return rms


lambda_crit = 1e-3

log_chi0_start = -1
log_m0_start = 9
log_beta_start = 0.3
log_chimax_start = -3
log_lamc_offset_start = -1

def minimize_chi2error(df):

  def chi2_error(all_parms):

    print(all_parms)
    chi_parms = all_parms[:-1]
    log_lamc_offset = all_parms[-1]

    log_mass,fagn_of_chi,size = get_Fagn_of_chi_to_optimize(df,
    lambda_crit,chi_parms,log_lamc_offset)
    nagn_of_chi = fagn_of_chi*size

    log_fagn_manga = get_Fagn_rmanga(log_mass)
    nagn_manga = size*10**log_fagn_manga

    sigma2_manga = size*(1-10**log_fagn_manga)*10**log_fagn_manga


    chi2 = ((nagn_manga - nagn_of_chi)**2/sigma2_manga).sum()

    print(chi2)

    return chi2

  initial_guess = [log_chi0_start, log_m0_start, log_beta_start, log_chimax_start,log_lamc_offset_start]

  res = minimize(chi2_error, initial_guess, method = 'Nelder-Mead')
  best_parms = res.x

  print("Best Parameters are: ", best_parms)

  return best_parms




def is_above_modified_tngline(log_bhm, Mdot, all_parms_opt):
    log_bhm = np.asarray(log_bhm)
    Mdot = np.asarray(Mdot)

    chi_parms_opt = all_parms_opt[:-1]
    log_lamc_offset_opt = all_parms_opt[-1]
    log_chi0_opt, log_M0_opt, log_beta_opt, log_chi_max_opt = chi_parms_opt

    chi_vals = chi_of_mbh(log_bhm, log_chi0_opt, log_M0_opt, log_beta_opt, log_chi_max_opt)

    return Mdot > chi_vals


def is_below_modified_tngline(log_bhm, Mdot, all_parms_opt):
    log_bhm = np.asarray(log_bhm)
    Mdot = np.asarray(Mdot)

    chi_parms_opt = all_parms_opt[:-1]
    log_lamc_offset_opt = all_parms_opt[-1]
    log_chi0_opt, log_M0_opt, log_beta_opt, log_chi_max_opt = chi_parms_opt

    chi_vals = chi_of_mbh(log_bhm, log_chi0_opt, log_M0_opt, log_beta_opt, log_chi_max_opt)

    return Mdot <= chi_vals


def is_above_default_tngline(log_bhm, Mdot):
    log_bhm = np.asarray(log_bhm)
    Mdot = np.asarray(Mdot)

    log_chi0_def, log_M0_def, log_beta_def, log_chi_max_def = -2.70, 8, 0.3, -0.70

    chi_vals = chi_of_mbh(log_bhm, log_chi0_def, log_M0_def, log_beta_def, log_chi_max_def)

    return Mdot > chi_vals


def is_below_default_tngline(log_bhm, Mdot):  
    log_bhm = np.asarray(log_bhm)
    Mdot = np.asarray(Mdot)

    log_chi0_def, log_M0_def, log_beta_def, log_chi_max_def = -2.70, 8, 0.3, -0.70

    chi_vals = chi_of_mbh(log_bhm, log_chi0_def, log_M0_def, log_beta_def, log_chi_max_def)

    return Mdot <= chi_vals


def is_between_def_mod(log_bhm, Mdot, all_parms_opt):
    log_bhm = np.asarray(log_bhm)
    Mdot = np.asarray(Mdot)

    above_mod = is_above_modified_tngline(log_bhm, Mdot, all_parms_opt)
    below_def = is_below_default_tngline(log_bhm, Mdot)

    return above_mod & below_def  # element-wise AND



def plot_Fagn_mbins_constraint():
  f = 30
  Ms = np.array([10.2, 10.6, 11, 11.4, 11.8])
  log_Fracs_q = np.array([-1.66, -1.17, -0.84])
  log_Fracs_sf = np.array([-1.62, -1.58, -1.10, -0.65])
  log_ebars_q = np.array([[0.1, 0.05, 0.05], 
                          [0.09, 0.05, 0.05]])
  log_ebars_sf = np.array([[0.4, 0.22, 0.13, 0.22], 
                            [0.37, 0.19, 0.12, 0.17]])
  
  log_Fagn_uls = np.array([-0.78, -1.88, -0.30])

  # Define colors using hex codes
  red_hex = '#d62728'
  blue_hex = '#1f77b4'

  # Convert to linear
  Fracs_q = 10**log_Fracs_q
  Fracs_sf = 10**log_Fracs_sf

  # Compute asymmetric errors in linear space
  q_err_low = Fracs_q - 10**(log_Fracs_q - log_ebars_q[0])
  q_err_high = 10**(log_Fracs_q + log_ebars_q[1]) - Fracs_q
  sf_err_low = Fracs_sf - 10**(log_Fracs_sf - log_ebars_sf[0])
  sf_err_high = 10**(log_Fracs_sf + log_ebars_sf[1]) - Fracs_sf

  # Fill between for SF and Q — central line removed
  plt.fill_between(Ms[1:], Fracs_sf - sf_err_low, Fracs_sf + sf_err_high,
                    color=blue_hex, alpha=0.5, label='Observed Star-Forming')
  plt.fill_between(Ms[2:], Fracs_q - q_err_low, Fracs_q + q_err_high,
                    color=red_hex, alpha=0.5, label='Observed Quiescent')

  # Example: single upper limit arrow (Q2 bin)
  plt.hlines(10**log_Fagn_uls[1], Ms[1] - 0.05, Ms[1] + 0.05, colors=red_hex, linewidth=1, alpha=0.5)
  plt.annotate('', xy=(Ms[1], 10**(log_Fagn_uls[1] - 0.3)), xytext=(Ms[1], 10**log_Fagn_uls[1]),
              arrowprops=dict(arrowstyle='-|>', color=red_hex, linewidth=1, alpha=0.5))