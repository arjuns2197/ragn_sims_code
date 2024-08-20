import sys
sys.path.append("..")

import caesar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
log_lambda_max = 1
p = (17/12) + 2 ## Dummy Lagn model ; Lagn = (M_dot_bhm)^ps


def get_sf_radio_thermal(log_sfr):

    sfr = 10**log_sfr

    nu = 1.4
    coef = 5.5e20
    term1 = (nu)**(-0.1)
    
    L_sf_t = (nu*1e9)*coef*term1*sfr*(1e7)
    L_sf_t = L_sf_t
    
    return L_sf_t



def get_sf_radio_non_thermal(log_sfr):

    sfr = 10**log_sfr
    
    nu = 1.4
    coef = 5.3e21
    term1 = (nu)**(-0.8)
    
    L_sf_nt = (nu*1e9)*coef*term1*sfr*(1e7)
    L_sf_nt = L_sf_nt
    
    return L_sf_nt


def get_sf_radio(log_sfr):
    
    L_t = get_sf_radio_thermal(log_sfr)
    L_nt = get_sf_radio_non_thermal(log_sfr)
    
    Lsfr = np.log10(L_t + L_nt)
    
    return Lsfr
    

def get_agn_feedback(bhm_dot, fedd, log_bhm, log_mstar):
    
    if log_bhm > 8:
        if 0 < fedd < 0.02:

            if log_mstar > 9.5:
                L_agn = (1e30)*(bhm_dot*6.3e25/4e17)**(17/12)
                L_agn = np.log10(L_agn)
            
            else:
                L_agn = 0
        
        else:
            L_agn = 0
    
    else:
        L_agn = 0
        
    return L_agn




def get_total_radio(log_sfr, bhm_dot, fedd, log_bhm, log_mstar):
    

    L_sfr = 10**get_sf_radio(log_sfr)
    L_agn = 10**get_agn_feedback(bhm_dot,fedd, log_bhm, log_mstar)
    
    L_radio = L_sfr + L_agn
    L_radio = np.log10(L_radio)

    return L_radio


def check_is_agn(Lagn):
    
    '''
    Returns --> Either True or False for a galaxy's AGN-ness
    Inputs --> The SFR and radio luminosity of the galaxy whose AGN-ness
    is to be determined.
    '''

    if Lagn > 0:
        return True
    else:
        return False

vector_is_agn = np.vectorize(check_is_agn)


def check_is_sagn(Lagn,ER):
    
    '''
    Returns --> Either True or False for a galaxy's AGN-ness
    Inputs --> The SFR and radio luminosity of the galaxy whose AGN-ness
    is to be determined.
    '''

    if Lagn > 0:
        if ER > 0.001:
            return True

        else:
            return False
    else:
        return False



vector_is_agn = np.vectorize(check_is_sagn)


def p_cav(log_Lr):

  log_pcav = 0.75*log_Lr + 13.91

  return log_pcav


def get_er_pcav(Lr, vd):

  pcav = p_cav(Lr)
  log_mass_BH = np.float64(8.12+4.24*np.log10(10**vd/200))  # log(M_bh/Msun)
  log_L_Edd= np.float64(np.log10(1.26e38)+log_mass_BH) 
  L_Edd= 10**log_L_Edd

  er = 10**pcav/L_Edd

  return er


def get_ER(log_bhm,radio_l,vd,is_agn): 
  
  bhm = 10**log_bhm

  if bhm > 0:

    if is_agn == True:
        lambda_ = get_er_pcav(radio_l, vd)

    else:
        return 9.999999e-20

  else:
      lambda_ = 9.999999e-20

  return lambda_


def get_strong_agn_frac(property,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  sorted_byprop = df.sort_values(by = [property])
  binned_byprop = pd.cut(sorted_byprop[property], bins= [10,10.25,10.5,10.75,11,11.25,11.5,11.75,12])
  sorted_byprop['bin prop'] = binned_byprop

  grped_byprop = sorted_byprop.groupby(["bin prop"])

  prop_center = []
  frac = []
  size = []

  for name, group in grped_byprop:

    string = str(name[0])
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    prop_center.append(center)

    average = group["is_strong_agn"].mean()
    frac.append(average)
    size.append(len(group))

  frac = np.array(frac)
  prop_center = np.array(prop_center)
  size = np.array(size)
  
  return [frac, prop_center, size]


def get_strong_agn_frac_fedd(property,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  sorted_byprop = df.sort_values(by = [property])
  binned_byprop = pd.cut(sorted_byprop[property], bins= [10,10.25,10.5,10.75,11,11.25,11.5,11.75,12])
  sorted_byprop['bin prop'] = binned_byprop

  grped_byprop = sorted_byprop.groupby(["bin prop"])

  prop_center = []
  frac = []
  size = []

  for name, group in grped_byprop:

    string = str(name[0])
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    prop_center.append(center)

    average = group["is_strong_agn_fedd"].mean()
    frac.append(average)
    size.append(len(group))

  frac = np.array(frac)
  prop_center = np.array(prop_center)
  size = np.array(size)
  
  return [frac, prop_center, size]


def get_strong_agn_frac_dummy(property,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  nbins = 14

  sorted_byprop = df.sort_values(by = [property])

  binned_byprop = pd.cut(sorted_byprop[property], bins= nbins)

  sorted_byprop['bin prop'] = binned_byprop

  grped_byprop = sorted_byprop.groupby(["bin prop"])

  prop_center = []
  frac = []
  size = []

  for name, group in grped_byprop:

    string = str(name[0])
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    prop_center.append(center)

    average = group["is_strong_agn_dummy"].mean()
    frac.append(average)
    size.append(len(group))

  frac = np.array(frac)
  prop_center = np.array(prop_center)
  size = np.array(size)
  
  return [frac, prop_center, size]


def get_Fagn_rmanga(log_M):
    
    a0 = -2.73
    a1 = 2.40
    a2 = -0.79
    a3 = 0.09
    
    F = a3*(log_M -10.5)**3 + a2*(log_M - 10.5)**2 + a1*(log_M - 10.5) + a0
    
    return F


def load_simba_df():

    sfr = []
    bhm_dot = []
    bhm = []
    fedd = []
    Lradio = []
    Lagn = []
    Lsfr = []
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

            L = get_total_radio(sfr1,gal.bhmdot,gal.bh_fedd,bhm1,log_sm)
            L_agn = get_agn_feedback(gal.bhmdot, gal.bh_fedd, bhm1,log_sm)

            Lradio.append(L)
            Lagn.append(L_agn)
            Lsfr.append(get_sf_radio(sfr1))
            sigma_value = gal.velocity_dispersions['stellar'].to_value(units = None, equivalence = None)
            sigma_value = np.log10(sigma_value)
            sigma.append(sigma_value)
            ssfr_value = gal.sfr.to_value(units = None, equivalence = None)/gal.masses['stellar'].to_value(units = None, equivalence = None)

            if ssfr_value == 0:
                ssfr.append(-1*np.inf)

            else:
                ssfr.append(np.log10(ssfr_value))
            




    dict = {"sfr" : sfr, 'bhmdot' : bhm_dot, "bhm" : bhm, "fedd" : fedd, "Lr" : Lradio,
    "sigma" : sigma, "sSFR" : ssfr, "mass" : stellar_mass,
    "Lagn" : Lagn, "Lsfr" : Lsfr}


    df = pd.DataFrame(dict)

    return df


def load_doug_df():

    gal_id = []
    sfr = []
    bhm_dot = []
    bhm = []
    fedd = []
    Lradio = []
    Lagn = []
    Lsfr = []
    sigma = []
    ssfr = []
    stellar_mass = []
    is_agn = []
    is_strong_agn_fedd = []
    fedd_gt_m2 = []
    ER = []
    is_strong_agn = []
    gas_mass = []
    gas_temp = []
    cgm_temp = []

    for num in range(142,151):
        num = str(num)
        path = "../ragn_simba_files/doug_data/caesar_" + num + ".hdf5"
        data = caesar.load(path)

        for gal in data.galaxies:

            id = gal.GroupID
            gal_id.append(id)
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

            gm = gal.masses['gas'].to_value(units = None, equivalence = None)
            log_gm = np.log10(gm)
            gas_mass.append(log_gm)

            gas_temp.append(gal.temperatures['mass_weighted'].to_value(units = None, equivalence = None))
            cgm_temp.append(gal.temperatures['mass_weighted_cgm'].to_value(units = None, equivalence = None))

            fedd_value = gal.bh_fedd.to_value(units = None, equivalence = None)
            fedd.append(fedd_value)

            L = get_total_radio(sfr1,gal.bhmdot,gal.bh_fedd,bhm1,log_sm)
            L_agn = get_agn_feedback(gal.bhmdot, gal.bh_fedd, bhm1,log_sm)


            Lradio.append(L)
            Lagn.append(L_agn)
            Lsfr.append(get_sf_radio(sfr1))
            sigma_value = gal.velocity_dispersions['stellar'].to_value(units = None, equivalence = None)
            sigma_value = np.log10(sigma_value)
            sigma.append(sigma_value)
            ssfr_value = gal.sfr.to_value(units = None, equivalence = None)/gal.masses['stellar'].to_value(units = None, equivalence = None)



            if ssfr_value == 0:
                ssfr.append(-1*np.inf)

            else:
                ssfr.append(np.log10(ssfr_value))
            

            is_agn.append(check_is_agn(L_agn))
            E_ratio = get_ER(bhm1,L_agn ,sigma_value,check_is_agn(L_agn))
            ER.append(E_ratio)
            is_strong_agn.append(check_is_sagn(L_agn, E_ratio))  

            if fedd_value > 1e-4:
                if L_agn > 0:
                    is_strong_agn_fedd.append(True)
                
                else:
                    is_strong_agn_fedd.append(False)
            
            else:
                is_strong_agn_fedd.append(False)


            if fedd_value > 1e-2:
                if L_agn > 0:
                    fedd_gt_m2.append(True)
                
                else:
                    fedd_gt_m2.append(False)
            
            else:
                fedd_gt_m2.append(False)


    dict = {"sfr" : sfr, 'bhmdot' : bhm_dot, "bhm" : bhm, "fedd" : fedd, "Lr" : Lradio,
    "sigma" : sigma, "sSFR" : ssfr, "mass" : stellar_mass,"gmass":gas_mass,
    "is_agn" : is_agn, "ER" : ER, "is_strong_agn" : is_strong_agn,"is_strong_agn_fedd" : is_strong_agn_fedd,
    "Lagn" : Lagn, "Lsfr" : Lsfr,
    "gas_temp" : gas_temp, "cgm_temp" : cgm_temp,
    "fedd_gt_m2" : fedd_gt_m2, "gal_id": gal_id}

    for key in dict:
        print(key,len(dict[key]))

    df = pd.DataFrame(dict)

    return df


def bin_galaxies(property1,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  if property1 == 'sSFR':
    nbins = [-np.inf,-11.5,-8]

  else:
    nbins = [10,10.5,11.25,12] #[10,10.25,10.5,10.75,11,11.25,11.5,11.75,12]    

  sorted_byprop1 = df.sort_values(by = [property1])
  binned_byprop1 = pd.cut(sorted_byprop1[property1], bins= nbins)
  df[property1 + "_bin"] = binned_byprop1

  return df


def schechter_prop(lamda, prop, alpha_0, beta, lambda_star, property_string):

  """This function is used while integrating the
  model, as it allows us to vary lambda in log units."""

  alpha = alpha_prop(alpha_0, prop, beta, property_string)

  
  p_Xi= np.log(10)*lambda_star*(lamda/lambda_star)**(-alpha+1)*np.exp(-lamda/lambda_star)
   
  return p_Xi


def normalisation_prop(prop, alpha_0, beta,
lambda_star,log_lambda_min, property_string):

  """This is the normalisation constant for the schechter_prop() function."""

  integral = integrate.quad(lambda x: schechter_prop(10**x, prop,alpha_0,beta,lambda_star,
  property_string), log_lambda_min,log_lambda_max)[0]

  return 1/integral

v_norma_prop = np.vectorize(normalisation_prop)


def alpha_prop(alpha_0, prop, beta,property):

    '''This function allows us to parametrise the Schechter function slope alpha as
    a function of various galaxy properties'''


    prop_0 = 10.5

    alpha = alpha_0*(prop/prop_0)**beta


    if type(alpha) == np.ndarray:

        ind_high_alpha = np.where(alpha >8)

        alpha[ind_high_alpha] = 8

        ind_low_alpha = np.where(alpha < -3)
        alpha[ind_low_alpha] = -3


    return alpha


def lies_on_m_bhm_relation(log_bhm,log_m_star):

    alpha = 1.147
    beta = 8.568

    m_star = 10**log_m_star
    m_star = m_star/1e11

    log_m_star = np.log10(m_star)

    log_bhm_max = (alpha + 0.02)*log_m_star + (beta + 0.214)
    log_bhm_min = (alpha - 0.02)*log_m_star + (beta - 0.214)

    if log_bhm_min < log_bhm < log_bhm_max:
        return True
    
    else:
        return False