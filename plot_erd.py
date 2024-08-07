import sys
sys.path.append("..")

import yt
import caesar
import matplotlib.pyplot as plt
import numpy as np
import agn_functions as agn
import pandas as pd


df = pd.read_pickle("../ragn_simba_files/data/df.pkl")


df_mass_bins = agn.bin_galaxies("mass", df)
grped_df_mass = df_mass_bins.groupby('mass_bin')

f = 30

locs = [(0,0),(0,2),(0,4),(0,6),(1,0),(1,2),(1,4), (1,6)]
i = 0

alpha_0 = 1.95
beta = -4.99
lambda_star = 0.59
log_lambda_min = -4.68


for name, group in grped_df_mass:

  string = str(name)
  split = string.split(',')
  left = float(split[0][1:])
  right = float(split[1][:-1])
  mass_bin = (left + right)/2


  ax = plt.subplot2grid(shape = (2,8), loc = locs[i], colspan = 2)


  N_bin_sf = 0
  N_bin_q = 0

  inds_sf = []
  inds_q = []

  for j in range(len(group)):

    if group['sSFR'].iloc[j] < -11.5:
      N_bin_q = N_bin_q + 1
      inds_q.append(j)

    if group['sSFR'].iloc[j] > -11.5:
      N_bin_sf = N_bin_sf + 1
      inds_sf.append(j)


  print(N_bin_q)

  group_q = group.iloc[inds_q]
  inds_sagn_q = group_q['is_strong_agn_fedd'].to_numpy(dtype = bool)
  er_agn_q = group_q['fedd'].iloc[inds_sagn_q]

  group_sf = group.iloc[inds_sf]
  inds_sagn_sf = group_sf['is_strong_agn_fedd'].to_numpy(dtype = bool)
  er_agn_sf = group_sf['fedd'].iloc[inds_sagn_sf]

  num_er_q, er_bin_q = np.histogram(np.log10(er_agn_q), bins = 4)
  err_br_q = 0.5/np.sqrt(num_er_q)

  num_er_sf, er_bin_sf = np.histogram(np.log10(er_agn_sf), bins = 4)
  err_br_sf = 0.5/np.sqrt(num_er_sf)

  log_lambdas = np.linspace(-2,1.5,1000)
  pdf_log_lambda = agn.v_norma_prop(mass_bin,alpha_0,beta,lambda_star,
                                      log_lambda_min,"log_Mass")*agn.schechter_prop(
                                          10**log_lambdas,mass_bin,alpha_0,beta,lambda_star,"log_Mass")

    
  if i == 0 or i == 4:
    plt.ylabel(r"$\log\left(\rm P_{\log \left(\lambda\right)}\right)$", fontsize = 0.5*f)
    plt.yticks([-1,-2,-3,-4],fontsize = 0.5*f)
  
  else:
    plt.yticks([])

  if i > 3:
    plt.xlabel(r"$\log\left(\lambda\right)$", fontsize = 0.5*f)
    plt.xticks([-2,-1.5,-1,-0.5,0], fontsize = 0.5*f)

  else:
    plt.xticks([])
      

  plt.annotate(r"$M_{\star} \sim ~$"+ str(mass_bin), (-1.4, -2.5), fontsize = 0.5*f)

  plt.xlim(-2.75,-0.5)
  plt.ylim(-5.25,0)
  
  lambda_bin_width = er_bin_q[1] - er_bin_q[0]

  plt.plot(log_lambdas,np.log10(pdf_log_lambda),  linewidth = 4,c= 'k')
  ax.scatter(er_bin_q[1:], np.log10(num_er_q/(N_bin_q*lambda_bin_width)),
            label = "Q",c = 'r')
  ax.errorbar(er_bin_q[1:], np.log10(num_er_q/(N_bin_q*lambda_bin_width)), yerr = err_br_q, c = 'r')


  ax.scatter(er_bin_sf[1:], np.log10(num_er_sf/(N_bin_sf*lambda_bin_width)),
            label = "SFG",c = 'b')
  ax.errorbar(er_bin_sf[1:], np.log10(num_er_sf/(N_bin_sf*lambda_bin_width)), yerr = err_br_sf, c = 'b')

  plt.legend(loc = 'lower left',prop = {"size": 10})

  i = i+1


plt.plot(log_lambdas,np.log10(pdf_log_lambda), linewidth = 4,  c= 'k', label = "Manga radio ERD fit")
plt.tight_layout()
plt.legend(loc = 'lower left',prop = {"size": 10})
plt.show()

