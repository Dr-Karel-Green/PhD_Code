#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 20:03:36 2025
Code that created Fig. 3.1 from my thesis.
@author: karelgreen
"""
#%% Modules
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from math import exp, pi
from scipy.interpolate import interp1d

"""
Note that the true version of this code actually loops through all galaxies 
I had in my dataset to produce the light curve, likelihood curve, variability
amplitude and error. This way you only needs to run the code once.

"""
#%% Functions
def percent_err(arr, err, new):
    frac = np.divide(err, arr)
    out = np.multiply(frac, new)
    return out

    
def likecurve(xi, avg, erri, sigq):
    num = exp((xi - avg)**2/((-2)*(erri**2+sigq**2)))
    denom = ((2*pi)**0.5)*((erri**2+sigq**2)**0.5)
    
    like = num/denom
    return like  
#%% Data
name = 224511
data = Table.read('/[Redacted]', format='fits')
ID = np.array(data['DR11_ID'])
sems = np.array(['05B', '06B', '07B', '08B', '09B', '10B', '11B', '12B'])
idx = np.where(name==ID)[0][0]

chi_k = data[idx][38]
chi_j = data[idx][36]
    
#%% Accessing row data
row = list(data[idx]) #ID of specific galacy

row_k = row[3:19] #K-band data of the specific galaxy
cols_k = np.array(data.colnames[3:19]) #column names for the galaxy data 
k_mags = np.array(row_k[::2])
k_err = np.array(row_k[1::2])


row_j = row[19:35] #J-band data of the specific galaxy
cols_j = np.array(data.colnames[19:35]) #column names for the galaxy data 
j_mags = np.array(row_j[::2])
j_err = row_j[1::2]


#%% Normalising the data
kavg = np.nanmean((k_mags))
k_n = k_mags/kavg
kn_err = percent_err(k_mags, k_err, k_n)
kn_avg = np.nanmean(k_n) #Redundant as this is equal to 1 but put it in so I don't get confused below why I use the value "1" in the likelihood curve calculation

javg = np.nanmean((j_mags))
j_n = j_mags/javg
jn_err = percent_err(j_mags, j_err, j_n)
jn_avg = np.nanmean(j_n) #Redundant as this is equal to 1 but put it in so I don't get confused below why I use the value "1" in the likelihood curve calculation

#%% Plotting

fig, ax = plt.subplots()

ax.errorbar(sems, j_n, jn_err, color='dodgerblue', ecolor='lightgrey', marker='o', ls='none', label='J-band')
ax.errorbar(sems, k_n, kn_err, color='deeppink', ecolor='lightgrey', marker='o', ls='none', label='K-band')
ax.set(xlabel='Semester', ylabel='Normalised Flux')
ax.set_title(f'ID = {name}', loc='left')
ax.set_title(rf'$\chi^{2}_{{j}}$={chi_j}, $\chi^{2}_{{k}}$={chi_k}', loc='right')
plt.legend()
plt.tight_layout()
plt.savefig(f'{name}_lightcurve.pdf')
plt.show()

#%% Calculating maximum likelyhood + plotting curve    
sigma_rng = np.arange(0, 1.01, 0.01)

l_k = np.array([])
for i in sigma_rng:
    out = np.array([])
    for idx in zip(k_n, kn_err):
        out = np.append(out, likecurve(idx[0], kn_avg, idx[1], i))
    l_k = np.append(l_k, np.nanprod(out))



l_j = np.array([])
for i in sigma_rng:
    out = np.array([])
    for idx in zip(j_n, jn_err):
        out = np.append(out, likecurve(idx[0], jn_avg, idx[1], i))
    l_j = np.append(l_j, np.nanprod(out))
 

intp_k = interp1d(l_k/np.max(l_k), sigma_rng)
intp_j = interp1d(l_j/np.max(l_j), sigma_rng)

fig, ax = plt.subplots()

ax.plot(sigma_rng, l_k/np.max(l_k), color='deeppink', ls='--', label='K-Band')
ax.vlines(intp_k(1), 0, 1, linestyle='dashdot', color='lightgrey') 
ax.vlines(intp_j(1), 0, 1, linestyle='dashdot', color='lightgrey') 
ax.plot(sigma_rng, l_j/np.max(l_j), color='dodgerblue', ls='--', label='J-Band')
ax.set(xlabel=r'$\sigma$', ylabel='L')
ax.set_title(f'ID = {name}', loc='left')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{name}_likelihood.pdf')
plt.show() 