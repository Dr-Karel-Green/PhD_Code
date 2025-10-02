#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:36:33 2025
README: Code that recreates the plot figure 4.3 in my doctoral thesis
@author: karelgreen
"""
#%% Modules
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
#%% Functions
def remove_bad(arr):
    """Removes dummy values in an array and replaces them with nan"""
    dummy = np.array([float('-inf'), float('inf'), 0])
    mask = np.isin(arr, dummy) 
    arr[mask] = np.nan 
    return(arr)

#%% Data

"""Created three catalogues prior to this code separating the galaxies out into
xray only, variable only and both x-ray bright and variability detected"""
###Var only###
var = Table.read('Data/var.fits', format='fits') 
mass_v = np.log10(remove_bad(np.array(var['Mstar_z_p'])))

##X-ray only##
xray = Table.read('Data/xray.fits', format='fits') 
mass_x = np.log10(remove_bad(np.array(xray['Mstar_z_p'])))

##XV##
xrayvar = Table.read('Data/xrayvar.fits', format='fits')
mass_xv = np.log10(remove_bad(np.array(xrayvar['Mstar_z_p'])))

#%% Plotting figure
rng = [7,12.25] #range the x-axis spans
bns = 20 #number of bins used 
fig, ax = plt.subplots()

ax.hist(mass_x, bins=bns, color='deepskyblue', histtype='step', ls='solid', label='X-ray', density=True, linewidth=2, range=rng)
ax.hist(mass_xv, bins=bns, color='darkviolet', ls='dashdot', histtype='step', label='X-ray&Variable', density=True, linewidth=2, range=rng)
ax.hist(mass_v, bins=bns, color='red', ls='dashed', histtype='step', label='Variable', density=True, linewidth=2, range=rng)

ax.set_ylabel('Normalised Distribution', fontsize="x-large")
ax.set_xlabel(r'Stellar Mass log($M_{*}/M_{\odot}$)', fontsize="x-large")

plt.legend()
plt.savefig('fig_4.3.pdf', bbox_inches='tight')
plt.show()

#%% P-values for distribution comparison in text
x_v = (ks_2samp(mass_x, mass_v))[1]
x_xv = (ks_2samp(mass_x, mass_xv))[1]
v_xv = (ks_2samp(mass_v, mass_xv))[1]

print('P-Value (Xray, Var):', x_v)
print('P-Value (X, xray_var):', x_xv)
print('P-Value (Var, xray_var):', v_xv)

