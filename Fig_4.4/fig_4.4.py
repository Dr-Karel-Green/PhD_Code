#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 11:39:40 2025
README: Code that reacreates the stellar mass- magnitude distribution seen in
figure 4.4 of my thesis
@author: karelgreen
"""
#%% Modules
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%% Functions
def remove_bad(arr):
    """Removes dummy values in an array and replaces them with nan"""
    dummy = np.array([float('-inf'), float('inf'), 0])
    mask = np.isin(arr, dummy) 
    arr[mask] = np.nan 
    arr[arr<=0] = np.nan
    return(arr)

#%% Data
###Regular###
regular = Table.read('Data/regular_sc.fits', format='fits') 
z_reg = np.array(regular['z_use'])
mask_reg=z_reg<=5 #Only going up to z = 5

reg = regular[mask_reg]
massreg = np.log10(remove_bad(np.array(reg['Mstar_z_p'])))
zreg = np.array(reg['z_use'])

###variable###
var = Table.read('data/var.fits', format='fits') 
zv = np.array(var['z_use'])
maskv=zv<=5 #Only going up to z = 5
var = var[maskv]

massv = np.log10(remove_bad(np.array(var['Mstar_z_p'])))
zv = np.array(var['z_use'])

###X-ray and Variable###
xv = Table.read('data/xrayvar.fits', format='fits') 
zxv = np.array(xv['z_use'])
maskxv=zxv<=5 #Only going up to z = 5

xv = xv[maskxv]
massxv = np.log10(remove_bad(np.array(xv['Mstar_z_p'])))
zxv = np.array(xv['z_use'])

###xray###
xray = Table.read('data/xray.fits', format='fits')
zx = np.array(xray['z_use'])
maskx=zx<=5 #Only going up to z = 5

xray = xray[maskx]
massx = np.log10(remove_bad(np.array(xray['Mstar_z_p'])))
zx = np.array(xray['z_use'])

#%% Making mass completeness filter
z_complcurve = np.load('[redacted]/Data/complecurve_z.npy')
mass_complcurve = np.load('[redacted]/Data/complecurve_mass.npy')
intp = interp1d(z_complcurve, mass_complcurve, bounds_error=False, fill_value='extrapolate')
       
filt_x = massx >= intp(zx)
filt_xv = massxv >= intp(zxv)
filt_v = massv >= intp(zv)
filt_r = massreg >= intp(zreg)

#%% Magnitude filter
jmagreg_filt = np.array(reg['JMAG_20'])<=22.66
jmagv_filt = np.array(var['JMAG_20'])<=22.66
jmagx_filt = np.array(xray['JMAG_20'])<=22.66
jmagxv_filt = np.array(xv['JMAG_20'])<=22.66

#%% Applying filters
reg = reg[np.logical_and(filt_r, jmagreg_filt)]
xray = xray[np.logical_and(filt_x, jmagx_filt)]
var= var[np.logical_and(filt_v, jmagv_filt)]
xv = xv[np.logical_and(filt_xv, jmagxv_filt)]

#%% Reloading data 
###Regular###
massreg = np.log10(np.array(reg['Mstar_z_p']))
zreg = np.array(reg['z_use'])
Mj_reg = np.array(reg['M_J_z_p'])
shape_reg = np.array(reg['CLASS_STAR'])
shape_reg = shape_reg/np.nanmax(shape_reg) #normalising the shape quantity

###variable###
massv = np.log10(np.array(var['Mstar_z_p']))
zv = np.array(var['z_use'])
Mjv = np.array(var['M_J_z_p'])
sigjv = np.array(var['sigma j'])
sigjv = sigjv/np.nanmax(sigjv) #normalising the variability parameter
shapev = np.array(var['CLASS_STAR'])
shapev = shapev/np.nanmax(shapev) #normalising the shape quantity

###X-ray and Variable###
massxv = np.log10(np.array(xv['Mstar_z_p']))
zxv = np.array(xv['z_use'])
Mjx = np.array(xray['M_J_z_p'])
sigjx = np.array(xray['sigma j'])
sigjx = sigjx/np.nanmax(sigjx) #normalising the variability parameter
shapex = np.array(xray['CLASS_STAR'])
shapex = shapex/np.nanmax(shapex) #normalising the shape quantity

###xray###
massx = np.log10(np.array(xray['Mstar_z_p']))
zx = np.array(xray['z_use'])
Mjxv = np.array(xv['M_J_z_p'])
sigjxv = np.array(xv['sigma j'])
sigjxv = sigjxv/np.nanmax(sigjxv) #normalising the variability parameter
shapexv = np.array(xv['CLASS_STAR'])
shapexv = shapexv/np.nanmax(shapexv) #normalising the shape quantity

#%% Plotting
i=0

#redshift bins, corresponds to approximately equal lookback times of the universe
n, m = np.array([0.5, 0.75, 1, 1.5, 2, 3]), np.array([0.75, 1, 1.5, 2, 3, 5])

fig = plt.figure(figsize=[11.7, 16.5]) #figure size in inches, width by height
figs = fig.subfigures(3, 2) #three rows, two columns

for subfig in figs.flatten():
    
    maskv = np.logical_and(np.logical_and(zv>n[i], zv<=m[i]), shapev<0.9)
    maskx = np.logical_and(np.logical_and(zx>n[i], zx<=m[i]), shapex<0.9)
    maskxv = np.logical_and(np.logical_and(zxv>n[i], zxv<=m[i]), shapexv<0.9)

    maskv_star = np.logical_and(np.logical_and(zv>n[i], zv<=m[i]), shapev>=0.9)
    maskx_star = np.logical_and(np.logical_and(zx>n[i], zx<=m[i]), shapex>=0.9)
    maskxv_star = np.logical_and(np.logical_and(zxv>n[i], zxv<=m[i]), shapexv>=0.9)
    
    mask_reg = np.logical_and(zreg>n[i], zreg<=m[i])
    
    ax = subfig.subplots(1)
    
    #background population
    ax.scatter(Mj_reg[mask_reg], massreg[mask_reg], marker='.', s=5, color='lightgrey')
    
    #agn population
    ax.scatter(Mjv[maskv], massv[maskv], marker='o', fc='red', edgecolor='darkred', label='Variable')
    ax.scatter(Mjx[maskx], massx[maskx], marker='o', fc='deepskyblue', edgecolor='navy', label='X-ray')
    ax.scatter(Mjxv[maskxv], massxv[maskxv], marker='o', fc='mediumorchid', edgecolor='darkviolet', label='X-ray & Variable')

    #likely quasars
    ax.scatter(Mjv[maskv_star], massv[maskv_star], marker='*', fc='red', edgecolor='darkred', linewidth=1)
    ax.scatter(Mjx[maskx_star], massx[maskx_star], marker='*', fc='deepskyblue', edgecolor='navy', linewidth=1)
    ax.scatter(Mjxv[maskxv_star], massxv[maskxv_star], marker='*', fc='mediumorchid', edgecolor='darkviolet', linewidth=1)
    
    ax.set(xlabel='J - Band Absolute Magnitude', ylabel=r'Stellar Mass [$log(M_{*}$/$M_{\odot}$)]', xlim=(-27, -19), ylim=(8.5, 11.65), title=fr'{n[i]} $<$ z $\leq$ {m[i]}')
    ax.invert_xaxis()

    if n[i]==0.5:
        ax.legend(loc='upper left')
    
    i+=1

plt.savefig('fig_4.4.pdf')    
plt.show()
