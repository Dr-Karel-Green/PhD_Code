# -*- coding: utf-8 -*-
"""
README: Code that reproduces figure 4.5 in my doctoral thesis
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

## Inactive
inact = Table.read('[].fits', format='fits') 
z_inact = np.array(inact['z_use'])
mask_inact=z_inact<=5 #Only going up to z = 5

inact = inact[mask_inact]
mass_inact = np.array(inact['[]'])
mass_inact = np.log10(remove_bad(mass_inact))
z_inact = np.array(inact['z_use'])

##variable
var = Table.read('[].fits', format='fits')  
zv = np.array(var['[]'])
maskv=zv<=5 #Only going up to z = 5
var = var[maskv]

massv = np.array(var['[]'])
massv = np.log10(remove_bad(massv))
zv = np.array(var['[]'])

## X-ray and Variable
xv = Table.read('[].fits', format='fits') 
zxv = np.array(xv['[]'])
maskxv=zxv<=5 #Only going up to z = 5

xv = xv[maskxv]
massxv = np.array(xv['[]'])
massxv = np.log10(remove_bad(massxv))
zxv = np.array(xv['z_use_1'])

##xray
xray = Table.read('[].fits', format='fits') 
zx = np.array(xray['[]'])
maskx=zx<=5 #Only going up to z = 5

xray = xray[maskx]
massx = np.array(xray['[]'])
massx = np.log10(remove_bad(massx))
massx = massx.astype(float)
zx = np.array(xray['[]'])


#%% Filtering by mass completeness
z_complcurve = np.load('[]z.npy')
mass_complcurve = np.load('[]npy')
intp = interp1d(z_complcurve, mass_complcurve, bounds_error=False, fill_value='extrapolate')

filt_inact = intp(z_inact) < mass_inact
filtx = intp(zx) < massx
filtv = intp(zv) < massv
filtxv = intp(zxv) < massxv

inact = inact[filt_inact]
xray = xray[filtx]
var = var[filtv]
xv = xv[filtxv]

#%% Magnitude masks - J-band magnitude used
inact = inact[np.array(inact['[]'])<=22.66]
var = var[np.array(var['[]'])<=22.66]
xray = xray[np.array(xray['[]'])<=22.66]
xv = xv[np.array(xv['[]'])<=22.66]

#%% Reloading filtered data
##Inact
mass_inact = np.array(inact['[]'])
mass_inact = np.log10(remove_bad(mass_inact))
z_inact = np.array(inact['[]'])
Mj_inact = np.array(inact['[]'])
shape_inact = np.array(inact['[]'])
shape_inact = shape_inact/np.nanmax(shape_inact)
zsc = np.array(inact['[]'])

##Var
massv = np.array(var['[]'])
massv = np.log10(remove_bad(massv))
zv = np.array(var['[]'])
Mjv = np.array(var['[]'])
sigjv = np.array(var['[] j'])
sigjv = sigjv/np.nanmax(sigjv)
shapev = np.array(var['[]'])
shapev = shapev/np.nanmax(shapev)

##xray
massx = np.array(xray['[]'])
massx = np.log10(remove_bad(massx))
zx = np.array(xray['[]'])
Mjx = np.array(xray['[]'])
sigjx = np.array(xray['[]'])
sigjx = sigjx/np.nanmax(sigjx)
shapex = np.array(xray['[]'])
shapex = shapex/np.nanmax(shapex)

##XV
massxv = np.array(xv['[]'])
massxv = np.log10(remove_bad(massxv))
zxv = np.array(xv['[]'])
Mjxv = np.array(xv['[]'])
sigjxv = np.array(xv['[]'])
sigjxv = sigjxv/np.nanmax(sigjxv)
shapexv = np.array(xv['[]'])
shapexv = shapexv/np.nanmax(shapexv)

#%% Plot for redshift bin 1 < z < 1.5

#Possible redshift bins, corresponds to approximately equal lookback times of the universe
#n, m = np.array([0.5, 0.75, 1, 1.5, 2, 3]), np.array([0.75, 1, 1.5, 2, 3, 5])


# n, m = 1, 1.5
n, m = 1, 1.5
fig, ax = plt.subplots(2, 1, figsize=[4.8, 6.4*1.5])

maskv = np.logical_and(np.logical_and(zv>n, zv<=m), shapev<0.9)
maskx = np.logical_and(np.logical_and(zx>n, zx<=m), shapex<0.9)
maskxv = np.logical_and(np.logical_and(zxv>n, zxv<=m), shapexv<0.9)

maskv_star = np.logical_and(np.logical_and(zv>n, zv<=m), shapev>=0.9)
maskx_star = np.logical_and(np.logical_and(zx>n, zx<=m), shapex>=0.9)
maskxv_star = np.logical_and(np.logical_and(zxv>n, zxv<=m), shapexv>=0.9)
mask_inact = np.logical_and(z_inact>n, z_inact<=m)

ax[0].scatter(Mj_inact[mask_inact], mass_inact[mask_inact], marker='.', s=5, color='lightgrey')
ax[0].scatter(Mjv[maskv], massv[maskv], marker='o', fc='red', edgecolor='darkred', label='Variable')
ax[0].scatter(Mjx[maskx], massx[maskx], marker='o', fc='deepskyblue', edgecolor='navy', label='X-ray')
ax[0].scatter(Mjxv[maskxv], massxv[maskxv], marker='o', fc='mediumorchid', edgecolor='darkviolet', label='X-ray & Variable')

ax[0].scatter(Mjv[maskv_star], massv[maskv_star], marker='*', fc='red', edgecolor='darkred', linewidth=1)
ax[0].scatter(Mjx[maskx_star], massx[maskx_star], marker='*', fc='deepskyblue', edgecolor='navy', linewidth=1)
ax[0].scatter(Mjxv[maskxv_star], massxv[maskxv_star], marker='*', fc='mediumorchid', edgecolor='darkviolet', linewidth=1)

ax[0].set(xlim=(-26, -20), ylim=(9, 11.65), title=fr'{n} $<$ z $\leq$ {m}')
ax[0].invert_xaxis()

maskv = np.logical_and(np.logical_and(np.logical_and(zv>n, zv<=m), shapev<0.9), sigjv<=0.1)
maskx = np.logical_and(np.logical_and(np.logical_and(zx>n, zx<=m), shapex<0.9), sigjx<=0.1)
maskxv = np.logical_and(np.logical_and(np.logical_and(zxv>n, zxv<=m), shapexv<0.9), sigjxv<=0.1)

maskv_star = np.logical_and(np.logical_and(np.logical_and(zv>n, zv<=m), shapev>=0.9), sigjv<=0.1)
maskx_star = np.logical_and(np.logical_and(np.logical_and(zx>n, zx<=m), shapex>=0.9), sigjx<=0.1)
maskxv_star = np.logical_and(np.logical_and(np.logical_and(zxv>n, zxv<=m), shapexv>=0.9), sigjxv<=0.1)

ax[1].scatter(Mj_inact[mask_inact], mass_inact[mask_inact], marker='.', s=5, color='lightgrey')
ax[1].scatter(Mjv[maskv], massv[maskv], marker='o', fc='red', edgecolor='darkred', label='Variable')
ax[1].scatter(Mjx[maskx], massx[maskx], marker='o', fc='deepskyblue', edgecolor='navy', label='X-ray')
ax[1].scatter(Mjxv[maskxv], massxv[maskxv], marker='o', fc='mediumorchid', edgecolor='darkviolet', label='X-ray & Variable')

ax[1].scatter(Mjv[maskv_star], massv[maskv_star], marker='*', fc='red', edgecolor='darkred', linewidth=1)
ax[1].scatter(Mjx[maskx_star], massx[maskx_star], marker='*', fc='deepskyblue', edgecolor='navy', linewidth=1)
ax[1].scatter(Mjxv[maskxv_star], massxv[maskxv_star], marker='*', fc='mediumorchid', edgecolor='darkviolet', linewidth=1)

ax[1].set(xlim=(-26, -20), ylim=(9, 11.65), title=fr'{n} $<$ z $\leq$ {m}')
ax[1].set_title(r'$\sigma \leq 0.1$', loc='left')
ax[1].set_xlabel('J - Band Absolute Magnitude', fontsize="x-large")
ax[1].invert_xaxis()

fig.supylabel(r'Stellar Mass [$log(M_{*}$/$M_{\odot}$)]', x=-0.01, fontsize="x-large")
plt.savefig('Fig_4.5.pdf')
plt.show()
