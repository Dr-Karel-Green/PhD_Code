#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:21:29 2026
README: code that reproduces the mass function seen in figure 4.6 of my thesis
@author: karelgreen
"""

#%% Modules
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
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
###Xray###
xray_reg = Table.read('[]', format='fits') 
zx_reg = np.array(xray_reg['[]'])
maskx_reg=zx_reg&lt;=5 #Only going up to z = 5
xray_reg = xray_reg[maskx_reg]

cs = np.array(xray_reg['[]'])
xray_reg = xray_reg[cs]

IDx_reg = np.array(xray_reg['[]'])
massx_reg = np.log10(remove_bad(np.array(xray_reg['[]'])))
zx_reg = np.array(xray_reg['[]'])

xray = Table.read('[]', format='fits')
IDx = np.array(xray['[]'])
IDxkeep = np.load('[]')

xfilt = np.array([])

for i in IDxkeep:
    xfilt = np.append(xfilt, np.where(i==IDx)[0][0])

xfilt = xfilt.astype(int)
xray = xray[xfilt] 
###Variable###
var_reg = Table.read('[]', format='fits') 
zv_reg = np.array(var_reg['[]'])
maskv_reg=zv_reg&lt;=5 #Only going up to z = 5
var_reg = var_reg[maskv_reg]

IDv_reg = np.array(var_reg['[]'])
massv_reg = np.log10(remove_bad(np.array(var_reg['[]'])))
zv_reg = np.array(var_reg['[]'])

var = Table.read('[]', format='fits') 
IDv = np.array(var['[]'])
IDvkeep = np.load('[]')

vfilt = np.array([])

for i in IDvkeep:
    vfilt = np.append(vfilt, np.where(i==IDv)[0][0])

vfilt = vfilt.astype(int)
var = var[vfilt]   
#%% Removing quasars from opposing galaxy samples
idx_vinit = np.arange(0, IDv.size, 1)
idx_vbad = np.delete(idx_vinit, vfilt)
IDvbad = IDv[idx_vbad] #ID's of the variable quasars I want gone

xregfilt = np.array([])
for i in IDvbad: #Removing the quasar variable AGN from the xray comp samp
    try:
        xregfilt = np.append(xregfilt, np.where(i==IDx_reg)[0][0])
    except IndexError:
        pass
    
xregfilt = xregfilt.astype(int)
xray_reg.remove_rows(xregfilt)
cs = np.array(xray_reg['[]'])
xray_reg = xray_reg[cs]
IDx_reg = np.array(xray_reg['[]'])
massx_reg = np.array(xray_reg['[]'])
massx_reg = remove_bad(massx_reg)
massx_reg = np.log10(massx_reg)
zx_reg = np.array(xray_reg['[]'])

idx_xinit = np.arange(0, IDx.size, 1)
idx_xbad = np.delete(idx_xinit, xfilt)
IDxbad = IDx[idx_xbad] #ID's of the xray quasars I want gone

vregfilt = np.array([])
for i in IDxbad: #Removing the quasar xray AGN from the var comp samp
    try:
        vregfilt = np.append(vregfilt, np.where(i==IDv_reg)[0][0])
    except IndexError:
        pass
    
vregfilt = vregfilt.astype(int)

var_reg.remove_rows(vregfilt)

IDv_reg = np.array(var_reg['[]'])
massv_reg = np.array(var_reg['[]'])
massv_reg = remove_bad(massv_reg)
massv_reg = np.log10(massv_reg)
zv_reg = np.array(var_reg['[]'])

#%% Filtering by mass completeness
z_complcurve = np.load('[]')
mass_complcurve = np.load('[]')
intp = interp1d(z_complcurve, mass_complcurve, bounds_error=False, fill_value='extrapolate')

xray_reg = xray_reg[intp(zx_reg)&lt; massx_reg]
var_reg = var_reg[intp(zv_reg)&lt; massv_reg]

#%% Reloading filtered data
###Xray###
massx_reg = np.log10(remove_bad(np.array(xray_reg['[]'])))
zx_reg = np.array(xray_reg['[]'])

massx = np.array(xray['[]'])
massx = np.log10(remove_bad(np.array(xray['[]'])))
massx = massx.astype(float)
zx = np.array(xray['[]'])

###Variable###
massv_reg = np.log10(remove_bad(np.array(var_reg['[]'])))
zv_reg = np.array(var_reg['[]'])

massv = np.log10(remove_bad(np.array(var['[]'])))
zv = np.array(var['[]'])

#%% Plotting to check
fig, ax = plt.subplots()
ax.scatter(zx_reg, massx_reg, marker='.', color='lightgrey', label='Not X-ray', rasterized=True, alpha=0.2)
ax.scatter(zx, massx, marker='x', color='deepskyblue', label='X-ray')
ax.plot(z_complcurve, mass_complcurve, color='green')
ax.set(xlabel='z', ylabel='Stellar Mass', xlim=(-0.25, 5), ylim=(7, 11.6))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(zv_reg, massv_reg, marker='.', color='lightgrey', label='Not Variable', rasterized=True, alpha=0.2)
ax.scatter(zv, massv, marker='o', color='red', label='Variable')
ax.plot(z_complcurve, mass_complcurve, color='green')
ax.set(xlabel='z', ylabel='Stellar Mass', xlim=(-0.25, 5), ylim=(7, 11.8))
plt.legend()
plt.show()

#%% Magnitude masks
mjx_reg = np.array(xray_reg['[]'])
jmagxreg_filt = np.array(xray_reg['[]'])&lt;=22.66

mjx = np.array(xray['[]'])
# jmagx_filt = np.array(xray['[]'])&lt;=22.66

mjv_reg = np.array(var_reg['[]'])
jmagvreg_filt = np.array(var_reg['[]'])&lt;=22.66

mjv = np.array(var['[]'])
# jmagv_filt = np.array(var['[]'])&lt;=22.66

#%% Collecting masses
massx_all = np.concatenate((massx_reg, massx))
zx_all = np.concatenate((zx_reg, zx))

massx_all_magfilt = np.concatenate((massx_reg[jmagxreg_filt], massx))
zx_all_magfilt = np.concatenate((zx_reg[jmagxreg_filt], zx))
mjx_all = np.concatenate((mjx_reg, mjx))

massv_all = np.concatenate((massv_reg, massv))
zv_all = np.concatenate((zv_reg, zv))

massv_all_magfilt = np.concatenate((massv_reg[jmagvreg_filt], massv))
zv_all_magfilt = np.concatenate((zv_reg[jmagvreg_filt], zv))
mjv_all = np.concatenate((mjv_reg, mjv))

#%% Plotting
i=0
n, m = np.array([0.5, 0.75, 1, 1.5, 2, 3]), np.array([0.75, 1, 1.5, 2, 3, 5])
bins = 15
rng = (8.5, 12)
xlim = (8.5, 12)
ylim=(10**-8, 10**-3.1)
fig = plt.figure(figsize=[11.7, 16.5])
figs = fig.subfigures(3, 2)

for subfig in figs.flatten():
    maskx = np.logical_and(np.logical_and(zx&gt;n[i], zx&lt;=m[i]), mjx&lt;=22.66)
    maskxall = np.logical_and(zx_all&gt;n[i], zx_all&lt;=m[i])
    maskxallmag  = np.logical_and(np.logical_and(zx_all&gt;n[i], zx_all&lt;=m[i]), mjx_all&lt;=22.66)
    
    normx = (cosmo.comoving_volume(z=m[i]) - cosmo.comoving_volume(z=n[i]))*(0.33/21253)
    """Comoving volume of the universe at that z range.
    Need to divide this by the size of the universe and then multiply by the 
    size of the survey to get the area subtended by the survey in that z range.
    0.33 square degrees is the size of the chandra survey"""

    maskv = np.logical_and(np.logical_and(zv&gt;n[i], zv&lt;=m[i]), mjv&lt;=22.66)
    maskvall = np.logical_and(zv_all&gt;n[i], zv_all&lt;=m[i])
    maskvallmag  = np.logical_and(np.logical_and(zv_all&gt;n[i], zv_all&lt;=m[i]), mjv_all&lt;=22.66)
    
    normv = (cosmo.comoving_volume(z=m[i]) - cosmo.comoving_volume(z=n[i]))*(0.58202/21253)
    """Comoving volume of the universe at that z range.
    Need to divide this by the size of the universe and then multiply by the 
    size of the survey to get the area subtended by the survey in that z range.
    0.33 square degrees is the size of the chandra survey"""
    
    ax = subfig.subplots(2, 1, gridspec_kw={'height_ratios':[10, 3]}, sharex=True)

    ax[0].hist(massv_all[maskvall], bins=bins, color='lightgrey', histtype='step', linewidth=2, range=rng, weights=np.ones(massv_all[maskvall].size)/normv, zorder=1)
    all_galsv = ax[0].hist(massv_all[maskvallmag], bins=bins, color='black', histtype='step', linewidth=2, label=r'$m_{j} \leq$ 22.66', range=rng, weights=np.ones(massv_all[maskvallmag].size)/normv, zorder=1)
    var_agn = ax[0].hist(massv[maskv], bins=bins, color='red', ls='dashed', histtype='step', linewidth=2, label=r'Variable $m_{j} \leq$ 22.66', range=rng, weights=np.ones(massv[maskv].size)/normv, zorder=2)
    
    # ax[0].hist(massx_all[maskxall], bins=bins, color='lightgrey', histtype='step', linewidth=2, range=rng, weights=np.ones(massx_all[maskxall].size)/normx)
    all_galsx = ax[0].hist(massx_all[maskxallmag], bins=bins, linestyle='none', histtype='step', range=rng, weights=np.ones(massx_all[maskxallmag].size)/normx)
    xray_agn = ax[0].hist(massx[maskx], bins=bins, color='deepskyblue', histtype='step', linewidth=2, label=r'Xray $m_{j} \leq$ 22.66', range=rng, weights=np.ones(massx[maskx].size)/normx, zorder=1)
    ax[0].vlines(intp(m[i]), ylim[0], ylim[1]+1, color='green', linestyle='dashed')
    # ax[0].grid(zorder=0)
    ax[0].set(ylabel=r'$\Phi$/$Mpc^{-3}$', yscale='log', xlim=xlim, ylim=ylim, title=fr'{n[i]} $&lt;$ z $\leq$ {m[i]}')
    ax[0].set_yticks([10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3], ['',r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'])

    if n[i]==3.0:
        ax[0].legend(loc='upper left')
    
    pcntv = (var_agn[0]/all_galsv[0])*100
    pcntx = (xray_agn[0]/all_galsx[0])*100
    
    bin_middle = np.array([((var_agn[1][i]+var_agn[1][i+1])/2) for i in range((var_agn[1].size-1))])

    ax[1].fill_between(bin_middle, pcntx, color='deepskyblue', alpha=0.1)
    ax[1].fill_between(bin_middle, pcntv, color='red', alpha=0.1)
    ax[1].plot(bin_middle, pcntx, linewidth=2, color='deepskyblue')
    ax[1].plot(bin_middle, pcntv, linestyle='dashed', linewidth=2, color='red')
    ax[1].set(xlabel=r'Stellar Mass [log($M_{*}$/$M_{\odot}$)]', xlim=xlim,  ylabel='AGN %', yscale='symlog', ylim=(0, 125))
    ax[1].yaxis.set_major_formatter('{x:1.0f}%')
    ax[1].grid()
    i+=1
    
plt.subplots_adjust(hspace=0)
plt.savefig('Fig_4.6.pdf')
plt.show()
