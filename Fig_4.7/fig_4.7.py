
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:01:21 2026

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
from tqdm import tqdm

#%% Functions
def remove_bad(arr):
    """Removes dummy values in an array and replaces them with nan"""
    dummy = np.array([float('-inf'), float('inf'), 0])
    mask = np.isin(arr, dummy) 
    arr[mask] = np.nan 
    arr[arr <= 0] = np.nan
    return arr

#%% quasar data
zxq = np.load('[]')
mjxq = np.load('[]')
zx_allq = np.load('[]')
mjx_allq = np.load('[]')
Mjxq = np.load('[]')
MJx_allq = np.load('[]')

zvq = np.load('[]')
mjvq = np.load('[]')
Mjvq = np.load('[]')
zv_allq = np.load('[]')
mjv_allq = np.load('[]')    
MJv_allq = np.load('[]')

#%% Not quasar data
zx = np.load('[]')
mjx = np.load('[]')
zx_all = np.load('[]')
mjx_all = np.load('[]')
Mjx = np.load('[]')
MJx_all = np.load('[]')

zv = np.load('[]')
mjv = np.load('[]')
Mjv = np.load('[]')
zv_all = np.load('[]')
mjv_all = np.load('[]')  
MJv_all = np.load('[]')

#%% Plotting
i = 0
n, m = np.array([0.5, 0.75, 1, 1.5, 2, 3]), np.array([0.75, 1, 1.5, 2, 3, 5])
bins = 15
rng = (-27.5, -16)
xlim = (-27.5, -19)
ylim = (10**-8, 10**-2.35)
fig = plt.figure(figsize=[11.7, 16.5])
figs = fig.subfigures(3, 2)

for subfig in figs.flatten():
    maskx = np.logical_and(np.logical_and(zx > n[i], zx <= m[i]), mjx <= 22.66)
    maskxq = np.logical_and(np.logical_and(zxq > n[i], zxq <= m[i]), mjxq <= 22.66)
    maskxall = np.logical_and(zx_all > n[i], zx_all <= m[i])
    maskxallmag = np.logical_and(np.logical_and(zx_allq > n[i], zx_allq <= m[i]), mjx_allq <= 22.66)
    
    normx = (cosmo.comoving_volume(z=m[i]) - cosmo.comoving_volume(z=n[i]))*(0.33/21253)
    """Comoving volume of the universe at that z range.
    Need to divide this by the size of the universe and then multiply by the 
    size of the survey to get the area subtended by the survey in that z range.
    0.33 square degrees is the size of the chandra survey"""

    maskv = np.logical_and(np.logical_and(zv > n[i], zv <= m[i]), mjv <= 22.66)
    maskvq = np.logical_and(np.logical_and(zvq > n[i], zvq <= m[i]), mjvq <= 22.66)
    maskvall = np.logical_and(zv_allq > n[i], zv_allq <= m[i])
    maskvallmag = np.logical_and(np.logical_and(zv_allq > n[i], zv_allq <= m[i]), mjv_allq <= 22.66)
    
    normv = (cosmo.comoving_volume(z=m[i]) - cosmo.comoving_volume(z=n[i]))*(0.58202/21253)
    """Comoving volume of the universe at that z range.
    Need to divide this by the size of the universe and then multiply by the 
    size of the survey to get the area subtended by the survey in that z range.
    0.33 square degrees is the size of the chandra survey"""
    
    ax = subfig.subplots(2, 1, gridspec_kw={'height_ratios':[10, 3]}, sharex=True)

    ax[0].hist(MJv_allq[maskvall], bins=bins, color='lightgrey', histtype='step', linewidth=2, range=rng, weights=np.ones(MJv_allq[maskvall].size)/normv, zorder=1)
    all_galsv = ax[0].hist(MJv_allq[maskvallmag], bins=bins, color='black', histtype='step', linewidth=2, label=r'$m_{j} \leq$ 22.66', range=rng, weights=np.ones(MJv_allq[maskvallmag].size)/normv, zorder=1)
    var_q = ax[0].hist(Mjvq[maskvq], bins=bins, color='red', ls='solid', histtype='step', linewidth=2, range=rng, weights=np.ones(Mjvq[maskvq].size)/normv, zorder=3)
    var_agn = ax[0].hist(Mjv[maskv], bins=bins, color='lightcoral', ls='dashed', histtype='step', linewidth=2, label=r'Variable $m_{j} \leq$ 22.66', range=rng, weights=np.ones(Mjv[maskv].size)/normv, zorder=2)
    
    # ax[0].hist(MJx_all[maskxall], bins=bins, color='lightgrey', histtype='step', linewidth=2, range=rng, weights=np.ones(MJx_all[maskxall].size)/normx)
    all_galsx = ax[0].hist(MJx_allq[maskxallmag], bins=bins, linestyle='none', histtype='step', range=rng, weights=np.ones(MJx_allq[maskxallmag].size)/normx, zorder=1)
    xray_q = ax[0].hist(Mjxq[maskxq], bins=bins, color='deepskyblue', ls='solid', histtype='step', linewidth=2, range=rng, weights=np.ones(Mjxq[maskxq].size)/normx, zorder=3)
    xray_agn = ax[0].hist(Mjx[maskx], bins=bins, color='lightskyblue', ls='dashed', histtype='step', linewidth=2, label=r'Xray $m_{j} \leq$ 22.66', range=rng, weights=np.ones(Mjx[maskx].size)/normx, zorder=2)
    
    # ax[0].grid(zorder=0)
    ax[0].set(ylabel=r'$\Phi$/$Mpc^{-3}$', yscale='log', xlim=xlim, ylim=ylim, title=fr'{n[i]} $&lt;$ z $\leq$ {m[i]}')
    ax[0].set_yticks([10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3], ['',r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'])
    ax[0].invert_xaxis()

    if n[i] == 0.75:
        ax[0].legend(loc='upper right')
    
    pcntv = (var_agn[0]/all_galsv[0])*100
    pcntx = (xray_agn[0]/all_galsx[0])*100    
    
    pcntvq = (var_q[0]/all_galsv[0])*100
    pcntxq = (xray_q[0]/all_galsx[0])*100
    
    bin_middle = np.array([((var_agn[1][i]+var_agn[1][i+1])/2) for i in range((var_agn[1].size-1))])

    ax[1].fill_between(bin_middle, pcntx, color='lightskyblue', alpha=0.1)
    ax[1].fill_between(bin_middle, pcntv, color='lightcoral', alpha=0.1)
    ax[1].plot(bin_middle, pcntx, linestyle='dashed', linewidth=2, color='lightskyblue')
    ax[1].plot(bin_middle, pcntv, linestyle='dashed', linewidth=2, color='lightcoral')
    
    ax[1].fill_between(bin_middle, pcntxq, color='deepskyblue', alpha=0.1)
    ax[1].fill_between(bin_middle, pcntvq, color='red', alpha=0.1)
    ax[1].plot(bin_middle, pcntxq, linestyle='solid', linewidth=2, color='deepskyblue')
    ax[1].plot(bin_middle, pcntvq, linestyle='solid', linewidth=2, color='red')
    
    ax[1].set(xlabel='J - band Absolute Magnitude', xlim=xlim,  ylabel='AGN %', yscale='symlog', ylim=(0, 125))
    ax[1].yaxis.set_major_formatter('{x:1.0f}%')
    ax[1].grid()
    ax[1].invert_xaxis()
    i += 1
    
plt.subplots_adjust(hspace=0)
plt.savefig('fig_4.7.pdf')
plt.show()
