
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:55:37 2026

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
    arr[arr <= 0] = np.nan
    return arr

#%% Data
### Xray ###
xray_reg = Table.read('[]', format='fits')
zx_reg = np.array(xray_reg['[]'])
maskx_reg = zx_reg <= 5  # Only going up to z = 5
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
    xfilt = np.append(xfilt, np.where(i == IDx)[0][0])

xfilt = xfilt.astype(int)
xray = xray[xfilt]

### Variable ###
var_reg = Table.read('[]', format='fits')
zv_reg = np.array(var_reg['[]'])
maskv_reg = zv_reg <= 5  # Only going up to z = 5
var_reg = var_reg[maskv_reg]

IDv_reg = np.array(var_reg['[]'])
massv_reg = np.log10(remove_bad(np.array(var_reg['[]'])))
zv_reg = np.array(var_reg['[]'])

var = Table.read('[]', format='fits')
IDv = np.array(var['[]'])
IDvkeep = np.load('[]')

vfilt = np.array([])

for i in IDvkeep:
    vfilt = np.append(vfilt, np.where(i == IDv)[0][0])

vfilt = vfilt.astype(int)
var = var[vfilt]

#%% Removing quasars from opposing galaxy samples
idx_vinit = np.arange(0, IDv.size, 1)
idx_vbad = np.delete(idx_vinit, vfilt)
IDvbad = IDv[idx_vbad]  # ID's of the variable quasars I want gone

xregfilt = np.array([])
for i in IDvbad:  # Removing the quasar variable AGN from the xray comp samp
    try:
        xregfilt = np.append(xregfilt, np.where(i == IDx_reg)[0][0])
    except IndexError:
        pass

xregfilt = xregfilt.astype(int)
xray_reg.remove_rows(xregfilt)
cs = np.array(xray_reg['[]'])
xray_reg = xray_reg[cs]
IDx_reg = np.array(xray_reg['[]'])
massx_reg = np.log10(remove_bad(np.array(xray_reg['[]'])))
zx_reg = np.array(xray_reg['[]'])

idx_xinit = np.arange(0, IDx.size, 1)
idx_xbad = np.delete(idx_xinit, xfilt)
IDxbad = IDx[idx_xbad]  # ID's of the xray quasars I want gone

vregfilt = np.array([])
for i in IDxbad:  # Removing the quasar xray AGN from the var comp samp
    try:
        vregfilt = np.append(vregfilt, np.where(i == IDv_reg)[0][0])
    except IndexError:
        pass

vregfilt = vregfilt.astype(int)

var_reg.remove_rows(vregfilt)

IDv_reg = np.array(var_reg['[]'])
massv_reg = np.log10(remove_bad(np.array(var_reg['[]'])))
zv_reg = np.array(var_reg['[]'])

#%% Filtering by mass completeness
z_complcurve = np.load('[]')
mass_complcurve = np.load('[]')
intp = interp1d(z_complcurve, mass_complcurve, bounds_error=False, fill_value='extrapolate')

xray_reg = xray_reg[intp(zx_reg) < massx_reg]
var_reg = var_reg[intp(zv_reg) < massv_reg]

#%% Reloading filtered data
### Xray ###
massx_reg = np.log10(remove_bad(np.array(xray_reg['[]'])))
zx_reg = np.array(xray_reg['[]'])

massx = np.log10(remove_bad(np.array(xray['[]'])))
zx = np.array(xray['[]'])

### Variable ###
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
jmagxreg_filt = np.array(xray_reg['[]']) <= 22.66

mjx = np.array(xray['[]'])
# jmagx_filt = np.array(xray['[]']) <= 22.66

mjv_reg = np.array(var_reg['[]'])
jmagvreg_filt = np.array(var_reg['[]']) <= 22.66

mjv = np.array(var['[]'])
# jmagv_filt = np.array(var['[]']) <= 22.66

#%% Absolute magnitudes
Mjx_reg = np.array(xray_reg['[]'])
Mjx = np.array(xray['[]'])

Mjv_reg = np.array(var_reg['[]'])
Mjv = np.array(var['[]'])

MJx_all = np.concatenate((Mjx_reg, Mjx))
mjx_all = np.concatenate((mjx_reg, mjx))
zx_all = np.concatenate((zx_reg, zx))

MJv_all = np.concatenate((Mjv_reg, Mjv))
mjv_all = np.concatenate((mjv_reg, mjv))
zv_all = np.concatenate((zv_reg, zv))

MJx_all_magfilt = np.concatenate((Mjx_reg[jmagxreg_filt], Mjx))
zx_all_magfilt = np.concatenate((zx_reg[jmagxreg_filt], zx))

MJv_all_magfilt = np.concatenate((Mjv_reg[jmagvreg_filt], Mjv))
zv_all_magfilt = np.concatenate((zv_reg[jmagvreg_filt], zv))

#%% Saving data
save_path = '[]'

np.save(save_path + '/[]', zx)
np.save(save_path + '/[]', mjx)
np.save(save_path + '/[]', Mjx)
np.save(save_path + '/[]', zx_all)
np.save(save_path + '/[]', mjx_all)
np.save(save_path + '/[]', MJx_all)

np.save(save_path + '/[]', zv)
np.save(save_path + '/[]', mjv)
np.save(save_path + '/[]', Mjv)
np.save(save_path + '/[]', zv_all)
np.save(save_path + '/[]', mjv_all)
np.save(save_path + '/[]', MJv_all)
