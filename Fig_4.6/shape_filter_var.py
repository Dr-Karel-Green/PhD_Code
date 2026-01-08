#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:42:56 2026

@author: karelgreen
"""
#%% Modules
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.path as pltPath

#%% Functions
def remove_bad(arr):
    """Removes dummy values in an array and replaces them with nan"""
    dummy = np.array([float('-inf'), float('inf'), 0])
    mask = np.isin(arr, dummy) 
    arr[mask] = np.nan 
    arr[arr<=0] = np.nan
    return(arr)

#%% Data
inact = Table.read('Data/reggals_sc.fits', format='fits') 
"""Semreggals catalogued matched on ID in topcat using VWSC0.5-3dr11.fits"""
z_inact = np.array(inact['z_use'])
mask_inact=z_inact<=5 #Only going up to z = 5
inact = inact[mask_inact]

mass_inact = np.log10(remove_bad(np.array(inact['Mstar_z_p'])))
z_inact = np.array(inact['z_use'])

data_reg = Table.read('Data/notvar.fits', format='fits') 
z_reg = np.array(data_reg['z_use'])
mask_reg=z_reg<=5 #Only going up to z = 5
data_reg = data_reg[mask_reg]

mass_reg = np.log10(remove_bad(np.array(data_reg['Mstar_z_p'])))
z_reg = np.array(data_reg['z_use'])

data = Table.read('Data/varonly.fits', format='fits') 
z = np.array(data['z_use'])
mask=z<=5 #Only going up to z = 5
data = data[mask]

mass = np.log10(remove_bad(np.array(data['Mstar_z_p'])))
z = np.array(data['z_use'])

xv = Table.read('Data/xrayvar.fits', format='fits') 
zxv = np.array(xv['z_use'])
maskxv=zxv<=5 #Only going up to z = 5
xv = xv[maskxv]

massxv = np.log10(remove_bad(np.array(xv['Mstar_z_p'])))
zxv = np.array(xv['z_use'])

#%% Filtering by mass completeness
z_complcurve = np.load('Data/complecurve_z.npy')
mass_complcurve = np.load('Data/complecurve_mass.npy')
intp = interp1d(z_complcurve, mass_complcurve, bounds_error=False, fill_value='extrapolate')

inact = inact[intp(z_inact)< mass_inact]
data_reg = data_reg[intp(z_reg)< mass_reg]
data = data[intp(z)< mass]
xv = xv[intp(zxv)< massxv]

#%% Magnitude masks
inact = inact[np.array(inact['JMAG_20'])<=22.66]
data_reg = data_reg[np.array(data_reg['JMAG_20'])<=22.66]
data = data[np.array(data['JMAG_20'])<=22.66]
xv = xv[np.array(xv['JMAG_20'])<=22.66]

#%% Reloading filtered data
mass_inact = np.log10(remove_bad(np.array(inact['Mstar_z_p'])))
z_inact = np.array(inact['z_use'])
Mj_inact = np.array(inact['M_J_z_p'])
shape_inact = np.array(inact['CLASS_STAR'])/np.nanmax(np.array(inact['CLASS_STAR']))

pas = np.array(inact['CLASS'])==1
sf = np.array(inact['CLASS'])==2
psb = np.array(inact['CLASS'])==3
dusty = np.array(inact['CLASS'])==4
zsc = np.array(inact['z_use'])

mass_reg = np.log10(remove_bad(np.array(data_reg['Mstar_z_p'])))
z_reg = np.array(data_reg['z_use'])
Mj_reg = np.array(data_reg['M_J_z_p'])

mass = np.log10(remove_bad(np.array(data['Mstar_z_p'])))
z = np.array(data['z_use'])
Mj = np.array(data['M_J_z_p'])
sigj = np.array(data['sigma j'])/np.nanmax(np.array(data['sigma j']))
shape = np.array(data['CLASS_STAR'])/np.nanmax(np.array(data['CLASS_STAR']))

massxv = np.log10(remove_bad(np.array(xv['Mstar_z_p'])))
zxv = np.array(xv['z_use'])
Mjxv = np.array(xv['M_J_z_p'])
sigjxv = np.array(xv['sigma j'])/np.nanmax(np.array(xv['sigma j']))
shapexv = np.array(xv['CLASS_STAR'])/np.nanmax(np.array(xv['CLASS_STAR']))

#%% shape distribution

fig, ax = plt.subplots()

ax.hist(shape, bins=20, histtype='step', color='red', range=(0,1), density=True, label='Variable only')
ax.hist(shapexv, bins=20, histtype='step', color='mediumorchid', range=(0,1), density=True, label='X-ray and variable')
ax.set(xlabel='SEXtractor Class_Star Value', ylabel='Normalised Distribution')
plt.legend()
plt.show()

#%% Plotting
i=0
n, m = np.array([0.5, 0.75, 1, 1.5, 2, 3]), np.array([0.75, 1, 1.5, 2, 3, 5])
fig = plt.figure(figsize=[11.7, 16.5])
figs = fig.subfigures(3, 2)

for subfig in figs.flatten():
    
    """Filtering by shape to find quasars as well as by redshift bins"""
    mask = np.logical_and(np.logical_and(z>n[i], z<=m[i]), shape<0.9)
    mask_star = np.logical_and(np.logical_and(z>n[i], z<=m[i]), shape>=0.9)
    maskxv = np.logical_and(np.logical_and(zxv>n[i], zxv<=m[i]), shapexv<0.9)
    maskxv_star = np.logical_and(np.logical_and(zxv>n[i], zxv<=m[i]), shapexv>=0.9)
    mask_inact = np.logical_and(z_inact>n[i], z_inact<=m[i])
    
    table = Table(np.column_stack((Mj_inact[mask_inact],mass_inact[mask_inact])))
    dataframe_plot = table.to_pandas()

    ax = subfig.subplots(1)
    
    ax.scatter(Mj_inact[mask_inact], mass_inact[mask_inact], marker='.', s=5, color='lightgrey')
    kde = sns.kdeplot(data = dataframe_plot, x='col0', y='col1', levels=1, thresh=0.05, color='black') #Line that contains 95% of the regular galaxies data
    test = kde.collections[0].get_paths()[0]
    ax.scatter(Mj[mask], mass[mask], marker='o', fc='red', edgecolor='darkred', label='Variable')
    ax.scatter(Mj[mask_star], mass[mask_star], marker='*', fc='red', edgecolor='darkred', linewidth=1)
    ax.scatter(Mjxv[maskxv], massxv[maskxv], marker='o', fc='mediumorchid', edgecolor='darkviolet', label='X-ray & Variable')
    ax.scatter(Mjxv[maskxv_star], massxv[maskxv_star], marker='*', fc='mediumorchid', edgecolor='darkviolet', linewidth=1)
    ax.set(xlabel='J - Band Absolute Magnitude', ylabel=r'Stellar Mass [$log(M_{*}$/$M_{\odot}$)]', xlim=(-27, -19), ylim=(8.5, 11.65), title=fr'{n[i]} $<$ z $\leq$ {m[i]}')
    ax.invert_xaxis()

    if n[i]==0.5:
        ax.legend(loc='upper left')
    
    i+=1

plt.show()

#%%
i=0
n, m = np.array([0.5, 0.75, 1, 1.5, 2, 3]), np.array([0.75, 1, 1.5, 2, 3, 5])
total_var_agn=0#After loop printing this value gives the total number of AGN
total_xv_agn=0#After loop printing this value gives the total number of AGN
var_in_contour=0 #After loop printing this value gives the number of AGN that was in the contour
xv_in_contour=0 #After loop printing this value gives the number of AGN that was in the contour
fig = plt.figure(figsize=[11.7, 16.5])
figs = fig.subfigures(3, 2)

for subfig in figs.flatten():
    
    mask = np.logical_and(np.logical_and(z>n[i], z<=m[i]), shape<0.9)
    mask_star = np.logical_and(np.logical_and(z>n[i], z<=m[i]), shape>=0.9)
    maskxv = np.logical_and(np.logical_and(zxv>n[i], zxv<=m[i]), shapexv<0.9)
    maskxv_star = np.logical_and(np.logical_and(zxv>n[i], zxv<=m[i]), shapexv>=0.9)
    mask_inact = np.logical_and(z_inact>n[i], z_inact<=m[i])
    
    table = Table(np.column_stack((Mj_inact[mask_inact],mass_inact[mask_inact])))
    dataframe_plot = table.to_pandas()

    ax = subfig.subplots(1)
    
    kde = sns.kdeplot(data = dataframe_plot, x='col0', y='col1', levels=1, thresh=0.05, color='black')
    edge = kde.collections[0].get_paths()[0]
    path = pltPath.Path(edge.vertices) #This and following lines below keep the AGN that exist within the contours of the regular galaxies
    
    total_var_agn += np.nansum(mask) + np.nansum(mask_star)
    total_xv_agn += np.nansum(maskxv) + np.nansum(maskxv_star)
    
    filt = path.contains_points(np.column_stack((Mj[mask], mass[mask])))
    filt_star = path.contains_points(np.column_stack((Mj[mask_star], mass[mask_star])))

    filtxv = path.contains_points(np.column_stack((Mjxv[maskxv], massxv[maskxv])))
    filtxv_star = path.contains_points(np.column_stack((Mjxv[maskxv_star], massxv[maskxv_star])))
 
    var_in_contour += np.nansum(filt) + np.nansum(filt_star)
    xv_in_contour += np.nansum(filtxv) + np.nansum(filtxv_star)
    
    ax.scatter(Mj_inact[mask_inact], mass_inact[mask_inact], marker='.', s=5, color='lightgrey')
    ax.scatter(Mj[mask][filt], mass[mask][filt], marker='o', fc='red', edgecolor='darkred', label='Variable')
    ax.scatter(Mj[mask_star][filt_star], mass[mask_star][filt_star], marker='*', fc='red', edgecolor='darkred', linewidth=1)
    ax.scatter(Mjxv[maskxv][filtxv], massxv[maskxv][filtxv], marker='o', fc='mediumorchid', edgecolor='darkviolet', label='X-ray & Variable')
    ax.scatter(Mjxv[maskxv_star][filtxv_star], massxv[maskxv_star][filtxv_star], marker='*', fc='mediumorchid', edgecolor='darkviolet', linewidth=1)
    ax.set(xlabel='J - Band Absolute Magnitude', ylabel=r'Stellar Mass [$log(M_{*}$/$M_{\odot}$)]', xlim=(-27, -19), ylim=(8.5, 11.65), title=fr'{n[i]} $<$ z $\leq$ {m[i]}')
    ax.invert_xaxis()

    if n[i]==0.5:
        ax.legend(loc='upper left')
    
    i+=1
    
#%%
ID = np.array(data['DR11_ID'])
IDxv = np.array(xv['DR11_ID'])

var_keep = np.array([]) #Variable AGN that have all the mass and mag filters but are also within 2-sigma of the regular galaxies
xv_keep = np.array([])

i=0
n, m = np.array([0.5, 0.75, 1, 1.5, 2, 3]), np.array([0.75, 1, 1.5, 2, 3, 5])

fig = plt.figure(figsize=[11.7, 16.5])
figs = fig.subfigures(3, 2)

for subfig in figs.flatten():
    
    mask = np.logical_and(z>n[i], z<=m[i])
    maskxv = np.logical_and(zxv>n[i], zxv<=m[i])
    mask_inact = np.logical_and(z_inact>n[i], z_inact<=m[i])
    
    table = Table(np.column_stack((Mj_inact[mask_inact], mass_inact[mask_inact])))
    dataframe_plot = table.to_pandas()

    ax = subfig.subplots(1)
    
    kde = sns.kdeplot(data = dataframe_plot, x='col0', y='col1', levels=1, thresh=0.05, color='black')
    edge = kde.collections[0].get_paths()[0]
    path = pltPath.Path(edge.vertices)
    
    filt = path.contains_points(np.column_stack((Mj[mask], mass[mask])))
    filtxv = path.contains_points(np.column_stack((Mjxv[maskxv], massxv[maskxv])))
    
    ax.scatter(Mj_inact[mask_inact], mass_inact[mask_inact], marker='.', s=5, color='lightgrey')
    ax.scatter(Mj[mask][filt], mass[mask][filt], marker='o', fc='red', edgecolor='darkred', label='Variable')
    ax.scatter(Mjxv[maskxv][filtxv], massxv[maskxv][filtxv], marker='o', fc='mediumorchid', edgecolor='darkviolet', label='X-ray & Variable')
    ax.set(xlabel='J - Band Absolute Magnitude', ylabel=r'Stellar Mass [$log(M_{*}$/$M_{\odot}$)]', xlim=(-27, -19), ylim=(8.5, 11.65), title=fr'{n[i]} $<$ z $\leq$ {m[i]}')
    ax.invert_xaxis()

    if n[i]==0.5:
        ax.legend(loc='upper left')
    
    bin_filt = np.copy(mask) 
    bin_filt[mask==True] = filt
    ID_keep = ID[bin_filt]
    var_keep = np.append(var_keep, ID_keep)
    
    bin_filt = np.copy(maskxv) 
    bin_filt[maskxv==True] = filtxv
    IDxv_keep = IDxv[bin_filt]
    xv_keep = np.append(xv_keep, IDxv_keep)
    
    i+=1
plt.show()    

#%%
np.save('Data/var_id.npy', var_keep) #The variable AGN that lie within the 2sigma (95%) locus of the regular galaxies
np.save('Data/xv_id.npy', xv_keep) #The xray and variable AGN that lie within the 2sigma locus of the regular galaxies
