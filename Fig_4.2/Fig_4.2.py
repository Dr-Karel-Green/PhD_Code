#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:11:13 2025
README: Code that compares the mass vs the redshift for the galaxies in this 
study. It is a recreation of the code used to create figure 4.2 in my doctoral
thesis.
@author: karelgreen
"""
#%% Modules
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
from scipy.interpolate import interp1d

#%% Functions
def get_z(tbdata):
    ''' Get z with spectroscopic data where possible 
    Inputs:
        tbdata = table of data for all the objects we want z for (should contain
                  DR11 data in order to get z)
    Outputs:
        z = array of redshifts with spectroscopic redshifts where possible and 
            z_p where spec is not available.
    '''
    # z = tbdata['z_p_2']
    z = tbdata['z_spec_1']
    z[z==-1] = tbdata['z_a_2'][z==-1]
    try:
        mass = tbdata['Mstar_z_p']
    except KeyError:
        mass = tbdata['Mstar_z_p_1']
    mass = np.log10(mass)
    mask = np.logical_and(np.logical_and(z>1, z<=4), mass<8.1)
    z[mask] = tbdata['z_p_1'][mask]
    return z

def remove_bad(arr):
    """Removes dummy values in an array and replaces them with nan"""
    dummy = np.array([float('-inf'), float('inf'), 0])
    mask = np.isin(arr, dummy) 
    arr[mask] = np.nan 
    arr[arr<=0] = np.nan
    return(arr)

#%% Data
"""Created three catalogues prior to this code separating the galaxies out into
xray only, variable only and both x-ray bright and variability detected"""
###Var only###
var = Table.read('Data/var.fits', format='fits') 
cs = np.array(var['chandra_sky']).astype(bool)

IDv = np.array(var['DR11_ID'])

mass_v = np.array(var['Mstar_z_p'])
mass_v = remove_bad(mass_v)
mass_v = np.log10(mass_v)
z_v = get_z(var)

jk = np.array(var['Variable_both'])
j = np.array(var['Variable_jonly'])
k = np.array(var['Variable_konly'])

##X-ray only##
xray = Table.read('Data/xray.fits', format='fits') 

z_x = np.array(xray['z_use'])
mass_x = np.array(xray['Mstar_z_p'])
mass_x = np.log10(mass_x)

##XV##
xrayvar = Table.read('Data/xrayvar.fits', format='fits')
z_xv = np.array(xrayvar['z_use'])
mass_xv = np.array(xrayvar['Mstar_z_p'])
mass_xv = np.log10(mass_xv)

##Regular##
reg = Table.read('Data/regular.fits', format='fits') 

mass_r = np.array(reg['Mstar_z_p'])
mass_r = remove_bad(mass_r)
mass_r = np.log10(mass_r)
z_r = np.array(reg['z_use']) 
z_r = remove_bad(z_r)

##All DR11 - Completeness curve##
z_complcurve = np.load('Data/complecurve_z.npy')
mass_complcurve = np.load('Data/complecurve_mass.npy')
intp = interp1d(z_complcurve, mass_complcurve, bounds_error=False, fill_value='extrapolate') #Interpolating so curve is smooth

#%% Unused but filters to see if given galaxy is above completeness limit or not

filt_x = mass_x >= intp(z_x)
filt_xv = mass_xv >= intp(z_xv)
filt_v = mass_v >= intp(z_v)
filt_r = mass_r >= intp(z_r)


#%% AGN mass plot with completeness curve

fig, ax = plt.subplots()

ax.scatter(z_r, mass_r, fc='none', edgecolor='silver', marker='.', alpha=0.1, s=5, rasterized=True, zorder=1, label='Inactive Galaxy')
ax.scatter(z_v, mass_v, marker='o', s=20, color='red', label='Variable', zorder=2)
ax.scatter(z_x, mass_x, marker='x', s=20, fc='deepskyblue', label='X-ray', zorder=3)
ax.scatter(z_xv, mass_xv, marker='o', fc='darkviolet', label='X-ray&Variable', zorder=3)
ax.plot(z_complcurve, mass_complcurve, color='black', ls='solid', zorder=4)


ax.set(xlim=(-0.25, 5.125), ylim=(7, 12))
ax.set_xlabel('z', fontsize="x-large")
ax.set_ylabel(r'Stellar Mass log($M_{\odot}/M_{*}$)', fontsize="x-large")
plt.legend()
plt.savefig('mass_z.pdf', bbox_inches='tight')
plt.show()