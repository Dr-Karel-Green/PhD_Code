#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:32:25 2025
README: Code that makes the completeness curve used in figure 4.2 of my thesis
@author: karelgreen
"""

#%% Modules
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
from tqdm import tqdm

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
    dummy = np.array([float('-inf'), float('inf'), 0, 99, -99])
    mask = np.isin(arr, dummy) 
    arr[mask] = np.nan 
    arr[arr<=0] = np.nan
    return(arr)

def limiting_mass(mass, mag, limiting_mag):
    lim_mass = mass + (0.4*(mag - limiting_mag))
    return lim_mass

#%% Data
##All DR11##
z_low, z_high = 0, 6
dr11 = Table.read('Data/DR11-2arcsec-Jun-30-2019.fits', format='fits')

###only keeping the DR11 between redshifts >0 and <=6###
z_dr11 = np.array(dr11['z_use']) 
zdr11filt = np.logical_and(z_dr11>z_low, z_dr11<=z_high)
dr11 = dr11[zdr11filt]

###Only keeping galaxies with the "best galaxies" flag###
best_gals = np.array(dr11['Best galaxies'])
dr11 = dr11[best_gals]

mass_dr11 = np.array(dr11['Mstar_z_p'])
mass_dr11 = remove_bad(mass_dr11)
mass_dr11 = np.log10(mass_dr11)

z_dr11 = np.array(dr11['z_use']) 
z_dr11 = remove_bad(z_dr11)

kmag_dr11 = np.array(dr11['KMAG_20'])
kmag_dr11 = remove_bad(kmag_dr11)
# kmag_lim = 25.3 #Point source limiting magnitude of K-band AGN
kmag_lim = 24.5 #Limit where we are 99% complete in galaxy detection
"""The limit for point sources is dimmer than that for extended sources as 
its easier to detect a point source"""

compl_90A = (-0.11*z_dr11**2) + 1.01*(z_dr11) + 7.97  #90% completeness limit from Aarons paper for comparison

#%% Calculating limiting mass 
lim_mass = np.array([limiting_mass(i[0], i[1], kmag_lim) for i in zip(mass_dr11, tqdm(kmag_dr11))])

fig, ax = plt.subplots()

ax.scatter(z_dr11, mass_dr11, color='lightblue', marker='.', rasterized=True, zorder=0, label='Stellar mass')
ax.scatter(z_dr11, lim_mass, color='silver', marker='.', rasterized=True, zorder=0, label='Limiting mass')
ax.plot(z_dr11[sortA], compl_90A[sortA], color='lime', ls='dashed', label='Aaron')

ax.set(xlabel='z', ylabel='Mass', ylim=(6, 11.5))
plt.legend(loc='upper left')
plt.show()


#%% Calculating 90% completeness limit
z_bin = 0.25
n,m = z_low, z_low+z_bin

mass_final = np.array([])
z_final = np.array([])

mass_faint = np.array([])
mag_faint = np.array([])
z_faint = np.array([])

while m<=z_high:
    zfilt = np.logical_and(z_dr11>=n, z_dr11<m)
    
    #Split data into redshift bin
    z = z_dr11[zfilt]
    mass = lim_mass[zfilt]
    mag = kmag_dr11[zfilt]
    
    #Sorting the data from brightest to faintest
    magfilt = np.argsort(mag)
    mag = mag[magfilt]
    z = z[magfilt]
    mass = mass[magfilt]
    
    #Faintest 20% of galaxies
    faint =  np.nanpercentile(mag, 80)
    filt_mag = mag>=faint
    
    fig, ax = plt.subplots()
    
    ax.hist(mag, bins=30, edgecolor='white', fc='black', density=True)
    ax.vlines(faint, 0, 1, ls='--', color='red')
    ax.set(xlabel='K-band Mag', ylabel='Normalised Distribution')
    ax.set_title(f'{np.round(faint, 2)}')
    plt.show()    
    
    mag = mag[filt_mag]
    z = z[filt_mag]
    mass = mass[filt_mag]

    mass_faint = np.append(mass_faint, mass)
    mag_faint = np.append(mag_faint, mag)
    z_faint = np.append(z_faint, z)
     
    #Upper 90th percentile of the faintest 20% of the limiting galaxies 
    percentile = np.nanpercentile(mass, 90) 
    
    fig, ax = plt.subplots()
    
    ax.hist(mass, bins=30, edgecolor='white', fc='black', density=True, range=(7.5,12))
    ax.vlines(percentile, 0, 3, ls='--', color='red')
    ax.set(xlabel=r'Limiting Mass log($M_{\odot}/M_{*}$)', ylabel='Normalised Distribution')
    ax.set_title(f'{np.round(percentile, 2)}')
    plt.show()
    
    
    mass_final = np.append(mass_final, percentile)
    z_final = np.append(z_final, ((n+m)/2))
    m+=z_bin
    n+=z_bin

#%% Plotting mass
fig, ax = plt.subplots()

ax.scatter(z_dr11, lim_mass, color='black', marker='.', s=1, rasterized=True, zorder=0, label='Limiting mass')
ax.scatter(z_faint, mass_faint, color='lime', marker='.',s=1, rasterized=True, zorder=0, label='Faint mass')
ax.scatter(z_final, mass_final, color='red', marker='x', zorder=0, label='90% lim gals')
ax.set(xlabel='z', ylabel='Mass')
plt.legend(loc='upper left')
plt.show()

#%% Fitting line to data 
polynomials = np.polyfit(z_final, mass_final, 2) #Polymonial values for checking and to quote in paper
print(polynomials)
model = np.poly1d(polynomials) #Plots the line based on limiting mass data
polyline = np.linspace(z_low, z_high, 1000)

#%% 90% completeness curve based on full UDS

fig, ax = plt.subplots()

ax.scatter(z_dr11, lim_mass, fc='none', edgecolor='black', marker='.', alpha=0.3, rasterized=True, zorder=0, label='limiting mass')
ax.plot(z_dr11[np.argsort(z_dr11)], compl_90A[np.argsort(z_dr11)], color='lime', ls='dashed', label='Aaron')
ax.plot(polyline, model(polyline), color='red', ls='solid', label='All DR11')

# ax.hlines(10, 0, 6, color='black', ls='--', zorder=1)
ax.set(xlabel='z', ylabel=r'Stellar Mass log($M_{\odot}/M_{*}$)')#, xlim=(-0.25, z_high+0.25), ylim=(7.5, 12))
plt.legend(loc='upper left')
plt.show()

np.save('Data/complecurve_z.npy', polyline)
np.save('Data/complecurve_mass.npy', model(polyline))

