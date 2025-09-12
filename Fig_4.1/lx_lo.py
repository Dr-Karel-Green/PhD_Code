#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:07:51 2025
README: Code that recreates the plots 4.1 in my doctoral thesis.

This code plots the ratio of the x-ray luminosity to the optical luminosity of
the variability detected active galaxies. 

Galaxies are split into three catalogues based on detection type: variable only,
 dual detected and x-ray only prior to this code using topcat and java.
@author: karelgreen
"""
#%% Modules
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u
from astropy.constants import c, h
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
from math import pi, log10
from scipy.interpolate import interp1d
#%% Functions
def mag2flux(mag, wavelength):
    """Converts an AB magntiude into a flux in erg s^-1 cm^-2. Requires the wavelngth (or frequency)
    of the band the magnitude was measured in.
    mag = magntiude
    wavelength = corresponding wavelength to the band the magnitude was 
    measured in
    """
    flux_den = 10**((mag + 48.6)/-2.5)
    wavelength = wavelength*(u.m)
    flux = flux_den*(c/wavelength)
    return flux

def flux2lum(flux, z):
    """Converts a flux to a luminosity.
    Flux = Flux in erg/(s*cm**2)
    Z = Redshift"""
    distance = cosmo.luminosity_distance(z)
    distance = distance.to(u.cm)
    flux = flux*(u.erg/((u.s**1)*(u.cm**2)))
    luminosity = 4*pi*(distance**2)*flux
    return luminosity

def remove_bad(arr):
    """Removes dummy values in an array and replaces them with nan"""
    dummy = np.array([float('-inf'), float('inf'), 99, -99])
    mask = np.isin(arr, dummy) 
    arr[mask] = np.nan 
    return(arr)

def rf_opt(z, data):
    rfopt = 2500 * (u.AA)
    rfopt = rfopt.to(u.micron)
    bnd_nums = np.array([0.3546, 0.4320, 0.5956,0.6156, 0.7471, 0.8918, 1.0305, 1.2483, 1.6313, 2.2010])
    bnd_names = np.array(['uMAG_20','BMAG_20','VMAG_20','RMAG_20', 'iMAG_20', 'zMAG_20', 'YMAG_20', 'JMAG_20', 'HMAG_20', 'KMAG_20'])
    lum = np.array([])
    n=0
    for i in z:
        if np.isnan(i):
            lum = np.append(lum, np.nan)
            n+=1
        else:
            sed = np.array([])
            for idx in bnd_names:
                sed = np.append(sed, data[idx][n])
            n+=1
            intp = interp1d(bnd_nums, sed, bounds_error=False, fill_value='extrapolate')
            obs = rfopt*(1+i)
            
            flux = mag2flux(intp(obs.value), obs.to(u.m))
            flux = flux.value
            
            lum = np.append(lum, flux2lum(flux, i).value)
    return(lum)
    
    
def get_z(tbdata):
    ''' Get z with spectroscopic data where possible 
    Inputs:
        tbdata = table of data for all the objects we want z for (should contain
                  DR11 data in order to get z)
    Outputs:
        z = array of redshifts with spectroscopic redshifts where possible and 
            z_p where spec is not available.
    '''
    z = tbdata['z_spec']
    z[z==-1] = tbdata['z_a_1'][z==-1]
    return z


def get_xray_L_2(tbdata, Xray=True, band='Full'):
    ''' Function to find the monochromatic X-ray luminosity at 2keV in a 
    variety of flux bands. This assumes a power law for flux density and then
    finds the constant in this equation. The flux density is then used to find
    the monochromatic flux density which is in turn used to calulate the 
    monochromatic luminosity.
    Inputs:
        tbdata = data table for all the objects
        X-ray = whether the object has an X-ray counterpart. If it does not 
                then the survey limit is used as an upperlimit. Default is True
        band = what band of flux should be used:
                - 'Hard' 2-10keV (default)
                - 'Full' 0.5-10keV
                - 'Soft' 0.2-5keV
                - 'Uhrd' 5-10keV (not recommended)
    Outputs:
        L_2 = monochromatic luminosity at 2 keV in units of ergs/s/Hz
        F_2 = monochromatic flux at 2 keV in units of ergs/keV/s/cm^2
        flux = broad band flux in units of ergs/cm^2/s
        z = redshift
    '''
    z = np.array(tbdata['z_use'])
    z[z<=0] = np.nan
    
    ### get luminosity distance ###
    DL = cosmo.luminosity_distance(z) # need to sort out units
    DL = DL.to(u.cm)
    
    if band=='Hard': 
        upplim = 10 ## set band limits in keV
        lowlim = 2
        if Xray == True:
            # if it is an X-ray source, get flux from catalogue
            flux = tbdata['FbH']*10**-15 
            #flux# Units of erg cm**-2 s**-1
        else: # if it is non X-ray - use the upper limit
            flux = np.zeros(len(tbdata))
            flux += 6.5e-16 # Units of erg cm**-2 s**-1
    elif band=='Full': 
        upplim = 10
        lowlim = 0.5
        if Xray == True:
            flux = tbdata['FbF']*10**-15 # Units of erg cm**-2 s**-1
        else:
            flux = 4.4e-16 # Units of erg cm**-2 s**-1
    elif band=='Soft': 
        upplim = 2
        lowlim = 0.5
        if Xray == True:
            flux = tbdata['Fbs']*10**-15 # Units of erg cm**-2 s**-1
        else:
            flux = 1.4e-16 # Units of erg cm**-2 s**-1
    elif band=='Uhrd': 
        upplim = 10
        lowlim = 5
        if Xray == True:
            flux = tbdata['FbU']*10**-15 # Units of erg cm**-2 s**-1
        else:
            flux = 9.2e-15 # Units of erg cm**-2 s**-1
            
    ### Add units ###
    flux = flux* (u.erg) * (u.cm)**-2 * (u.s)**-1 
    upplim = upplim * u.keV
    lowlim = lowlim * u.keV

    ### get integrated flux density ###
    denom = ((upplim**(0.1))/(0.1)) - ((lowlim**(0.1))/(0.1))
    ### use this and flux value to get the power law constant ###
    const = flux / denom
    
    ### calculate flux density ###
    nu = 2 * u.keV # 2kev is value to evaluate at
    F_2 = const * (nu**(-0.9))
    
    
    ### calculate luminosity density ###
    L_2 = 4 * np.pi * (DL**2) * F_2
    
    L_2 = L_2.to((u.erg) * (u.s)**-1 * (u.Hz)**-1, equivalencies=u.spectral())
        
    L_2[L_2==0] = np.nan
    
    return L_2, F_2, z, flux

#%% Data

###X-ray and optical frequencies###
xfreq = 2*(u.keV)
xfreq = xfreq.to(u.J)/h

optfreq = 2500*(u.AA)
optfreq = c/(optfreq.to(u.m))

###Var Only###
var = Table.read('./data/varonly.fits', format='fits') 
cs = np.array(var['chandra_sky']).astype(bool)
var = var[cs]

#Variable Type#
j_v = np.array(var['Variable_j'])
k_v = np.array(var['Variable_k'])
jk_v = np.array(var['Variable_both'])
j_v[jk_v]=False
k_v[jk_v]=False
z_v = get_z(var)
z_v[z_v<=0] = np.nan

L2_v = get_xray_L_2(var, Xray=False)[0]
L2500_v = rf_opt(z_v, var)/optfreq

###Dual detected###
xrayvar = Table.read('./data/xrayvar.fits', format='fits') 

#Variable Type#
j_xv = np.array(xrayvar['Variable_j'])
k_xv = np.array(xrayvar['Variable_k'])
jk_xv = np.array(xrayvar['Variable_both'])
j_xv[jk_xv]=False
k_xv[jk_xv]=False

z_xv = get_z(xrayvar) 

L2_xv = get_xray_L_2(xrayvar)[0]
L2500_xv = rf_opt(z_xv, xrayvar)/optfreq

###X-ray only###
xray = Table.read('./data/xrayonly.fits', format='fits') 

z_x = np.array(xray['z_use'])
z_x[z_x<=0] = np.nan

L2_x = get_xray_L_2(xray)[0]
L2500_x = rf_opt(z_x, xray)/optfreq

###A-ox lines###
x = np.logspace(25,32,10)
y1 = 10 ** (np.log10(x)-1/0.3838)
y2 = 10 ** (np.log10(x) - 2/0.3838)

#%% Plotting

fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True, figsize=[4.8*1.1, 6.4*1.65])

ax0.scatter((L2500_x.value), (L2_x.value), marker='x', s=50, color='deepskyblue', label='X-ray', zorder=0)
ax0.scatter((L2500_v.value)[k_v], (L2_v.value)[k_v], marker=r'${\downarrow}$', s=100, color='red', label='Variable', zorder=2)
ax0.scatter((L2500_xv.value)[k_xv], (L2_xv.value)[k_xv], marker='.', s=100, color='darkviolet', label='X-ray&Variable', zorder=1)
ax0.scatter((L2500_xv.value)[jk_xv], (L2_xv.value)[jk_xv], marker='.', s=100, color='darkviolet', zorder=1)
ax0.scatter((L2500_v.value)[jk_v], (L2_v.value)[jk_v], marker=r'${\downarrow}$', s=100, color='red', zorder=2)

ax1.scatter((L2500_x.value), (L2_x.value), marker='x', s=50, color='deepskyblue', label='X-ray', zorder=0)
ax1.scatter((L2500_v.value)[j_v], (L2_v.value)[j_v], marker=r'${\downarrow}$', s=100, color='red', label='Variable', zorder=2)
ax1.scatter((L2500_xv.value)[j_xv], (L2_xv.value)[j_xv], marker='.', s=100, color='darkviolet', label='X-ray&Variable', zorder=1)
ax1.scatter((L2500_xv.value)[jk_xv], (L2_xv.value)[jk_xv], marker='.', s=100, color='darkviolet', zorder=1)
ax1.scatter((L2500_v.value)[jk_v], (L2_v.value)[jk_v], marker=r'${\downarrow}$', s=100, color='red', zorder=2)

ax0.plot(x,(y1), 'k', label=r'$\alpha_{OX} = 1$',zorder=0)
ax0.plot(x,(y2), 'k--', label=r'$\alpha_{OX} = 2$',zorder=0)

ax1.plot(x,(y1), 'k', label=r'$\alpha_{OX} = 1$',zorder=0)
ax1.plot(x,(y2), 'k--', label=r'$\alpha_{OX} = 2$',zorder=0)

ax0.set(xscale='log', yscale='log')
ax0.set_ylim(1e23 ,1e28)
ax0.set_xlim(1e26, 1e31)
# ax0.set_title('K-band', loc='left')

ax1.set(xscale='log', yscale='log')
ax1.set_ylim(1e23 ,1e28)
ax1.set_xlim(1e26, 1e31)
ax1.set_xlabel(r'$L_{{o}}$ (erg$s^{{-1}}Hz^{{-1}}$)', fontsize="x-large")
# ax1.set_title('J-band', loc='left')

# ax1.legend()
ax0.legend()

fig.supylabel(r'$L_{{2keV}}$ (erg$s^{{-1}}Hz^{{-1}}$)', x=-0.01, fontsize="x-large")
plt.subplots_adjust(hspace=0)

fig.text(0.45, 0.86, 'K-band', fontsize="x-large")
fig.text(0.45, 0.48, 'J-band', fontsize="x-large")
plt.savefig('lx_lo.pdf',bbox_inches='tight')
plt.show()
