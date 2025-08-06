#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 13:17:42 2025

@author: karelgreen
"""
#%% Modules
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from math import exp, pi
from tqdm import tqdm
from scipy.signal import peak_widths, find_peaks
from scipy.interpolate import interp1d
from astropy.time import Time
#%%Functions 
def likecurve(xi, avg, erri, sigq):
    num = exp((xi - avg)**2/((-2)*(erri**2+sigq**2)))
    denom = ((2*pi)**0.5)*((erri**2+sigq**2)**0.5)
    
    like = num/denom
    return like  

def percent_err(arr, err, new):
    frac = np.divide(err, arr)
    out = np.multiply(frac, new)
    out = np.absolute(out)
    return out

#%%Data
data = Table.read('data/[redacted]', format='fits', memmap=True) 
"""File that contains the light curves taken on monthly timescales for the galaxies
in the dataset. Using memory mapping to be able to load the file without crashing
my computer"""
# data = data[0:5]
ID = np.array(data['DR11_ID'])

MJD = Table.read('data/[redacted]', format = 'fits')
"""File that contains the actual Gregorian months that corresponds to the modified
Julian date of when the data was collected for plotting purposes.
"""
mjd = np.array(MJD['MJD'])
months = np.array(MJD['Month']).astype(str)

colnames = np.array(data.colnames)
sigma_rng = np.arange(0, 1.01, 0.001)

#%% K-lightcurves
kbool = np.char.startswith(data.colnames, 'K_').astype(bool)
kflux_bool = np.char.startswith(data.colnames, 'K_flux_').astype(bool)
kfluxerr_bool = np.char.startswith(data.colnames, 'K_fluxerr_').astype(bool)

kfluxcols = colnames[kflux_bool]
kfluxerrcols = colnames[kfluxerr_bool]

kmonths = np.char.strip(kfluxcols, 'K_flux_')
kmonths = np.char.replace(kmonths, 'eb', 'feb')
mask_k = np.isin(months, kmonths)


#%% Raw monthly lightcurves
for i in enumerate(data):
    lc = np.array(list(i[1]))[kflux_bool] #Must convert to list first to stop the array being marked as void type
    err = np.array(list(i[1]))[kfluxerr_bool]
    
    fig, ax  = plt.subplots()
     
    ax.errorbar(mjd[mask_k], lc, err, marker='none', color='blue', ecolor='lightgrey', ls='None', zorder=0)
    ax.scatter(mjd[mask_k], lc, marker='o', color='red', zorder=1)
    ax.set_xlabel('MJD', fontsize='x-large')
    ax.set_ylabel('K-band flux', fontsize='x-large')
    ax.set_title(f'ID = {ID[i[0]]}', loc='left')
    plt.savefig(f'month_lightcurves/K/month_lc_{ID[i[0]]}.pdf')
    plt.close()

#%%Splitting up by year 
timek = Time(mjd[mask_k], format='mjd') #converting from mjd to a datetime object
timek = timek.to_datetime()
n=0
kyearsplit = [] #Will eventually become an array that is used for colour coding the months

for i in tqdm(range(len(timek))):
    diff = timek[i] - timek[i-1] #difference between subsequent measurements as a datetime object
    #use [i] and [i-1] instead of [i+1] and [i] to avoid array length errors  
    diff_days = diff.days #difference between subsequent measurements in number of days
    if diff_days < 120:
        kyearsplit.append(n)
    else:
        n+=1
        kyearsplit.append(n)

            
kyearsplit = np.array(kyearsplit)
kdates = mjd[mask_k]
colours = ['red', 'orange', 'gold', 'green', 'deepskyblue', 'royalblue', 'purple', 'black']

for i in tqdm(enumerate(data)):
    lc = np.array(list(i[1]))[kflux_bool] #Must convert to list first to stop the array being marked as void type
    err = np.array(list(i[1]))[kfluxerr_bool]
    
    fig, ax  = plt.subplots()
    
    for group_id in range(kyearsplit.max()):
        plt_mask = np.where(kyearsplit == group_id)
        if group_id < len(colours):
            clr = colours[group_id]
        else:
            clr = 'grey'
        
        ax.errorbar(kdates[plt_mask], lc[plt_mask], err[plt_mask], marker='o', color=clr, ecolor='lightgrey', linestyle='none')
    
    ax.set_xlabel('MJD', fontsize='x-large')
    ax.set_ylabel('K-band flux', fontsize='x-large')
    ax.set_title(f'ID = {ID[i[0]]}', loc='left')
    plt.savefig(f'month_lightcurves_colourcoded/K/month_lc_{ID[i[0]]}.pdf')
    plt.close()

#%% Normalising the light curves

for i in tqdm(enumerate(data)):
    lc = np.array(list(i[1]))[kflux_bool] #Must convert to list first to stop the array being marked as void type
    err = np.array(list(i[1]))[kfluxerr_bool]
    
    overall_avg = np.nanmean(lc)
    fig, ax  = plt.subplots()
    
    for group_id in range(kyearsplit.max()):
        plt_mask = np.where(kyearsplit == group_id)
        month_avg = np.nanmean(lc[plt_mask])
        diff = overall_avg - month_avg
        
        newlc = lc[plt_mask] + diff #include the different between the month error and overall error to avoid under or over representing each individual month 
        normedlc = newlc/overall_avg
        normedlcerr = percent_err(lc[plt_mask], err[plt_mask], normedlc)

        if group_id < len(colours):
            clr = colours[group_id]
        else:
            clr = 'grey'
        
        ax.errorbar(kdates[plt_mask], normedlc, normedlcerr, marker='o', color=clr, ecolor='lightgrey', linestyle='none')
    
    ax.set_xlabel('MJD', fontsize='x-large')
    ax.set_ylabel('K-band flux', fontsize='x-large')
    ax.set_title(f'ID = {ID[i[0]]}', loc='left')
    plt.savefig(f'month_lightcurves_colourcoded/K/month_lc_{ID[i[0]]}_normed.pdf')
    plt.close()