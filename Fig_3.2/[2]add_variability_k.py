#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 15:33:37 2025
README: Code that takes the galaxies in a certain magnitude bin, adds in an amount
of variability to the galaxies, and then calculates what percentage of them
are now classed as variable.
@author: karelgreen
"""

#%%Modules
from datetime import datetime
startTime = datetime.now()
import numpy as np
import os
from math import pi, exp
from tqdm import tqdm
from astropy.table import Table
import matplotlib.pyplot as plt
#%%Functions
def percent_err(arr, err, new):
    """Calculates the error on a value as a percentage. Useful for if you are 
    normalising values and want to keep the ratio of the vlaue to its error
    the same
    arr = original array
    err = original errors on array
    new = new array, modified version of arr. (for example new could be arr but
    but normalised
    out = the new error for "new" """
    frac = np.divide(err, arr)
    out = np.multiply(frac, new)
    return out
    
def chi_2(flux, err):
    """Calculates the chi^2 of the data according to the equation in Lizzies
    paper. 
    flux = array of fluxes 
    err = array of errors
    chi2 = calculated chi^2 of the curve"""
    avg = np.nanmean(flux)
    arr = np.array([])
    for i in zip(flux, err):
        arr = np.append(arr, (((i[0]-avg)**2)/((i[1])**2)))
    chi2 = np.round(np.nansum(arr), 2)
    return chi2

def likecurve(xi, avg, erri, sigq):
    """Calculates the likelihood value
    xi = individual value in an array
    erri = associated error on value xi
    avg = average of all values in the array
    singq = value of noise subtracted variability being tested
    """
    num = exp((xi - avg)**2/((-2)*(erri**2+sigq**2)))
    denom = ((2*pi)**0.5)*((erri**2+sigq**2)**0.5)
    
    like = num/denom
    return like  

#%%Data
np.random.seed(5)
sems = np.array(['05B', '06B', '07B', '08B', '09B', '10B', '11B', '12B'])

input_dir = 'data/K/'
output_dir = 'data/Knew/'

files = np.sort(os.listdir(input_dir))  

for fil in tqdm(files):
    data = Table.read(os.path.join(input_dir, fil), format='fits')

    
#%%Getting the flux from the data
    ID = np.array(data['DR11_ID'])
    RA = np.array(data['RA'])
    Dec = np.array(data['Dec'])
    
    colnames = data.colnames
    kflux_start = colnames.index('K_flux_05B')
    kflux_end = colnames.index('K_fluxerr_12B') + 1
    
    chi2k_org = np.array(data['chi2_k'])
    sigk_org = np.array(data['sigma k'])
    
    chi2k_final = np.empty((101, 0), float)
    sigk_final = np.empty((101, 0), float)
    vark_final = np.empty((101, 0), float)
    
    for i in tqdm(ID):
        rw = list(data[np.where(int(i)==ID)[0][0]])
        
        flux_all = rw[kflux_start:kflux_end] 
        
        flux = flux_all[::2]
        flux[1] = np.nan #Converting masked elements to nan values

        fluxerr = flux_all[1::2]
        fluxerr[1] = np.nan #Converting maksked elements to nan values

        #%%Normalising the flux such that its mean is unity
        avg = np.nanmean((flux))
        flux_norm = flux/avg
        norm_avg = np.nanmean(flux_norm)
        
        fluxerr_norm = percent_err(flux, fluxerr, flux_norm)
        
        #%%Plotting normalised array
        
        chi2 = chi_2(flux_norm, fluxerr_norm)
        if chi2>30:
            var=True
        else:
            var=False
        chi2 = np.round(chi2, 2)
        
        # fig, ax = plt.subplots()
        # ax.errorbar(sems, flux_norm, fluxerr_norm, color='deeppink', ecolor='lightgrey', marker='o', ls='none')
        # ax.set(xlabel='Semester', ylabel='K-Band Flux (Normalised)')
        # ax.set_title(f'ID = {i}', loc='left')
        # ax.set_title(fr'$\chi^2$ = {chi2}', loc='right')
        # plt.tight_layout()
        # plt.show()
        
    #%%Generating likelihood curve maximum sigma
        sigma_rng = np.arange(0, 1.01, 0.01)
        sigma_rng = np.round(sigma_rng, 2)
        
        l = np.array([])
        for sig in sigma_rng:
            out = np.array([])
            for idx in zip(flux_norm, fluxerr_norm):
                out = np.append(out, likecurve(idx[0], norm_avg, idx[1], sig))
            l = np.append(l, np.nanprod(out))
        
        max_sigma = sigma_rng[np.where((l/np.max(l))==1)[0][0]]
        max_sigma = np.round(max_sigma, 2)
        #%%
        # fig, ax = plt.subplots()
        
        # ax.plot(sigma_rng, l/np.max(l), color='deeppink', ls='--')
        # ax.vlines(max_sigma, 0, 1, color='lightgrey', ls='-.')
        # ax.set(xlabel=r'$\sigma$', ylabel='L')
        # ax.set_title(f'ID = {i}', loc='left')
        # ax.set_title(fr'$\chi^{2}$ = {chi2}, var = {var},  $\sigma$ = {max_sigma}', loc='right')
        # plt.tight_layout()
        # plt.show()  
                
        #%%Getting Gaussian distribution of numbers
        chi2_newsave = np.array([])
        var_newsave = np.array([])
        max_sigma_newsave = np.array([])
        
        mu = 0 #Want the Gaussian distribution to be about the value 0.
        
        for sigq in sigma_rng:
            s = np.random.normal(mu, sigq, 8)
            s = np.sort(s)
            
            #%%Making new light curve
            
            flux_new = np.add(flux_norm, s) #adding fake variability to the values
            
            chi2_new = chi_2(flux_new, fluxerr_norm) #new chi2 values from flux with fake variability
            
            if chi2_new>30:
                var_new=True
            else:
                var_new=False
            
            chi2_new = np.round(chi2_new, 2)
            
            chi2_newsave = np.append(chi2_newsave, chi2_new)
            var_newsave = np.append(var_newsave, var_new)
            
            # fig, ax = plt.subplots()
            
            # ax.errorbar(sems, flux_new, fluxerr_norm, color='deeppink', ecolor='lightgrey', marker='o', ls='none')
            # ax.set(xlabel='Semester', ylabel='K-Band Flux (Normalised)')
            # ax.set_title(f'ID = {i}', loc='left')
            # ax.set_title(fr'$\chi^2$ = {chi2_new}', loc='right')
            # plt.tight_layout()
            # plt.show()
            
            #%%Generating likelihood curve maximum sigma for new curve
            
            l_new = np.array([])
            for sig in sigma_rng:
                out = np.array([])
                for idx in zip(flux_new, fluxerr_norm):
                    out = np.append(out, likecurve(idx[0], norm_avg, idx[1], sig))
                l_new = np.append(l_new, np.nanprod(out))
            
            max_sigma_new = sigma_rng[np.where((l_new/np.max(l_new))==1)[0][0]]
            max_sigma_new = np.round(max_sigma_new, 2)
            
            max_sigma_newsave = np.append(max_sigma_newsave, max_sigma_new)
            #%%
            # fig, ax = plt.subplots()
            
            # ax.plot(sigma_rng, l_new/np.max(l_new), color='deeppink', ls='--')
            # ax.vlines(max_sigma_new, 0, 1, color='lightgrey', ls='-.')
            # ax.set(xlabel=r'$\sigma$', ylabel='L')
            # ax.set_title(f'ID = {i}', loc='left')
            # ax.set_title(fr'$\chi^{2}$ = {chi2_new}, var = {var_new},  $\sigma$ = {max_sigma_new}', loc='right')
            # plt.tight_layout()
            # plt.show()  
    
        chi2k_final = np.column_stack((chi2k_final, chi2_newsave))
        sigk_final = np.column_stack((sigk_final, max_sigma_newsave))
        vark_final = np.column_stack((vark_final, var_newsave))
    
        #%% Adding columns to data and saving file
    
    col_sigq = np.char.add(np.array(['sigq']).astype(str), sigma_rng.astype(str))
    col_chi2k = np.char.add(np.array(['chi2_k']).astype(str), sigma_rng.astype(str))
    col_var_k = np.char.add(np.array(['Var_k']).astype(str), sigma_rng.astype(str))
    
    final_names = np.concatenate((col_chi2k, col_sigq, col_var_k), axis=0)
    
    data_calc =  np.vstack((chi2k_final, sigk_final, vark_final)).T
    
    print("Shape of data:", len(data))  # should be 3001
    print("Shape of data_calc:", data_calc.shape)  # should be (3001 rows, 303 columns)
    print("Length of final_names:", len(final_names))  # should be 303
    data.add_columns([data_calc[:, i] for i in range(data_calc.shape[1])], names=final_names)
    data.write(output_dir + f'{fil}_new.fits', overwrite=True)  
        
    print('\n',datetime.now() - startTime)