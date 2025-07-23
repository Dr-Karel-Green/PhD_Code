#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:46:38 2025
README: Code that imports a file, calculates the percentage of galaxies calssed
as variable for each sigq of added variance and plots it as a curve.
@author: karelgreen
"""
#%% Modules
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True) #Allws for underlining of text in the legend
import os
from tqdm import tqdm
from scipy import interpolate 
import matplotlib.ticker as mtick

#%%Functions
def percent_var(arr):
    total = arr.size
    true = arr.sum()
    percent = (true/total)*100
    percent = np.round(percent, 2)
    return percent

fp = '/Volumes/Cherry/PhD/Cleaned_UP/Fig_3.2/data/'
#%% Data 
files_all = np.sort(os.listdir(fp+'Knew/'))
sigma_rng = np.arange(0, 1.01, 0.01)
sigma_rng = np.round(sigma_rng, 2)
nums = np.array([99, 90, 80, 50]) #Detection limit curve percentage. e.g. if nums=90 it saves the 90% detection limit

colours = ['red', 'orangered','tab:orange', 'gold', 'yellow','forestgreen', 'limegreen', 'blue', 'dodgerblue', 'blueviolet' ,'mediumpurple', 'violet', 'pink']

sigma = np.array([])
flow = np.array([])
fhigh = np.array([])

fig, ax = plt.subplots()

for fil in zip(files_all, tqdm(colours)):
    data = Table.read(fp+f'Knew/{fil[0]}', format='fits')

    flxlow, flxhigh = fil[0].split('_')[2:-1] #Getting the magnitude values from how I named the file
    flxhigh = flxhigh.strip('.fits')
    
    flxlow = float(flxlow)
    flxhigh = float(flxhigh)
    
    flow = np.append(flow, flxlow)
    fhigh = np.append(fhigh, flxhigh)
    
    """Getting the column names for the amount of simulated variability added"""
    var_names = np.char.add(np.array(['Var_k']).astype(str), sigma_rng.astype(str))
    
    """Calculating the percentage of galaxies classed as variable for this brightness bin and 
    this amount of fake variability added"""
    pcent = np.array([percent_var(np.array(data[f'{i}']).astype(bool)) for i in var_names])

    """Interpolating the curves so I can save them and read off detection limits for them"""
    intp = interpolate.interp1d(pcent, sigma_rng, bounds_error=False, fill_value=np.array([1]))

    for num in nums:
        sigma = intp(num)
        np.save(f'./data/detection_curve_k/k{int(num)}_detect_lim.npy', sigma)

    
    #%% Plotting
    if flxlow == 20.0:
        lbl = f' $<$ {flxlow}' 
    elif flxhigh == 25.5:
        lbl = f' $>$ {flxhigh}' 
    else:
        lbl = f'{flxlow} - {flxhigh}'  
         
    ax.plot(sigma_rng, pcent, ls='--', color=f'{fil[1]}', label=lbl)

ax.set_xlabel(r'$\sigma_{q}$ added', fontsize="x-large")
ax.set_ylabel(r'% classed as variable', fontsize="x-large")   
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title('K-band', loc='left')
ax.legend(title='AB Magnitude', bbox_to_anchor=(1, 0.995))
plt.savefig('Fig_3.3_kband.pdf', bbox_inches="tight")
plt.show()

np.save('./data/klowflx.npy', flow)
np.save('./data/khighflx.npy', fhigh)    