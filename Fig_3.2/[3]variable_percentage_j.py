#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:44:52 2025
README: Code that calculates the number of galaxies class as variable after
having simulated variability added in and then plots that as a curve based on
the brightness bin. 
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

#%% Data

files_all = np.sort(os.listdir('/data/Jnew/'))
sigma_rng = np.arange(0, 1.01, 0.01)
sigma_rng = np.round(sigma_rng, 2)
num = 99 #Detection limit curve percentage. e.g. if num=90 it saves the 90% detection limit
sigma = np.array([])
flow = np.array([]) #lowest flux (flux low)
fhigh = np.array([]) #hisgest flux (flux high)
colours = ['red', 'orangered','tab:orange', 'gold', 'yellow','forestgreen', 'limegreen', 'blue', 'dodgerblue', 'blueviolet' ,'mediumpurple', 'violet', 'pink']

fig, ax = plt.subplots()

for fil in zip(files_all, tqdm(colours)):
    data = Table.read(f'data/Jnew/{fil[0]}', format='fits')

    flxlow, flxhigh = fil[0].split('_')[2:-1]
    flxhigh = flxhigh.strip('.fits')
    
    flxlow = float(flxlow)
    flxhigh = float(flxhigh)
    
    flow = np.append(flow, flxlow)
    fhigh = np.append(fhigh, flxhigh)
    
    var_names = np.char.add(np.array(['Var_j']).astype(str), sigma_rng.astype(str))
   
    pcent = np.array([]) #percentage of galaxies classed as variable in a given brightness bin, with the amount of fake variability added
    for i in var_names:
        var = np.array(data[f'{i}']).astype(bool)
        pcent = np.append(pcent , percent_var(var))
    
    intp = interpolate.interp1d(pcent, sigma_rng, bounds_error=False, fill_value=np.array([1]))
    sigma = np.append(sigma, intp(num))
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
ax.set_title('J-band', loc='left')
ax.legend(title='AB Magnitude', bbox_to_anchor=(1, 0.995))
plt.savefig('jband_pcnt_curve.pdf', bbox_inches="tight")
plt.show()
