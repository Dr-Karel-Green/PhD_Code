#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 15:08:10 2025
README: Code that takes all the values of the magnitudes of the inactive galaxies, 
sorts them into bins based on their apparent magnitude and saves arrays of these
galaxies based on these filters.
"""

#%%Modules
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%Data - Inactive galaxies
data = Table.read('[redacted]')

#%%Columns - Wavelengths come from Hewett et al 2006
jmag = np.array(data['JMAG_20'])
kmag = np.array(data['KMAG_20'])

#%% Histograms of apparent magnitudes to determine limits
"""
Galaxies outside of the grey lines are grouped into two bins with a wider range.
Galaxies inside the grey lines are binned according to the limits on each bar
of the histogram.
"""
bns = 20
rng=(17,27)
fig, ax = plt.subplots()

ax.hist(jmag, bins = bns, histtype='step', edgecolor='dodgerblue', range=rng, linewidth=1.5, label='J-band', zorder=1)
ax.hist(kmag, bins = bns, histtype='step', edgecolor='deeppink', range=rng, linewidth=1.5, linestyle='--',label='K-band', zorder=2)
ax.vlines(20, 0, 25000, ls='--', color='grey', zorder=3)
ax.vlines(25.5, 0, 25000, ls='--', color='grey', zorder=3)

ax.set(ylabel='# of galaxies', xlabel='Apparent Magnitude (AB)', ylim=(0,22500))
ax.invert_xaxis()

plt.legend()
plt.savefig('flux_bins.pdf')
plt.show()


#%% Widths 
"""Aything lower than -15.25, then 0.25 width bins, then anything above -13.25"""
low = np.concatenate(([27], np.arange(25.5, 19.9, -0.5)))
high = np.concatenate((np.arange(25.5, 19.9, -0.5), [17]))

for i in zip(low, tqdm(high)):
    filt = np.logical_and(jmag <= i[0], jmag > i[1])
    save = data[filt]
    save.write(f'data/J/sems_reggals_{i[0]}_{i[1]}.fits', overwrite=True)      
    
for i in zip(low, tqdm(high)):
    filt = np.logical_and(kmag <= i[0], kmag > i[1])
    save = data[filt]
    save.write(f'data/K/sems_reggals_{i[0]}_{i[1]}.fits', overwrite=True)      

