import time, sys, os
import h5py
import numpy as np
import scipy
from scipy import stats
import astropy
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from multiprocessing import Pool
import subprocess
import pickle
import os
import numpy as np
from scipy.special import gamma, gammainc
import re
import pandas as pd


home_dir = '/Users/brian/Library/CloudStorage/OneDrive-UniversityofHertfordshire/DwarfAGN'
cl_dir = '/beegfs/car/bbichanga/DwarfAGN'

dir_pick = [home_dir,cl_dir]
dir_pick = dir_pick[1]

cosmos = fits.open(dir_pick+'/COSMOS2020_FARMER_R1_v2.1_p3.fits',memmap=True)
cosmos_dat = cosmos[1].data
dire = dir_pick+'/dwarf_agn/emcee3_renamed/emcee3/' # my data
dwarf_picker = np.where((cosmos_dat['lp_zPDF_l68'] > 0.1) & (cosmos_dat['lp_zPDF_u68'] < 0.25) & (cosmos_dat['lp_zPDF_l68'] > 0.1) & (cosmos_dat['lp_mass_med_max68'] < 9.5)
                        & (cosmos_dat['lp_mass_med_min68'] > 8.))[0]
results_type = "emcee"


directory = os.fsencode(dire)
columns = ['mass', 'dust', 'tage', 'tau', 'fagn', 'agn_tau', 'duste_umin', 'duste_qpah', 'duste_gamma','mass16', 'dust16', 'tage16', 'tau16', 'fagn16', 'agn_tau16', 'duste_umin16', 'duste_qpah16', 'duste_gamma16', 'psi', 'psi1']
keey = []


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pickle"):
        # f  = [int(s) for s in filename.split('_') if s.isdigit()]
        # print(f, filename)
        # try:
        #     f  = [int(s) for s in filename.split('_') if s.isdigit()]
        #     # print(f)
        #     new_string = re.sub('%s'%(f[0]), '%s' %(cosmos_dat['ID'][dwarf_picker][f][0]), filename)
        #     keey.append([f[0],cosmos_dat['ID'][dwarf_picker][f][0]])
        # except:
        f  = [int(s) for s in filename if s.isdigit()]
        a = map(str, f)    
        b = ''.join(a) 
        c =int(b)
        f = np.asarray([c])
        new_string = re.sub('%1.0f'%(f) , '%1.0f' %(cosmos_dat['ID'][dwarf_picker][f][0]), filename)
        # print(f,new_string)
        
        os.rename(dir_pick+'/dwarf_agn/emcee3_renamed/emcee3/'+filename, dir_pick+'/dwarf_agn/emcee3_renamed/emcee3/'+new_string)
