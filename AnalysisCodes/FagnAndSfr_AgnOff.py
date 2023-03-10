home_dir = '/Users/brian/Library/CloudStorage/OneDrive-UniversityofHertfordshire/DwarfAGN'
cl_dir = '/beegfs/car/bbichanga/DwarfAGN'

dir_pick = [home_dir,cl_dir]
dir_pick = dir_pick[1]


import prospect.io.read_results as reader
import numpy as np
from scipy.special import gamma, gammainc
from prospect.plotting.corner import quantile
import pandas as pd
import os
    
results_type = "emcee"


directory = (dir_pick + '/dwarf_agn/emcee3_renamed/emcee7/')
columns = ['ID','mass', 'dust', 'tage', 'tau', 'duste_umin', 'duste_qpah', 'duste_gamma','mass16', 'dust16', 'tage16', 'tau16', 'duste_umin16', 'duste_qpah16', 'duste_gamma16', 'psi', 'psi1']
dat = []


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    
    if filename.endswith(".h5"):
        # filename = os.fsdecode(file)
        print(filename)
        # try:
        f  = [int(s) for s in filename.split('_') if s.isdigit()]
        result, obs, model  = reader.results_from(os.path.join(directory, filename), dangerous=False)

        # Maximum posterior probability sample
        imax = np.argmax(result['lnprobability'])
        csz = result["chain"].shape
        # if result["chain"].ndim > 2:
        # emcee
        i, j = np.unravel_index(imax, result['lnprobability'].shape)
        theta_max = result['chain'][i, j, :].copy()
        flatchain = result["chain"].reshape(csz[0] * csz[1], csz[2])
        #         else:
        #     # dynesty
        # theta_max = result['chain'][imax, :].copy()
        # flatchain = result["chain"]

        # 16th, 50th, and 84th percentiles of the posterior

        weights = result.get("weights", None)

        post_pcts = np.median(flatchain.T,axis=1)
        post_pcts16 = quantile(flatchain.T, q=[0.16, 0.50, 0.84], weights=weights)

        #----------------------------------------------------------------------------------------
        ID, mass, dust, tage, tau, duste_umin, duste_qpah, duste_gamma = f[0],post_pcts[0],post_pcts[1],post_pcts[2],post_pcts[3],post_pcts[4],post_pcts[5],post_pcts[6]
        ID, mass16, dust16, tage16, tau16, duste_umin16, duste_qpah16, duste_gamma16 = f[0],post_pcts16[0,0],post_pcts16[1,0],post_pcts16[2,0],post_pcts16[3,0],post_pcts16[4,0],post_pcts16[5,0],post_pcts16[6,0]
        #----------------------------------------------------------------------------------------
        #We calculate the star formation rate:

        tage, tau, mass = post_pcts[2], post_pcts[3], post_pcts[0]
        # for delay tau this function gives the (unnormalized) SFR 
        # for any t, tau combo in M_sun/Gyr
        # sfr = lambda t,tau: return (t/tau) * np.exp(-t/tau)

        sfr = lambda t,tau:(t/tau) * np.exp(-t/tau)
        # now we numerically integrate this SFH from 0 to tage to get the mass formed
        times = np.linspace(0, tage, 1000)
        A = np.trapz(sfr(times, tau), times)
        # But this could also be done using an incomplete gamma function (integral of xe^{-x})
        A = tau * gamma(2) * gammainc(2, tage/tau)
        # and now we renormalize the formed mass to the actual mass value 
        # to get the the SFR in M_sun per Gyr 
        psi = mass * sfr(tage, tau) / A
        # if we want SFR in Msun/year
        psi1 = psi/1e9

        row = ID, mass, dust, tage, tau, duste_umin, duste_qpah, duste_gamma,mass16, dust16, tage16, tau16, duste_umin16, duste_qpah16, duste_gamma16, psi, psi1,
        dat.append(row)
        # except:
        #     print(filename)
            
    
df = pd.DataFrame(dat,columns=columns)
df.to_csv(dir_pick+'/analysis/prospector_results_AgnOff.csv')