#!/usr/bin/env python
# coding: utf-8

import time, sys, os
import h5py
import numpy as np
import scipy
from scipy import stats
import astropy
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import fsps
import sedpy
import prospect
import emcee
import dynesty
from prospect.models import priors
from prospect.models.templates import TemplateLibrary
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
from prospect.likelihood import chi_spec, chi_phot
from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect import prospect_args
import prospect.io.read_results as reader
from multiprocessing import Pool
import subprocess
import pickle


ext_dir = '/Volumes/RyanSSD/Ryan_Work/'
home_dir = 'E:/Ryan_Work/'
home_dir = '/Users/brian/Library/CloudStorage/OneDrive-UniversityofHertfordshire/DwarfAGN'
cl_dir = '/beegfs/car/bbichanga/DwarfAGN'

dir_pick = [ext_dir,home_dir,cl_dir]
dir_pick = dir_pick[2]

cosmos = astropy.io.fits.open(dir_pick+'/COSMOS2020_FARMER_R1_v2.1_p3.fits',memmap=True)
cosmos_dat = cosmos[1].data
cosmos_hdr = cosmos[1].header
cosmos_cols = cosmos[1].columns
cosmos.close()

dwarf_picker = np.where((cosmos_dat['lp_zPDF_l68'] > 0.1) & (cosmos_dat['lp_zPDF_u68'] < 0.25) & (cosmos_dat['lp_zPDF_l68'] > 0.1) & (cosmos_dat['lp_mass_med_max68'] < 9.5)
                        & (cosmos_dat['lp_mass_med_min68'] > 8.))[0]


def run_prospector(galaxy):

    gal = galaxy

    def build_obs(**extras):
        """
        
        :param snr:
            The S/N to assign to the photometry, since none are reported 
            in Johnson et al. 2013
            
        :param ldist:
            The luminosity distance to assume for translating absolute magnitudes 
            into apparent magnitudes.
            
        :returns obs:
            A dictionary of observational data to use in the fit.
        """
        from prospect.utils.obsutils import fix_obs
        import sedpy

        # The obs dictionary, empty for now
        obs = {}

        # These are the names of the relevant filters, 
        # in the same order as the photometric data (see below)
        galex = ['galex_FUV', 'galex_NUV']
        spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]
        # sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]
        # acs = ['acs_wfc_f814w']
        hsc = ['hsc_{0}'.format(b) for b in ['g','r','i','z','y']]
        # subaru = ['subaru_suprimecam_ia'+n for n in ['484','527','624','679','738','767']]
        # vista = ['vista_vircam_{0}'.format(b) for b in ['Y','J','H','Ks']]
        filternames = galex + hsc + spitzer
        # And here we instantiate the `Filter()` objects using methods in `sedpy`,
        # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
        obs["filters"] = sedpy.observate.load_filters(filternames)

        # Now we store the measured fluxes for a single object, **in the same order as "filters"**
        # In this example we use a row of absolute AB magnitudes from Johnson et al. 2013 (NGC4163)
        # We then turn them into apparent magnitudes based on the supplied `ldist` meta-parameter.
        # You could also, e.g. read from a catalog.
        # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
        mags = np.array([cosmos_dat['GALEX_FUV_MAG'][dwarf_picker[gal]],cosmos_dat['GALEX_NUV_MAG'][dwarf_picker[gal]],
                         cosmos_dat['HSC_g_MAG'][dwarf_picker[gal]],cosmos_dat['HSC_r_MAG'][dwarf_picker[gal]],cosmos_dat['HSC_i_MAG'][dwarf_picker[gal]],
                         cosmos_dat['HSC_z_MAG'][dwarf_picker[gal]],cosmos_dat['HSC_y_MAG'][dwarf_picker[gal]],cosmos_dat['IRAC_CH1_MAG'][dwarf_picker[gal]],
                         cosmos_dat['IRAC_CH2_MAG'][dwarf_picker[gal]],cosmos_dat['IRAC_CH3_MAG'][dwarf_picker[gal]],cosmos_dat['IRAC_CH4_MAG'][dwarf_picker[gal]]])
        # dm = 25 + 5.0 * np.log10(ldist)
        obs["maggies"] = 10**(-0.4 * mags)


        errs = np.array([cosmos_dat['GALEX_FUV_MAGERR'][dwarf_picker[gal]],cosmos_dat['GALEX_NUV_MAGERR'][dwarf_picker[gal]],
                         cosmos_dat['HSC_g_MAGERR'][dwarf_picker[gal]],cosmos_dat['HSC_r_MAGERR'][dwarf_picker[gal]],cosmos_dat['HSC_i_MAGERR'][dwarf_picker[gal]],
                         cosmos_dat['HSC_z_MAGERR'][dwarf_picker[gal]],cosmos_dat['HSC_y_MAGERR'][dwarf_picker[gal]],cosmos_dat['IRAC_CH1_MAGERR'][dwarf_picker[gal]],
                         cosmos_dat['IRAC_CH2_MAGERR'][dwarf_picker[gal]],cosmos_dat['IRAC_CH3_MAGERR'][dwarf_picker[gal]],cosmos_dat['IRAC_CH4_MAGERR'][dwarf_picker[gal]]])
        
        obs["maggies_unc"] = 10**(-0.4 * (mags-errs)) - 10**(-0.4 * mags)
        
        # Now we need a mask, which says which flux values to consider in the likelihood.
        # IMPORTANT: the mask is *True* for values that you *want* to fit, 
        # and *False* for values you want to ignore.  Here we ignore the spitzer bands.


        # This is an array of effective wavelengths for each of the filters.  
        # It is not necessary, but it can be useful for plotting so we store it here as a convenience
        obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

        # We do not have a spectrum, so we set some required elements of the obs dictionary to None.
        # (this would be a vector of vacuum wavelengths in angstroms)
        obs["wavelength"] = None
        # (this would be the spectrum in units of maggies)
        obs["spectrum"] = None
        # (spectral uncertainties are given here)
        obs['unc'] = None
        # (again, to ignore a particular wavelength set the value of the 
        #  corresponding elemnt of the mask to *False*)
        obs['mask'] = None
        
        obs['redshift'] = cosmos_dat['lp_zPDF'][dwarf_picker[gal]]

        # This function ensures all required keys are present in the obs dictionary,
        # adding default values if necessary
        obs = fix_obs(obs)

        return obs


    def build_model(object_redshift=cosmos_dat['lp_zPDF'][dwarf_picker[gal]], fixed_metallicity=None, add_duste=True, 
                    **extras):
        """Build a prospect.models.SedModel object
        
        :param object_redshift: (optional, default: None)
            If given, produce spectra and observed frame photometry appropriate 
            for this redshift. Otherwise, the redshift will be zero.
            
        :param fixed_metallicity: (optional, default: None)
            If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
            
        :param add_duste: (optional, default: False)
            If `True`, add dust emission and associated (fixed) parameters to the model.
            
        :returns model:
            An instance of prospect.models.SedModel
        """
        from prospect.models.sedmodel import SedModel
        from prospect.models.templates import TemplateLibrary
        from prospect.models import priors

        # Get (a copy of) one of the prepackaged model set dictionaries.
        # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
        model_params = TemplateLibrary["parametric_sfh"]
        model_params.update(TemplateLibrary["agn"])
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['fagn']["isfree"] = True
        model_params['agn_tau']["isfree"] = True
        # model_params["zred"]['isfree'] = True
        model_params["logzsol"]['isfree'] = False
        model_params["duste_umin"]['isfree'] = True
        model_params["duste_qpah"]['isfree'] = True
        model_params["duste_gamma"]['isfree'] = True
        model_params["zred"]['init'] = object_redshift
        # model_params['add_agn_dust']["isfree"] = True
        
        # Let's make some changes to initial values appropriate for our objects and data
        # model_params["zred"]["init"] = 0.0
        # model_params["dust2"]["init"] = 0.05
        # model_params["logzsol"]["init"] = -0.5
        # model_params["tage"]["init"] = 13.
        model_params["mass"]["init"] = 10**cosmos_dat['lp_mass_med'][dwarf_picker[gal]]
        
        # These are dwarf galaxies, so lets also adjust the metallicity prior,
        # the tau parameter upward, and the mass prior downward
        model_params["zred"]["prior"] = priors.TopHat(mini=0.1, maxi=0.3)
        model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
        # model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=2.0)
        model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=1e2)
        model_params["tage"]["prior"] = priors.TopHat(mini=0.1,maxi=13.8)
        model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

        # If we are going to be using emcee, it is useful to provide a 
        # minimum scale for the cloud of walkers (the default is 0.1)
        # model_params["zred"]["disp_floor"] = 0.01
        model_params["mass"]["disp_floor"] = 1e5
        model_params["tau"]["disp_floor"] = 1.0
        model_params["tage"]["disp_floor"] = 1.0

        # Now instantiate the model object using this dictionary of parameter specifications
        model = SedModel(model_params)
        with open(dir_pick+"/dwarf_agn/emcee/gal_"+str(gal)+".pickle", 'wb') as handle:
            pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return model


    def build_sps(zcontinuous=1, **extras):
        """
        :param zcontinuous: 
            A vlue of 1 insures that we use interpolation between SSPs to 
            have a continuous metallicity parameter (`logzsol`)
            See python-FSPS documentation for details
        """
        from prospect.sources import CSPSpecBasis
        sps = CSPSpecBasis(zcontinuous=zcontinuous)
        return sps


    run_params = {}

    # emcee
    run_params["optimize"] = False
    run_params["emcee"] = True
    run_params["dynesty"] = False
    run_params["nmin"] = 2
    run_params["min_method"] = 'lm'
    # Number of emcee walkers
    run_params["nwalkers"] = 128
    # Number of iterations of the MCMC sampling
    run_params["niter"] = 5000
    # Number of iterations in each round of burn-in
    # After each round, the walkers are reinitialized based on the 
    # locations of the highest probablity half of the walkers.
    run_params["nburn"] = [512,1024]
    run_params["verbose"] = False
    run_params["zcontinuous"] = 1

    # #dynasty
    # run_params["dynesty"] = True
    # run_params["optmization"] = False
    # run_params["emcee"] = False
    # run_params["nested_method"] = 'rwalk'
    # run_params["nlive_init"] = 250
    # run_params["nlive_batch"] = 250
    # run_params["nested_dlogz_init"] = 0.1
    # run_params["nested_posterior_thresh"] = 0.1
    # run_params["verbose"] = True
    # run_params["print_progress"] = False
    # run_params["nested_maxcall"] = int(1e7) 

    # Here we will run all our building functions
    obs = build_obs(**run_params)
    sps = build_sps(**run_params)
    model = build_model(**run_params)


    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    # print("Done optmization in {}s".format(output["optimization"][1]))
    # print('done emcee in {0}s'.format(output["sampling"][1]))


    #File writing

    hfile = dir_pick+"/dwarf_agn/emcee/gal_"+str(gal)+"_emcee_mcmc.h5"
    writer.write_hdf5(hfile, run_params, model , obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    print('Finished')

    #File reading

    results_type = "emcee" # | "dynesty"
    # grab results (dictionary), the obs dictionary, and our corresponding models
    # When using parameter files set `dangerous=True`
    result, obs, _ = reader.results_from(dir_pick+"/dwarf_agn/emcee/gal_"+str(gal)+"_{}_mcmc.h5".format(results_type), dangerous=False)


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
    from prospect.plotting.corner import quantile
    weights = result.get("weights", None)
    post_pcts = quantile(flatchain.T, q=[0.16, 0.50, 0.84], weights=weights)

    print(post_pcts) 

    np.save(dir_pick+"/dwarf_agn/emcee/gal_"+str(gal)+"_post_pcts.npy", post_pcts)

# if __name__ == '__main__':
rand_pick = np.arange(3000, 3941)

ncpu = 64

pool = Pool(processes=ncpu)
result = pool.map(run_prospector, rand_pick)
