# Requires loading the following:
# lams - wavelength vector np array
# CMFs - Color matching functions

import numpy as np
from tmm.tmm_core import coh_tmm
# CMFs=r'./CCD_CMFs_ld.npy'

# CMFS = np.load(CMFs)
# lams = CMFS[:,0]

### Dispersion formulas ###




### TMM functions ###

def get_ref_spectra_d_ndisp(wavs, d, disp):
    """Computes the reflection spectra for 's' polarized light using TMM at a
    multiple wavelengths for single layer of material with constant refractive index on Si

    Inputs: [Loaded Silicon dispersion table]
        wavs [array] - Vector of wavelengths in nm
        d [float] - thickness [nm]
        disp [array] - refractive index values at each wavelength, must match wavs vector
    Output:
        coh_tmm reflection [array] - Amplitude value"""
    
 
    reflection=np.zeros(wavs.shape)
    d_list = [np.inf, d, 500000, np.inf] # Air, layer, Si, Air
    for q,n in enumerate(disp):
        wl0=wavs[q]
        nk_list = [1, n, Si_n(wl0), 1]
        reflection[q]=coh_tmm('s', nk_list, d_list, 0, wl0)['R']
    return reflection

def get_ref_spectra_dlst_nlst(wavs, dlst, nlst):
    """ Compute the reflection spectra for series of thin films of constant refractive
    index and thickness. Upper layer first, substrate last.
    
    Inputs
        wavs [array] - Vector of wavelengths in nm
        dlst [array] - Vector of film thicknesses [nm]
        nlst [array] - Vector of film refractive indeces
    Output
        coh_tmm reflection [array] - Reflection spectra """
    reflection=np.zeros(wavs.shape)
    for q, wav in enumerate(wavs):
        reflection[q]=coh_tmm('s',nlst,dlst,0,wav)['R']

def get_refl(wl0, d_cell, n_cell, d_air, d_Pd):
    """ Reflection at single wavelength"""
    nk_list = [1, n_cell, n_alumina(wl0), n_air(wl0), n_Pd(wl0), n_Au(wl0), 1]
    d_list = [np.inf, d_cell, 0, d_air, d_Pd, 20000, np.inf]
    return coh_tmm('s', nk_list, d_list, 0, wl0)['R']


def spec2rgb_d_ndisp(d, disp):
    """ Compute RGB values for a simulated reflection spectra of a single thin film on Si      
        Inputs:
            d [float] - Thickness of the film [nm]
            disp [array] - Refractive index values of the film over the 'lams' vector"""
    spectra = get_ref_spectra_d_ndisp(lams,d,disp)
    rgb_pred = np.array([np.trapz(spectra*RCMF),np.trapz(spectra*GCMF),np.trapz(spectra*BCMF)])
    return rgb_pred

def spec2rgb_d_n(lams,d,n):
    """ Compute RGB values for a simulated spectra
        Inputs
            d - Thickness in nm
            n - constant refractive index value"""
    spectra = get_ref_spectra_d_ndisp(lams, d, np.ones(lams.shape)*n)
    rgb_pred = np.array([np.trapz(spectra*RCMF),np.trapz(spectra*GCMF),np.trapz(spectra*BCMF)])
    return rgb_pred

def mse(vecA,vecB):
    """ Computes the mean square error between two vectors"""
    return 1/vecA.size*np.sum((vecA-vecB)**2)