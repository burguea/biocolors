# This file contains dispersion formulas
# Requieres numpy
# The Si dispersion requires loading a table of values c-Si_HJY_dispersion.txt
import numpy as np

def NAM_disp(wl, ninf, wg, fj, wj, Gj):
    """ New amorphous ellipsometer dispersion formula
    
    Inputs:
        wl0 - Wavelenght in nm
        ninf, wg, fj, wj, Gj - Coefficients
    
    Output:
        n+ik - refractive index value"""
    
    B = fj/Gj*(Gj**2-(wj-wg)**2)
    C = 2*fj*Gj*(wj-wg)
    
    w = 1240/wl # in eV
    
    n = ninf+(B*(w-wj)+C)/((w-wj)**2+Gj**2)
    if w>wg:
        k= (fj*(w-wg)**2)/((w-wj)**2+Gj**2)
    else:
        k=0
    
    return n+1j*k

def hartman(wl,A,B,C):
    """ Hartman dispersion formula    
    Inputs:
        wl0 - Wavelenght in nm
        A,B,C - Coefficients
    
    Output:
        n - refractive index value (there is no extinction coef.)"""
    
    return A+C/(wl-B)

def sell_PMMA(wl, c1=1.1819, c2=0.011313):
    """ Sellmeier dispersion formula for PMMA"""
    wl1 = wl/1000
    try:
        n = np.sqrt(1+c1*wl1**2/(wl1**2-c2))
    except ValueError:
        n = 0
    return n


def Si_n(wl0):
    """ [Requieres table of values]
    Returns the complex refractive index of crystalline Si at a given wavelenth
    from tabulated data 
    
    Inputs:
        wl0- Wavelenght 
    
    Output:
        n+ik - refractive index value"""
    Si_disp = np.loadtxt(Si_disp_file, skiprows=4)
    wl, n, k = Si_disp[:, 0], Si_disp[:, 1], Si_disp[:, 2]
    return np.interp(wl0,wl,n)+1j*np.interp(wl0,wl,k)