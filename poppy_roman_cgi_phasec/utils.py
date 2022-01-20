import poppy
from poppy.poppy_core import PlaneType
from poppy import accel_math

import proper
import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import scipy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, clear_output

import os
import copy
import time
import glob
from pathlib import Path

from .import polmap

from importlib import reload
import misc

def glass_index( glass, lambda_m, cgi_dir ):
    a = np.loadtxt( str( cgi_dir/'glass'/(glass+'_index.txt') ) )  # lambda_um index pairs
    f = interp1d( a[:,0], a[:,1], kind='cubic' )
    return f( lambda_m*1e6 )

def make_inwave(cgi_dir, D, lambda_c_m, lambda_m, npix, oversample, offsets, polaxis):
    wfin = poppy.FresnelWavefront(beam_radius=D/2, wavelength=lambda_m, npix=npix, oversample=oversample)
    
    if polaxis!=0: 
        print('Employing polarization aberrations with polaxis={:d}.'.format(polaxis))
        polfile = cgi_dir/'pol'/'phasec_pol'
        polmap.polmap( wfin, str(polfile), npix, polaxis )
    else: print('Not employing polarization aberrations.')
    
    xoffset = offsets[0]
    yoffset = offsets[1]
    xoffset_lam = xoffset * (lambda_c_m / lambda_m).value # maybe use negative sign
    yoffset_lam = yoffset * (lambda_c_m / lambda_m).value 
    n = int(round(npix*oversample))
    x = np.tile( (np.arange(n)-n//2)/(npix/2.0), (n,1) )
    y = np.transpose(x)
    wfin.wavefront = wfin.wavefront * np.exp(complex(0,1) * np.pi * (xoffset_lam * x + yoffset_lam * y))
    
    misc.myimshow2(np.abs(wfin.wavefront)**2, np.angle(wfin.wavefront),
                   'Input Wave Intensity', 'Input Wave Phase',
                   pxscl=wfin.pixelscale, 
                   cmap1='gist_heat', cmap2='viridis')
    
    return wfin
    
    
    
    