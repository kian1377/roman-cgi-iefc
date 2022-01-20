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
from importlib import reload

from .import utils
from .import polmap
from .import misc

def run(mode='SPC730',
        npix=1000,
        oversample=2.048,
        npsf=None,
        pixelscale_lamD=None,
        lambda_m=None,
        offsets=(0,0),
        dm1_fname=None,
        dm2_fname=None,
        use_fpm=True,
        polaxis=0,
        cgi_dir=None,
        display_mode=False,
        display_inwave=False,
        display_intermediates=False,
        display_fpm=False,
        display_psf=True):
    
    diam = 2.363114*u.m
    
    #################### Initialize directories and file names for the masks and OPDs
    if mode=='SPC730':
        lambda_c_m = 730e-9*u.m
        opticsdir = cgi_dir/'spc-spec-compact'
        opddir = cgi_dir/'spc-spec-opds'
        pupil_fname = str(opticsdir/'pupil_SPC-20200617_1000.fits')
        spm_fname = str(opticsdir/'SPM_SPC-20200617_1000_rounded9.fits')
        fpm_fname = str(opticsdir/'fpm_0.05lamD.fits')
        lyotstop_fname = str(opticsdir/'LS_SPC-20200617_1000.fits')
    elif mode=='SPC825':
        lambda_c_m = 825e-9*u.m
        opticsdir = cgi_dir/'spc-wide'
        opddir = cgi_dir/'spc-wide-opds'
        pupil_fname = str(opticsdir/'pupil_SPC-20200610_1000.fits')
        spm_fname = str(opticsdir/'SPM_SPC-20200610_1000_rounded9_gray_rotated.fits')
        fpm_fname = str(opticsdir/'FPM_SPC-20200610_0.1_lamc_div_D.fits')
        lyotstop_fname = str(opticsdir/'LS_SPC-20200610_1000.fits')
    
    if lambda_m is None: lambda_m = lambda_c_m
    print(lambda_m)
    
    pupil = poppy.FITSOpticalElement('Roman Pupil', pupil_fname, planetype=PlaneType.pupil)
    SPM = poppy.FITSOpticalElement('SPM', spm_fname, planetype=PlaneType.pupil)
    if use_fpm: 
#         FPM = poppy.FixedSamplingImagePlaneElement('FPM', fpm_fname)
        FPM = poppy.FITSOpticalElement('FPM', fpm_fname)
    else: FPM = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FPM Plane (No Optic)')   
    LS = poppy.FITSOpticalElement('Lyot Stop', lyotstop_fname, planetype=PlaneType.pupil)
    
    if dm1_fname is None: dm1 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM1 Plane (No Optic)')
    else: dm1 = poppy.FITSOpticalElement('DM1', opd=str(opticsdir/dm1_fname), opdunits='meters', planetype=PlaneType.intermediate)
    if dm2_fname is None: dm2 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM2 Plane (No Optic)')
    else: dm2 = poppy.FITSOpticalElement('DM2', opd=str(opticsdir/dm2_fname), opdunits='meters', planetype=PlaneType.intermediate)
    
    misc.myimshow(pupil.amplitude, 'Roman Pupil', pxscl=pupil.pixelscale)
    if dm1_fname is not None: misc.myimshow(dm1.opd, 'DM1', pxscl=dm1.pixelscale)
    if dm2_fname is not None: misc.myimshow(dm2.opd, 'DM2', pxscl=dm2.pixelscale)
    misc.myimshow(SPM.amplitude, 'SPM', pxscl=SPM.pixelscale)
    if use_fpm: misc.myimshow2(FPM.amplitude, FPM.opd, 'FPM Amplitude', 'FPM OPD', pxscl=FPM.pixelscale)  
    misc.myimshow(LS.amplitude, 'Lyot stop', pxscl=LS.pixelscale)    
    
    fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
        
    #################### Create the optical system
    osys = poppy.OpticalSystem(npix=npix, oversample=oversample)
    osys.add_pupil(pupil)
    osys.add_pupil(dm1)
    osys.add_pupil(dm2)
    osys.add_pupil(SPM)
    osys.add_image(FPM)
    osys.add_pupil(LS)
    osys.add_detector(pixelscale=0.01, fov_arcsec=5)
    
#     optsys = poppy.OpticalSystem()
#     optsys.add_pupil( poppy.CircularAperture( radius=3, pad_factor=1.5)) #pad display area by 50%
#     optsys.add_pupil( poppy.FQPM_FFT_aligner())   # ensure the PSF is centered on the FQPM cross hairs
#     optsys.add_image()  # empty image plane for "before the mask"
#     optsys.add_image( poppy.IdealFQPM(wavelength=2e-6))
#     optsys.add_pupil( poppy.FQPM_FFT_aligner(direction='backward'))  # undo the alignment tilt after going back to the pupil plane
#     optsys.add_pupil( poppy.CircularAperture( radius=3)) # Lyot mask - change radius if desired
#     optsys.add_detector(pixelscale=0.01, fov_arcsec=10.0)

    #################### Calculate the PSF of the FresnelOpticalSystem
    start = time.time()
    wfin = utils.make_inwave(cgi_dir, diam, lambda_c_m, lambda_m, npix, oversample, offsets, polaxis, display_inwave) 
    
    psf_hdu, wfs = osys.calc_psf(wavelength=lambda_m, 
#                                  inwave=wfin,
                                 display_intermediates=display_intermediates, 
                                 return_intermediates=True)
    psf = wfs[-1].wavefront
    psf_pixelscale = wfs[-1].pixelscale
    print('PSF calculated in {:.2f}s'.format(time.time()-start))
    
    #################### Display options
    if display_fpm:
        fpmnum = 0
        print('FPM pixelscale: ', wfs[fpmnum].pixelscale)
        misc.myimshow2(np.abs(wfs[fpmnum].wavefront)**2, np.angle(wfs[fpmnum].wavefront),
                       'FPM Intensity', 'FPM Phase',
                       npix=64, lognorm1=True, cmap1='gist_heat', cmap2='viridis',
                       pxscl=wfs[fpmnum].pixelscale.to(u.mm/u.pix))
    
    if pixelscale_lamD is not None:
        print('Resampling PSF wavefront to a pixelscale of {:.3f} lam/D.'.format(pixelscale_lamD))
        if mode=='SPC730': pixelscale = 3.7584889e-6/0.1 * pixelscale_lamD
        else: pixelscale = 4.252458802590218e-06/0.1 * pixelscale_lamD
        psf, mag = utils.resample_psf(wfs[-1], pixelscale, npsf)
        psf_pixelscale /= mag
        
    if display_psf: 
        print(psf.shape, psf_pixelscale)
        misc.myimshow2(np.abs(psf)**2, np.angle(psf),
                       'PSF Intensity', 'PSF Phase',
                       npix=npsf, cmap1='gist_heat', cmap2='viridis',lognorm1=True, 
                       pxscl=psf_pixelscale)

    return psf, wfs
    
    
    
    