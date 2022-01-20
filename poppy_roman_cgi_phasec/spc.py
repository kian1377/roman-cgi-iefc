import poppy
from poppy.poppy_core import PlaneType
from poppy import accel_math

import proper
import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import scipy
from scipy.interpolate import interp1d
import multiprocessing as mp

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

def run_single(mode='SPC730',
               wavelength=None,
                npix=1000,
                oversample=2.048,
                npsf=64,
                psf_pixelscale=13*u.micron/u.pixel,
                offsets=(0,0),
                dm1=None,
                dm2=None,
                use_fpm=True,
                use_opds=False,
                use_pupil_defocus=False,
                use_apertures=False,
                polaxis=0,
                cgi_dir=None,
                display_mode=False,
                display_inwave=False,
                display_intermediates=False,
                return_intermediates=False):
    
    diam = 2.363114*u.m
    
    #################### Initialize directories and file names for the masks and OPDs
    if mode=='SPC730':
        wavelength_c = 730e-9*u.m
        opticsdir = cgi_dir/'spc-spec'
        opddir = cgi_dir/'spc-spec-opds'
        pupil_fname = str(opticsdir/'pupil_SPC-20200617_1000.fits')
        spm_fname = str(opticsdir/'SPM_SPC-20200617_1000_rounded9_rotated.fits')
        fpm_fname = str(opticsdir/'fpm_0.05lamD.fits')
        lyotstop_fname = str(opticsdir/'LS_SPC-20200617_1000.fits')
    elif mode=='SPC825':
        wavelength_c = 825e-9*u.m
        opticsdir = cgi_dir/'spc-wide'
        opddir = cgi_dir/'spc-wide-opds'
        pupil_fname = str(opticsdir/'pupil_SPC-20200610_1000.fits')
        spm_fname = str(opticsdir/'SPM_SPC-20200610_1000_rounded9_gray_rotated.fits')
        fpm_fname = str(opticsdir/'FPM_SPC-20200610_0.1_lamc_div_D.fits')
        lyotstop_fname = str(opticsdir/'LS_SPC-20200610_1000.fits')
    
    if wavelength is None: wavelength = wavelength_c
    print('Propagating wavelength {:.3f}.'.format(wavelength.to(u.nm)))
    
    pupil = poppy.FITSOpticalElement('Roman Pupil', pupil_fname, planetype=PlaneType.pupil)
    SPM = poppy.FITSOpticalElement('SPM', spm_fname, planetype=PlaneType.pupil)
    if use_fpm: FPM = poppy.FixedSamplingImagePlaneElement('FPM', fpm_fname)
    else: FPM = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FPM Plane (No Optic)')   
    LS = poppy.FITSOpticalElement('Lyot Stop', lyotstop_fname, planetype=PlaneType.pupil)
    
    if dm1 is None: dm1 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM1 Plane (No Optic)')
    elif isinstance(dm1,str): 
        dm1 = poppy.FITSOpticalElement('DM1', opd=str(opticsdir/dm1), opdunits='meters', planetype=PlaneType.intermediate)

    if dm2 is None: dm2 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM2 Plane (No Optic)')
    elif isinstance(dm2, str):
        dm2 = poppy.FITSOpticalElement('DM2', opd=str(opticsdir/dm2), opdunits='meters', planetype=PlaneType.intermediate)
    
    if display_mode:
        misc.myimshow(pupil.amplitude, 'Roman Pupil', pxscl=pupil.pixelscale)
        if dm1_fname is not None: misc.myimshow(dm1.opd, 'DM1', pxscl=dm1.pixelscale)
        if dm2_fname is not None: misc.myimshow(dm2.opd, 'DM2', pxscl=dm2.pixelscale)
        misc.myimshow(SPM.amplitude, 'SPM', pxscl=SPM.pixelscale)
        misc.myimshow2(FPM.amplitude, FPM.opd, 'FPM Amplitude', 'FPM OPD', pxscl=FPM.pixelscale)  
        misc.myimshow(LS.amplitude, 'Lyot stop', pxscl=LS.pixelscale)    
    
    fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
    
    # this section defines various optic focal lengths, diameters, and distances between optics.
    fl_pri = 2.838279206904720*u.m
    sm_despace_m = 0*u.m # despacing of the secondary mirror
    d_pri_sec = 2.285150508110035*u.m + sm_despace_m
    fl_sec = -0.654200796568004*u.m
    diam_sec = 0.58166*u.m
    d_sec_pomafold = 2.993753469304728*u.m + sm_despace_m
    diam_pomafold = 0.09*u.m
    d_pomafold_m3 = 1.680935841598811*u.m
    fl_m3 = 0.430216463069001*u.m
    diam_m3 = 0.2*u.m
    d_m3_pupil = 0.469156807765176*u.m
    d_m3_m4 = 0.943514749358944*u.m
    fl_m4 = 0.116239114833590*u.m
    diam_m4 = 0.07*u.m
    d_m4_m5 = 0.429145636743193*u.m
    fl_m5 = 0.198821518772608*u.m
    d_m5_pupil = 0.716529242927776*u.m
    diam_m5 = 0.07*u.m
    d_m5_ttfold = 0.351125431220770*u.m
    diam_ttfold = 0.06*u.m
    d_ttfold_fsm = d_m5_pupil - d_m5_ttfold 
    if use_pupil_defocus: d_ttfold_fsm = d_ttfold_fsm + 0.033609*u.m  # 33.6 mm to put pupil 6 mm from SPC mask
    diam_fsm = 0.0508*u.m
    d_fsm_oap1 = 0.354826767220001*u.m
    fl_oap1 = 0.503331895563883*u.m
    diam_oap1 = 0.060*u.m
    focm_z_shift_m = 0*u.m
    d_oap1_focm = 0.768029932093727*u.m + focm_z_shift_m
    diam_focm = 0.035*u.m
    d_focm_oap2 = 0.314507535543064*u.m + focm_z_shift_m
    fl_oap2 = 0.579205571254990*u.m
    diam_oap2 = 0.060*u.m
    d_oap2_dm1 = 0.775857408587825*u.m
    d_dm1_dm2 = 1.0*u.m
    d_dm2_oap3 = 0.394833855161549*u.m
    fl_oap3 = 1.217276467668519*u.m
    diam_oap3 = 0.06*u.m
    d_oap3_fold3 = 0.505329955078121*u.m
    diam_fold3 = 0.06*u.m
    d_fold3_oap4 = 1.158897671642761*u.m
    fl_oap4 = 0.446951159052363*u.m
    diam_oap4 = 0.06*u.m
    d_oap4_pupilmask = 0.423013568764728*u.m
    d_pupilmask_oap5 = 0.408810704327559*u.m
    fl_oap5 = 0.548189354706822*u.m
    diam_oap5 = 0.06*u.m
    d_oap5_fpm = fl_oap5                    # to front of FPM 
    fpm_thickness = 0.006363747896388863*u.m    # account for FPM thickness (inclination included)
    fpm_index = utils.glass_index('SILICA', wavelength.value, cgi_dir)
    d_fpm_oap6 = fpm_thickness / fpm_index + 0.543766629917668*u.m     # from front of FPM
    fl_oap6 = d_fpm_oap6 #### ????????????????????????????????????????????????????????????????????????????????????
    diam_oap6 = 0.054*u.m
    d_oap6_lyotstop = 0.687476361491529*u.m
    d_oap6_exitpupil = d_oap6_lyotstop - 6e-3*u.m
    d_lyotstop_oap7 = 0.401748561745987*u.m
    fl_oap7 = 0.708251420923810*u.m
    diam_oap7 = 0.054*u.m
    d_oap7_fieldstop = fl_oap7 
    d_fieldstop_oap8 = 0.210985170345932*u.m * 0.997651
    fl_oap8 = d_fieldstop_oap8
    diam_oap8 = 0.026*u.m
    d_oap8_pupil = 0.237561587674008*u.m
    d_pupil_filter = 0.130*u.m
    d_oap8_filter = d_oap8_pupil + d_pupil_filter   # to front of filter
    diam_filter = 0.009*u.m
    filter_thickness = 0.004016105782012525*u.m      # account for filter thickness (inclination included)
    filter_index = utils.glass_index('SILICA', wavelength.value, cgi_dir)
    d_filter_lens = filter_thickness / filter_index + 0.210581269256657095*u.m  # from front of filter
    diam_lens = 0.0104*u.m
    d_lens_fold4 = 0.202432155667761*u.m
    diam_fold4 = 0.036*u.m
    d_fold4_image = 0.050000152941020161*u.m
    
    # define parameters related to final imaging lens, which is an air-gapped doublet
    lens_1_index = utils.glass_index('S-BSL7R', wavelength.value, cgi_dir)
    lens_2_index = utils.glass_index('PBM2R', wavelength.value, cgi_dir)
    r11, r12, lens_1_t = (0.10792660718579995*u.m, -0.10792660718579995*u.m, 0.003*u.m)
    r21, r22, lens_2_t = (1e10*u.m, 0.10608379812011390*u.m, 0.0025*u.m)
    air_gap = 0.0005*u.m
    
    fl_1 = 1 / ( (lens_1_index - 1) * ( 1.0/r11 - 1.0/r12 + (lens_1_index - 1)*lens_1_t / (lens_1_index*r11*r12) ) )
    d_pp_11 = -fl_1*(lens_1_index - 1)*lens_1_t / (lens_1_index*r12)
    d_pp_12 = -fl_1*(lens_1_index - 1)*lens_1_t / (lens_1_index*r11)

    fl_2 = 1 / ( (lens_2_index - 1) * ( 1.0/r21 - 1.0/r22 + (lens_2_index - 1)*lens_2_t / (lens_2_index*r21*r22) ) )
    d_pp_21 = -fl_2*(lens_2_index - 1)*lens_2_t / (lens_2_index*r22)
    d_pp_22 = -fl_2*(lens_2_index - 1)*lens_2_t / (lens_2_index*r21)

    d_filter_lens_1_pp1 = d_filter_lens + d_pp_11 
    d_lens_1_pp2_lens_2_pp1 = -d_pp_12 + air_gap + d_pp_21
    d_lens_2_pp2_fold4 = -d_pp_22 + d_lens_fold4

#     0.10490069216127479 0.000993583483544923 -0.000993583483544923
#     -0.17078291119000114 0.0015421040653161094 1.635922563452011e-14
#     0.21432764805243154 0.0030356875488610326 0.25243230860876475

    #################### Define optics
    primary = poppy.QuadraticLens(fl_pri, name='Primary')
    secondary = poppy.QuadraticLens(fl_sec, name='Secondary')
    poma_fold = poppy.CircularAperture(radius=diam_pomafold/2,name="POMA Fold")
    m3 = poppy.QuadraticLens(fl_m3, name='M3')
    m4 = poppy.QuadraticLens(fl_m4, name='M4')
    m5 = poppy.QuadraticLens(fl_m5, name='M5')
    tt_fold = poppy.CircularAperture(radius=diam_ttfold/2,name="TT Fold")
    fsm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FSM')
    oap1 = poppy.QuadraticLens(fl_oap1, name='OAP1')
    focm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FOCM')
    oap2 = poppy.QuadraticLens(fl_oap2, name='OAP2')
    oap3 = poppy.QuadraticLens(fl_oap3, name='OAP3')
    fold3 = poppy.CircularAperture(radius=diam_fold3/2,name="Fold 3")
    oap4 = poppy.QuadraticLens(fl_oap4, name='OAP4')
    oap5 = poppy.QuadraticLens(fl_oap5, name='OAP5')
    oap6 = poppy.QuadraticLens(fl_oap6, name='OAP6')
    oap7 = poppy.QuadraticLens(fl_oap7, name='OAP7')
    oap8 = poppy.QuadraticLens(fl_oap8, name='OAP8')
    filt = poppy.CircularAperture(radius=diam_filter/2, name='Filter')
    lens_1 = poppy.QuadraticLens(fl_1, name='LENS 1') # first lens of the doublet
    lens_2 = poppy.QuadraticLens(fl_2, name='LENS 2')
    fold4 = poppy.CircularAperture(radius=diam_fold4/2,name="Fold 4")
    
    if use_opds:
        primary_opd = poppy.FITSOpticalElement('Primary OPD',
                                               opd=str(opddir/'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits'), opdunits='meters', 
                                               planetype=PlaneType.intermediate)
        secondary_opd = poppy.FITSOpticalElement('Secondary OPD',
                                                 opd=str(opddir/'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits'),
                                                 opdunits='meters', planetype=PlaneType.intermediate)
        pomafold_opd = poppy.FITSOpticalElement('POMA-Fold OPD',
                                             opd=str(opddir/'roman_phasec_POMAFOLD_measured_phase_error_V1.1.fits'), opdunits='meters',
                                             planetype=PlaneType.intermediate)
        m3_opd = poppy.FITSOpticalElement('M3 OPD',
                                          opd=str(opddir/'roman_phasec_M3_measured_phase_error_V1.1.fits'), opdunits='meters',
                                          planetype=PlaneType.intermediate)
        m4_opd = poppy.FITSOpticalElement('M4 OPD',
                                          opd=str(opddir/'roman_phasec_M4_measured_phase_error_V1.1.fits'), opdunits='meters',
                                          planetype=PlaneType.intermediate)
        m5_opd = poppy.FITSOpticalElement('M5 OPD',
                                          opd=str(opddir/'roman_phasec_M5_measured_phase_error_V1.1.fits'), opdunits='meters', 
                                          planetype=PlaneType.intermediate)
        ttfold_opd = poppy.FITSOpticalElement('TT-Fold OPD',
                                             opd=str(opddir/'roman_phasec_TTFOLD_measured_phase_error_V1.1.fits'), opdunits='meters',
                                             planetype=PlaneType.intermediate)
        fsm_opd = poppy.FITSOpticalElement('FSM OPD',
                                           opd=str(opddir/'roman_phasec_LOWORDER_phase_error_V2.0.fits'), opdunits='meters', 
                                           planetype=PlaneType.intermediate)
        oap1_opd = poppy.FITSOpticalElement('OAP1 OPD',
                                            opd=str(opddir/'roman_phasec_OAP1_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        focm_opd = poppy.FITSOpticalElement('FOCM OPD',
                                            opd=str(opddir/'roman_phasec_FCM_EDU_measured_coated_phase_error_V2.0.fits'), 
                                            opdunits='meters', planetype=PlaneType.intermediate)
        oap2_opd = poppy.FITSOpticalElement('OAP2 OPD',
                                            opd=str(opddir/'roman_phasec_OAP2_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        dm1_opd = poppy.FITSOpticalElement('DM1 OPD',
                                           opd=str(opddir/'roman_phasec_DM1_phase_error_V1.0.fits'), opdunits='meters',
                                           planetype=PlaneType.intermediate)
        dm2_opd = poppy.FITSOpticalElement('DM2 OPD',
                                           opd=str(opddir/'roman_phasec_DM2_phase_error_V1.0.fits'), opdunits='meters',
                                           planetype=PlaneType.intermediate)
        oap3_opd = poppy.FITSOpticalElement('OAP3 OPD',
                                            opd=str(opddir/'roman_phasec_OAP3_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        fold3_opd = poppy.FITSOpticalElement('Fold-3 OPD',
                                             opd=str(opddir/'roman_phasec_FOLD3_FLIGHT_measured_coated_phase_error_V2.0.fits'), 
                                             opdunits='meters', planetype=PlaneType.intermediate)
        oap4_opd = poppy.FITSOpticalElement('OAP4 OPD',
                                            opd=str(opddir/'roman_phasec_OAP4_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        spm_opd = poppy.FITSOpticalElement('SPM OPD',
                                           opd=str(opddir/'roman_phasec_PUPILMASK_phase_error_V1.0.fits'), opdunits='meters',
                                           planetype=PlaneType.intermediate)
        oap5_opd = poppy.FITSOpticalElement('OAP5 OPD',
                                            opd=str(opddir/'roman_phasec_OAP5_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        oap6_opd = poppy.FITSOpticalElement('OAP6 OPD',
                                            opd=str(opddir/'roman_phasec_OAP6_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        oap7_opd = poppy.FITSOpticalElement('OAP7 OPD',
                                            opd=str(opddir/'roman_phasec_OAP7_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        oap8_opd = poppy.FITSOpticalElement('OAP8 OPD',
                                            opd=str(opddir/'roman_phasec_OAP8_phase_error_V3.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        filter_opd = poppy.FITSOpticalElement('Filter OPD',
                                              opd=str(opddir/'roman_phasec_FILTER_phase_error_V1.0.fits'), opdunits='meters',
                                              planetype=PlaneType.intermediate)
        lens_opd = poppy.FITSOpticalElement('LENS OPD',
                                            opd=str(opddir/'roman_phasec_LENS_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        
    #################### Create the optical system
    beam_ratio = 1/oversample
    fosys = poppy.FresnelOpticalSystem(name=mode, pupil_diameter=diam, npix=npix, beam_ratio=beam_ratio, verbose=True)

    fosys.add_optic(pupil)
    fosys.add_optic(primary)
    if use_opds: fosys.add_optic(primary_opd)
        
    fosys.add_optic(secondary, distance=d_pri_sec)
    if use_opds: fosys.add_optic(secondary_opd)
        
    fosys.add_optic(poma_fold, distance=d_sec_pomafold)
    if use_opds: fosys.add_optic(pomafold_opd)
        
    fosys.add_optic(m3, distance=d_pomafold_m3)
    if use_opds: fosys.add_optic(m3_opd)
        
    fosys.add_optic(m4, distance=d_m3_m4)
    if use_opds: fosys.add_optic(m4_opd)
        
    fosys.add_optic(m5, distance=d_m4_m5)
    if use_opds: fosys.add_optic(m5_opd)
        
    fosys.add_optic(tt_fold, distance=d_m5_ttfold)
    if use_opds: fosys.add_optic(ttfold_opd)
        
    fosys.add_optic(fsm, distance=d_ttfold_fsm)
    if use_opds: fosys.add_optic(fsm_opd)
        
    fosys.add_optic(oap1, distance=d_fsm_oap1)
    if use_opds: fosys.add_optic(oap1_opd)
        
    fosys.add_optic(focm, distance=d_oap1_focm)
    if use_opds: fosys.add_optic(focm_opd)
        
    fosys.add_optic(oap2, distance=d_focm_oap2)
    if use_opds: fosys.add_optic(oap2_opd)
        
    fosys.add_optic(dm1, distance=d_oap2_dm1)
    if use_opds: fosys.add_optic(dm1_opd)
        
    fosys.add_optic(dm2, distance=d_dm1_dm2)
    if use_opds: fosys.add_optic(dm2_opd)
        
    fosys.add_optic(oap3, distance=d_dm2_oap3)
    if use_opds: fosys.add_optic(oap3_opd)
        
    fosys.add_optic(fold3, distance=d_oap3_fold3)
    if use_opds: fosys.add_optic(fold3_opd)
        
    fosys.add_optic(oap4, distance=d_fold3_oap4)
    if use_opds: fosys.add_optic(oap4_opd)
        
    fosys.add_optic(SPM, distance=d_oap4_pupilmask)
    if use_opds: fosys.add_optic(spm_opd)
        
    fosys.add_optic(oap5, distance=d_pupilmask_oap5)
    if use_opds: fosys.add_optic(oap5_opd)
        
    fosys.add_optic(FPM, distance=d_oap5_fpm)
    fosys.add_optic(oap6, distance=d_fpm_oap6)
    if use_opds: fosys.add_optic(oap6_opd)
        
    fosys.add_optic(LS, distance=d_oap6_lyotstop)
    fosys.add_optic(oap7, distance=d_lyotstop_oap7)
    if use_opds: fosys.add_optic(oap7_opd)
        
    fosys.add_optic(fieldstop, distance=d_oap7_fieldstop)
    fosys.add_optic(oap8, distance=d_fieldstop_oap8)
    if use_opds: fosys.add_optic(oap8_opd)
        
    fosys.add_optic(filt, distance=d_oap8_filter)
    if use_opds: fosys.add_optic(filter_opd)
        
    fosys.add_optic(lens_1, distance=d_filter_lens_1_pp1)
    if use_opds: fosys.add_optic(lens_opd)
        
    fosys.add_optic(lens_2, distance=d_lens_1_pp2_lens_2_pp1)
    fosys.add_optic(fold4, distance=d_lens_2_pp2_fold4)
    fosys.add_detector(pixelscale=psf_pixelscale, fov_pixels=npsf, distance=d_fold4_image)

    #################### Calculate the PSF of the FresnelOpticalSystem
    start = time.time()
    wfin = utils.make_inwave(cgi_dir, diam, wavelength_c, wavelength, npix, oversample, offsets, polaxis, display_inwave) 
    
    if return_intermediates: 
        psf_hdu, wfs = fosys.calc_psf(wavelength=wavelength, inwave=wfin,
                                      display_intermediates=display_intermediates, 
                                      return_intermediates=return_intermediates)
    else:
        psf_hdu, wfs = fosys.calc_psf(wavelength=wavelength, inwave=wfin,
                                 display_intermediates=display_intermediates, 
                                 return_intermediates=return_intermediates,
                                 return_final=True)
    psf_wf = wfs[-1].wavefront
    psf_pixelscale = wfs[-1].pixelscale
    
    if psf_pixelscale_lamD is not None and npsf is not None:
        print('Resampling PSF wavefront to a pixelscale of {:.3f} lam/D.'.format(psf_pixelscale_lamD))
        print('Current PSF pixelscale is {:.3e}.'.format(psf_pixelscale))
        mag = (1/oversample) / psf_pixelscale_lamD * (wavelength/wavelength_c).value
        psf_pixelscale = psf_pixelscale / mag
        psf_wf = proper.prop_magnify( psf_wf, mag, npsf, AMP_CONSERVE=True )
        print('Resampled PSF pixelscale is {:.3e}.'.format(psf_pixelscale))
        
        wfs[-1].wavefront = psf_wf
        wfs[-1].pixelscale = psf_pixelscale
        psf_hdu[0].data = wfs[-1].intensity
        psf_hdu[0].header['PIXELSCL'] = psf_pixelscale.value
    else:
        print('User-requested PSF pixelscale not provided so no resampling performed.')
        
    print('PSF calculated in {:.2f}s'.format(time.time()-start))

    return psf_hdu, wfs
    
    
def run_hlc(params):
    
    mode, wavelength, npix, oversample, npsf, psf_pixelscale, offsets, dm1, dm2, use_fpm, use_opds, use_pupil_defocus, polaxis, cgi_dir, return_intermediates = params
    
    psf, wfs = run_single(mode=mode,
                           wavelength=wavelength,
                           npix=npix,
                           oversample=oversample,
                           npsf=npsf, 
                           psf_pixelscale=psf_pixelscale,
                           offsets=offsets,
                           dm1=dm1, 
                           dm2=dm2,
                           use_fpm=use_fpm,
                           use_opds=use_opds,
                           use_pupil_defocus=use_pupil_defocus,
                           polaxis=polaxis,
                           cgi_dir=cgi_dir,
                           display_mode=False,
                           display_inwave=False,
                           display_intermediates=False,
                           return_intermediates=return_intermediates)
    return psf, wfs
    
def run_multi(mode='SPC730',
              wavelengths=None,
                npix=1000,
                oversample=2.048,
              npsf=64,
              psf_pixelscale=13*u.micron/u.pixel,
                offsets=(0,0),
                dm1=None,
                dm2=None,
                use_fpm=True,
                use_opds=False,
                use_pupil_defocus=False,
                polaxis=0,
                cgi_dir = None,
                return_intermediates = False):
    if wavelengths is None:
        if mode=='SPC730': wavelengths = 730e-9*u.m
        elif mode=='SPC825': wavelengths = 825e-9*u.m
        
    params = []
    if isinstance(wavelengths, np.ndarray) and wavelengths.ndim==1:
        for i in range(len(wavelengths)):
            params.append((mode, wavelengths[i], npix, oversample, npsf, psf_pixelscale,
                           offsets, dm1, dm2, use_fpm, use_opds, use_pupil_defocus, polaxis, 
                           cgi_dir, return_intermediates))
    else: 
        params.append((mode, wavelengths, npix, oversample, npsf, psf_pixelscale,
                       offsets, dm1, dm2, use_fpm, use_opds, use_pupil_defocus, polaxis, 
                       cgi_dir, return_intermediates))

    ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)
    results = pool.map(run_spc, params)
    pool.close()
    pool.join()

    psfs = []
    wfs = []
    if wavelengths.ndim==1:
        for i in range(len(wavelengths)): 
            psfs.append(results[i][0][0])
            wfs.append(results[i][1])
    else:
        psfs.append(results[0][0][0])
        wfs.append(results[0][1])
    
    return psfs, wfs
            
            
       
    