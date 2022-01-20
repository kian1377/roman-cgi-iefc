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

def run_single(mode='HLC575',
               wavelength=None,
               npix=310,
               oversample=1024/310,
               npsf=64,
               psf_pixelscale=13*u.micron/u.pixel,
               offsets=(0,0),
               dm1=None,
               dm2=None,
               use_fpm=True,
               use_fieldstop=False,
               use_opds=False,
               use_pupil_defocus=False,
               polaxis=0,
               cgi_dir=None,
               display_mode=False,
               show_fpm_steps=False,
               display_intermediates=False,
               return_intermediates=False):

    reload(misc)
    reload(polmap)
    diam = 2.363114*u.m
    D = diam*npix/309
    
    #################### Initialize directories and file names for the masks and OPDs    
    opticsdir = cgi_dir/'hlc'
    opddir = cgi_dir/'hlc-opds'
    dmsdir = cgi_dir/'dms'
    
    pupil_fname = str(opticsdir/'pupil.fits')
    lyotstop_fname = str(opticsdir/'lyot_rotated.fits')
        
    wavelength_c = 575e-9*u.m
    if wavelength is None: wavelength = wavelength_c
    print('Propagating wavelength {:.3f}.'.format(wavelength.to(u.nm)))

    # find nearest available FPM wavelength that matches specified wavelength
    lam_um = wavelength.value * 1e6
    f = open( str(opticsdir/'fpm_files.txt') )
    fpm_nlam = int(f.readline())
    fpm_lams = np.zeros((fpm_nlam),dtype=float)
    for j in range(0,fpm_nlam): fpm_lams[j] = float(f.readline())*1e-6
    fpm_root_fnames = [j.strip() for j in f.readlines()] 
    f.close()

    diff = np.abs(fpm_lams - wavelength.value)
    w = np.argmin( diff )
    if diff[w] > 0.1e-9: 
        raise Exception('Only wavelengths within 0.1nm of avalable FPM wavelengths can be used. Closest available to requested wavelength is {}.'.format(fpm_lams[w]))
    fpm_rootname = opticsdir/fpm_root_fnames[w]

    fpm_r_fname = str(fpm_rootname)+'real.fits'
    fpm_i_fname = str(fpm_rootname)+'imag.fits'
            
    #################### Initialize mode specific optics 
    pupil = poppy.FITSOpticalElement('Roman Pupil', pupil_fname, planetype=PlaneType.pupil)
    SPM = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='SPM Plane (No Optic)')
    LS = poppy.FITSOpticalElement('Lyot Stop', lyotstop_fname, planetype=PlaneType.pupil)

    fpm_r = fits.getdata(fpm_r_fname)
    fpm_i = fits.getdata(fpm_i_fname)
    fpm_phasor = fpm_r + 1j*fpm_i
    fpm_mask = (fpm_r != fpm_r[0,0]).astype(int)
    fpm_ref_wavelength = fits.getheader(fpm_r_fname)['WAVELENC']
    fpm_pxscl_lamD = fits.getheader(fpm_r_fname)['PIXSCLLD']
    FPM = poppy.FixedSamplingImagePlaneElement('COMPLEX OCCULTER', fpm_r_fname)
    FPM.amplitude = np.abs(fpm_phasor)
    FPM.opd = np.angle(fpm_phasor)*fpm_ref_wavelength/(2*np.pi)
    
    FPM_trans = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FPM Plane (No Optic)')
    
    if use_fieldstop: 
        radius = 9.7 / (309/(npix*oversample)) * (wavelength_c/wavelength) * 7.229503001768824e-06*u.m
        fieldstop = poppy.CircularAperture(radius=radius, name='HLC Field Stop')
    else: fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
    
    if dm1 is None: dm1 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM1 Plane (No Optic)')
    elif isinstance(dm1,str): 
        dm1 = poppy.FITSOpticalElement('DM1', opd=str(dmsdir/dm1), opdunits='meters', planetype=PlaneType.intermediate)
        dm1.opd = 2*dm1.opd

    if dm2 is None: dm2 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM2 Plane (No Optic)')
    elif isinstance(dm2, str):
        dm2 = poppy.FITSOpticalElement('DM2', opd=str(dmsdir/dm2), opdunits='meters', planetype=PlaneType.intermediate)
        dm2.opd = 2*dm2.opd
    
    if display_mode:
        misc.myimshow(pupil.amplitude, 'Roman Pupil', pxscl=pupil.pixelscale)
        if dm1 is not None and isinstance(dm1, poppy.ScalarTransmission)==False: 
            misc.myimshow(dm1.opd, 'DM1', pxscl=dm1.pixelscale)
        if dm2 is not None and isinstance(dm2, poppy.ScalarTransmission)==False: 
            misc.myimshow(dm2.opd, 'DM2', pxscl=dm2.pixelscale)
        misc.myimshow2(FPM.amplitude, FPM.opd, 'FPM Amplitude', 'FPM OPD', pxscl=FPM.pixelscale)
        misc.myimshow(LS.amplitude, 'Lyot stop', pxscl=LS.pixelscale)    
    
    # this section defines various optic focal lengths, diameters, and distances between optics.
    fl_pri = 2.838279206904720*u.m
#     fl_pri = 2.838279325*u.m
    sm_despace_m = 0*u.m
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
    fl_oap6 = d_fpm_oap6
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
#     if use_pupil_lens != 0: d_lens_fold4 = d_lens_fold4 - 0.0002*u.m # from back of pupil imaging lens
#     elif use_defocus_lens != 0: d_lens_fold4 = d_lens_fold4 + 0.001*u.m # doublet is 1 mm longer than singlet, so make up for it
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
    poma_fold = poppy.CircularAperture(radius=diam_pomafold/2,name="POMA_Fold")
    m3 = poppy.QuadraticLens(fl_m3, name='M3')
    m4 = poppy.QuadraticLens(fl_m4, name='M4')
    m5 = poppy.QuadraticLens(fl_m5, name='M5')
    tt_fold = poppy.CircularAperture(radius=diam_ttfold/2,name="TT_Fold")
    fsm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FSM')
    oap1 = poppy.QuadraticLens(fl_oap1, name='OAP1')
    focm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FOCM')
    oap2 = poppy.QuadraticLens(fl_oap2, name='OAP2')
    oap3 = poppy.QuadraticLens(fl_oap3, name='OAP3')
    fold3 = poppy.CircularAperture(radius=diam_fold3/2,name="Fold3")
    oap4 = poppy.QuadraticLens(fl_oap4, name='OAP4')
    oap5 = poppy.QuadraticLens(fl_oap5, name='OAP5')
    oap6 = poppy.QuadraticLens(fl_oap6, name='OAP6')
    oap7 = poppy.QuadraticLens(fl_oap7, name='OAP7')
    oap8 = poppy.QuadraticLens(fl_oap8, name='OAP8')
    filt = poppy.CircularAperture(radius=diam_filter/2, name='Filter')
    lens_1 = poppy.QuadraticLens(fl_1, name='LENS 1') # first lens of the doublet
    lens_2 = poppy.QuadraticLens(fl_2, name='LENS 2')
    fold4 = poppy.CircularAperture(radius=diam_fold4/2,name="Fold4")
    
    if use_opds:
        primary_opd = poppy.FITSOpticalElement('Primary OPD',
                                               opd=str(opddir/'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits'), opdunits='meters', 
                                               planetype=PlaneType.intermediate)
        secondary_opd = poppy.FITSOpticalElement('Secondary OPD',
                                                 opd=str(opddir/'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits'),
                                                 opdunits='meters', planetype=PlaneType.intermediate)
        poma_fold_opd = poppy.FITSOpticalElement('POMA-Fold OPD',
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
        tt_fold_opd = poppy.FITSOpticalElement('TT-Fold OPD',
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
        pupil_fold_opd = poppy.FITSOpticalElement('SPM OPD',
                                           opd=str(opddir/'roman_phasec_PUPILFOLD_phase_error_V1.0.fits'), opdunits='meters',
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
        
    #################### Create the first part of the optical system
    beam_ratio = 1/oversample
    fosys1 = poppy.FresnelOpticalSystem(name='HLC Part 1', pupil_diameter=D, npix=npix, beam_ratio=beam_ratio, verbose=True)

    fosys1.add_optic(pupil)
    fosys1.add_optic(primary)
    if use_opds: fosys1.add_optic(primary_opd)
        
    fosys1.add_optic(secondary, distance=d_pri_sec)
    if use_opds: fosys1.add_optic(secondary_opd)
        
    fosys1.add_optic(poma_fold, distance=d_sec_pomafold)
    if use_opds: fosys1.add_optic(poma_fold_opd)
        
    fosys1.add_optic(m3, distance=d_pomafold_m3)
    if use_opds: fosys1.add_optic(m3_opd)
        
    fosys1.add_optic(m4, distance=d_m3_m4)
    if use_opds: fosys1.add_optic(m4_opd)
        
    fosys1.add_optic(m5, distance=d_m4_m5)
    if use_opds: fosys1.add_optic(m5_opd)
        
    fosys1.add_optic(tt_fold, distance=d_m5_ttfold)
    if use_opds: fosys1.add_optic(tt_fold_opd)
        
    fosys1.add_optic(fsm, distance=d_ttfold_fsm)
    if use_opds: fosys1.add_optic(fsm_opd)
        
    fosys1.add_optic(oap1, distance=d_fsm_oap1)
    if use_opds: fosys1.add_optic(oap1_opd)
        
    fosys1.add_optic(focm, distance=d_oap1_focm)
    if use_opds: fosys1.add_optic(focm_opd)
        
    fosys1.add_optic(oap2, distance=d_focm_oap2)
    if use_opds: fosys1.add_optic(oap2_opd)
        
    fosys1.add_optic(dm1, distance=d_oap2_dm1)
    if use_opds: fosys1.add_optic(dm1_opd)
        
    fosys1.add_optic(dm2, distance=d_dm1_dm2)
    if use_opds: fosys1.add_optic(dm2_opd)
        
    fosys1.add_optic(oap3, distance=d_dm2_oap3)
    if use_opds: fosys1.add_optic(oap3_opd)
        
    fosys1.add_optic(fold3, distance=d_oap3_fold3)
    if use_opds: fosys1.add_optic(fold3_opd)
        
    fosys1.add_optic(oap4, distance=d_fold3_oap4)
    if use_opds: fosys1.add_optic(oap4_opd)
        
    fosys1.add_optic(SPM, distance=d_oap4_pupilmask)
    if use_opds: fosys1.add_optic(pupil_fold_opd)
        
    fosys1.add_optic(oap5, distance=d_pupilmask_oap5)
    if use_opds: fosys1.add_optic(oap5_opd)
        
    fosys1.add_optic(FPM_trans, distance=d_oap5_fpm)
    
    #################### Create second part of the optical system
    fosys2 = poppy.FresnelOpticalSystem(name='HLC part 2', npix=npix, beam_ratio=beam_ratio, verbose=True)
    
    fosys2.add_optic(FPM_trans)
    fosys2.add_optic(oap6, distance=d_fpm_oap6)
    if use_opds: fosys2.add_optic(oap6_opd)
        
    fosys2.add_optic(LS, distance=d_oap6_lyotstop)
    
    fosys2.add_optic(oap7, distance=d_lyotstop_oap7)
    if use_opds: fosys2.add_optic(oap7_opd)
        
    fosys2.add_optic(fieldstop, distance=d_oap7_fieldstop)
    
    fosys2.add_optic(oap8, distance=d_fieldstop_oap8)
    if use_opds: fosys2.add_optic(oap8_opd)
        
    fosys2.add_optic(filt, distance=d_oap8_filter)
    if use_opds: fosys2.add_optic(filter_opd)
        
    fosys2.add_optic(lens_1, distance=d_filter_lens_1_pp1)
    if use_opds: fosys2.add_optic(lens_opd)
        
    fosys2.add_optic(lens_2, distance=d_lens_1_pp2_lens_2_pp1)
    
    fosys2.add_optic(fold4, distance=d_lens_2_pp2_fold4)
    
    fosys2.add_detector(pixelscale=psf_pixelscale, fov_pixels=npsf, distance=d_fold4_image)
    
    # calculate a psf from the first optical system to retrieve the final wavefront at the FPM plane 
    start = time.time()
    wfin1 = utils.make_inwave(cgi_dir, D, wavelength_c, wavelength, npix, oversample, offsets, polaxis)
    psf1_hdu, wfs1 = fosys1.calc_psf(wavelength=wavelength, inwave=wfin1,
                                     display_intermediates=display_intermediates, 
                                     return_final=True, return_intermediates=return_intermediates)
    
    # perform unique MFT procedure at this FPM plane to apply the FPM data
    wfin2 = copy.deepcopy(wfs1[-1])
    if show_fpm_steps: misc.myimshow2(np.abs(wfin2.wavefront), np.angle(wfin2.wavefront), 'wavefront at FPM', npix=64)
    
    fpm_pxscl_lamD = ( FPM.pixelscale_lamD * FPM.wavelength_c.to(u.meter) / wfin2.wavelength.to(u.meter) ).value
    nfpm = fpm_phasor.shape[0]
    n = wfin2.wavefront.shape[0]
    nfpmlamD = nfpm*fpm_pxscl_lamD*wfin2.oversample
    
    # use MFTs to use super-sampled FPM
#     wavefront0 = ffts( wavefront0, 1 )              # to virtual pupil
#     wavefront0 *= fpm_array[0,0]                    # apply amplitude & phase from FPM clear area
#     nfpm = fpm_array.shape[0]
#     fpm_sampling_lamdivD = fpm_sampling_lam0divD * fpm_lam0_m / lambda_m    # FPM sampling at current wavelength in lambda_m/D
#     wavefront_fpm = mft2(wavefront0, fpm_sampling_lamdivD, pupil_diam_pix, nfpm, +1)   # MFT to highly-sampled focal plane
#     wavefront_fpm *= fpm_mask * (fpm_array - 1)      # subtract field inside FPM region, add in FPM-multiplied region
#     wavefront_fpm = mft2(wavefront_fpm, fpm_sampling_lamdivD, pupil_diam_pix, n, -1)        # MFT back to virtual pupil
#     wavefront0 += wavefront_fpm
#     wavefront_fpm = 0
#     wavefront0 = ffts( wavefront0, -1 )     # back to normally-sampled focal plane to continue propagation
#     wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
#     wavefront0 = 0
    
    wfin2.wavefront = accel_math._ifftshift(wfin2.wavefront)
    wfin2.wavefront = accel_math.fft_2d(wfin2.wavefront, forward=False, fftshift=True) # do a forward FFT to virtual pupil
    if show_fpm_steps: misc.myimshow2(np.abs(wfin2.wavefront), np.angle(wfin2.wavefront), 'after FFT to virtual pupil', npix=309)
    
    wfin2.wavefront *= fpm_phasor[0,0]
    if show_fpm_steps: misc.myimshow2(np.abs(wfin2.wavefront), np.angle(wfin2.wavefront), 'after applying FPM clear area in pupil', npix=309)
    
    mft = poppy.matrixDFT.MatrixFourierTransform(centering='ADJUSTABLE')
    wavefront_fpm = mft.perform(wfin2.wavefront, nfpmlamD, nfpm) # MFT back to highly sampled focal plane
    if show_fpm_steps: misc.myimshow2(np.abs(wavefront_fpm), np.angle(wavefront_fpm), 'after MFT back to FP')
    
    wavefront_fpm *= fpm_mask * (fpm_phasor - 1)
    if show_fpm_steps: misc.myimshow2(np.abs(wavefront_fpm), np.angle(wavefront_fpm), 'after applying FPM data')
    
    mft = poppy.matrixDFT.MatrixFourierTransform(centering='ADJUSTABLE')
    wavefront_fpm = mft.inverse(wavefront_fpm, nfpmlamD, n) # MFT to virtual pupil
    if show_fpm_steps: misc.myimshow2(np.abs(wavefront_fpm), np.angle(wavefront_fpm), 'after MFT to virtual pupil')
    
    wfin2.wavefront += wavefront_fpm
    if show_fpm_steps: misc.myimshow2(np.abs(wfin2.wavefront), np.angle(wfin2.wavefront), 'after adding FPM wavefront')
    
    wfin2.wavefront = accel_math.fft_2d(wfin2.wavefront, forward=True, fftshift=True) # FFT back to normally-sampled focal plane
    wfin2.wavefront = accel_math._fftshift(wfin2.wavefront)
    if show_fpm_steps: misc.myimshow2(np.abs(wfin2.wavefront), np.angle(wfin2.wavefront), 'after FFT to FP', npix=64)
    
    psf2_hdu, wfs2 = fosys2.calc_psf(wavelength=wavelength, inwave=wfin2, normalize='none',
                                     display_intermediates=display_intermediates, 
                                     return_final=True, return_intermediates=return_intermediates,)
    psf = wfs2[-1].wavefront
    psf_pixelscale = wfs2[-1].pixelscale.to(u.mm/u.pix)
    wfs1[-1] = wfin2 # set the FPM wavefront to be that of the wave after FPM is applied
    wfs1.pop(-1) # remove the final wavefront from wfs1
    wfs1.extend(wfs2) # add the second set of wfs to a single list
    
    psf_wf = wfs1[-1].wavefront
    psf_pixelscale = wfs1[-1].pixelscale
        
    print('PSF calculated in {:.2f}s'.format(time.time()-start))

    return psf2_hdu, wfs1


def run_hlc(params):
    
    mode, wavelength, npix, oversample, npsf, psf_pixelscale, offsets, dm1, dm2, use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, cgi_dir, return_intermediates = params
    
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
                          use_fieldstop=use_fieldstop,
                           use_opds=use_opds,
                           use_pupil_defocus=use_pupil_defocus,
                           polaxis=polaxis,
                           cgi_dir=cgi_dir,
                           display_mode=False,
                           display_inwave=False,
                           display_intermediates=False,
                           return_intermediates=return_intermediates)
    return psf, wfs
    
def run_multi(mode='HLC575',
              wavelength=None,
                npix=310,
                oversample=1024/310,
              npsf=64,
              psf_pixelscale=13*u.micron/u.pixel,
                offsets=(0,0),
                dm1=None,
                dm2=None,
                use_fpm=True,
              use_fieldstop=True,
                use_opds=False,
                use_pupil_defocus=False,
                polaxis=0,
                cgi_dir=None,
                return_intermediates = False):

    multi_param=None
    params = []

    if isinstance(wavelength, np.ndarray) or isinstance(wavelength, list): 
        multi_param = wavelength
        for i in range(len(wavelength)):
            params.append((mode, wavelength[i], npix, oversample, npsf, psf_pixelscale,
                           offsets, dm1, dm2, use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, 
                           cgi_dir, return_intermediates))
    elif isinstance(dm1, list) and isinstance(dm2, list):
        multi_param = dm1
        if len(dm1)==len(dm2):
            for i in range(len(dm1)):
                params.append((mode, wavelength, npix, oversample, npsf, psf_pixelscale,
                               offsets, dm1[i], dm2[i], use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, 
                               cgi_dir, return_intermediates))
        else: print('The length of the dm1 list must match the length of the dm2 list.')
    else: 
        params.append((mode, wavelength, npix, oversample, npsf, psf_pixelscale,
                       offsets, dm1, dm2, 
                       use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, 
                       cgi_dir, return_intermediates))

    ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)
    results = pool.map(run_hlc, params)
    pool.close()
    pool.join()

    psfs = []
    wfs = []
    if multi_param is not None:
        for i in range(len(multi_param)): 
            psfs.append(results[i][0][0])
            wfs.append(results[i][1])
    else:
        psfs.append(results[0][0][0])
        wfs.append(results[0][1])
    
    return psfs, wfs

