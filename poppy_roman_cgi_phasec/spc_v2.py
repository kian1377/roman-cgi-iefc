import poppy
from poppy.poppy_core import PlaneType
import copy

import numpy as np
import astropy.units as u

try:
    import cupy as cp
except:
    pass

import proper
from roman_phasec_proper import mft2, ffts, trim

def run(SPC):
    print('Running V2')
    # Define various optic focal lengths, diameters, and distances between optics.
    fl_pri = 2.838279206904720*u.m
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
    if SPC.use_pupil_defocus: d_ttfold_fsm = d_ttfold_fsm + 0.033609*u.m  # 33.6 mm to put pupil 6 mm from SPC mask
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
    fpm_index = SPC.glass_index('SILICA')
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
    filter_index = SPC.glass_index('SILICA')
    d_filter_lens = filter_thickness / filter_index + 0.210581269256657095*u.m  # from front of filter
    diam_lens = 0.0104*u.m
    d_lens_fold4 = 0.202432155667761*u.m
#     if use_pupil_lens != 0: d_lens_fold4 = d_lens_fold4 - 0.0002*u.m # from back of pupil imaging lens
#     elif use_defocus_lens != 0: d_lens_fold4 = d_lens_fold4 + 0.001*u.m # doublet is 1 mm longer than singlet, so make up for it
    diam_fold4 = 0.036*u.m
    d_fold4_image = 0.050000152941020161*u.m
    
    # Define parameters related to final imaging lens, which is an air-gapped doublet
    lens_1_index = SPC.glass_index('S-BSL7R')
    lens_2_index = SPC.glass_index('PBM2R')
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

    # Define optical elements
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
    
    # Create the optical system
    fosys1 = poppy.FresnelOpticalSystem(name='SPC Pre-FPM', pupil_diameter=SPC.D, 
                                        npix=SPC.npix, beam_ratio=1/SPC.oversample, verbose=True)

    fosys1.add_optic(SPC.PUPIL)
    fosys1.add_optic(primary)
    if SPC.use_opds: fosys.add_optic(SPC.primary_opd)
        
    fosys1.add_optic(secondary, distance=d_pri_sec)
    if SPC.use_opds: fosys.add_optic(SPC.secondary_opd)
        
    fosys1.add_optic(poma_fold, distance=d_sec_pomafold)
    if SPC.use_opds: fosys.add_optic(SPC.poma_fold_opd)
        
    fosys1.add_optic(m3, distance=d_pomafold_m3)
    if SPC.use_opds: fosys.add_optic(SPC.m3_opd)
        
    fosys1.add_optic(m4, distance=d_m3_m4)
    if SPC.use_opds: fosys.add_optic(SPC.m4_opd)
        
    fosys1.add_optic(m5, distance=d_m4_m5)
    if SPC.use_opds: fosys.add_optic(SPC.m5_opd)
        
    fosys1.add_optic(tt_fold, distance=d_m5_ttfold)
    if SPC.use_opds: fosys.add_optic(SPC.tt_fold_opd)
        
    fosys1.add_optic(fsm, distance=d_ttfold_fsm)
    if SPC.use_opds: fosys.add_optic(SPC.fsm_opd)
        
    fosys1.add_optic(oap1, distance=d_fsm_oap1)
    if SPC.use_opds: fosys.add_optic(SPC.oap1_opd)
        
    fosys1.add_optic(focm, distance=d_oap1_focm)
    if SPC.use_opds: fosys.add_optic(SPC.focm_opd)
        
    fosys1.add_optic(oap2, distance=d_focm_oap2)
    if SPC.use_opds: fosys.add_optic(SPC.oap2_opd)
        
    fosys1.add_optic(SPC.DM1, distance=d_oap2_dm1)
    if SPC.use_opds: fosys.add_optic(SPC.dm1_opd)
        
    fosys1.add_optic(SPC.DM2, distance=d_dm1_dm2)
    if SPC.use_opds: fosys.add_optic(SPC.dm2_opd)
        
    fosys1.add_optic(oap3, distance=d_dm2_oap3)
    if SPC.use_opds: fosys.add_optic(SPC.oap3_opd)
        
    fosys1.add_optic(fold3, distance=d_oap3_fold3)
    if SPC.use_opds: fosys.add_optic(SPC.fold3_opd)
        
    fosys1.add_optic(oap4, distance=d_fold3_oap4)
    if SPC.use_opds: fosys.add_optic(SPC.oap4_opd)
        
    fosys1.add_optic(SPC.SPM, distance=d_oap4_pupilmask)
    if SPC.use_opds: fosys.add_optic(SPC.pupil_mask_opd)
        
    fosys1.add_optic(oap5, distance=d_pupilmask_oap5)
    if SPC.use_opds: fosys.add_optic(SPC.oap5_opd)
    
    fosys1.add_optic(SPC.FPM_plane, distance=d_oap5_fpm)
    
    fosys2 = poppy.FresnelOpticalSystem(name='SPC Post-FPM', npix=SPC.npix, beam_ratio=1/SPC.oversample, verbose=True)
    
    fosys2.add_optic(SPC.FPM_plane)
    fosys2.add_optic(oap6, distance=d_fpm_oap6)
    if SPC.use_opds: fosys.add_optic(SPC.oap6_opd)
        
    fosys2.add_optic(SPC.LS, distance=d_oap6_lyotstop)
    fosys2.add_optic(oap7, distance=d_lyotstop_oap7)
    if SPC.use_opds: fosys.add_optic(SPC.oap7_opd)
        
    fosys2.add_optic(SPC.fieldstop, distance=d_oap7_fieldstop)
    fosys2.add_optic(oap8, distance=d_fieldstop_oap8)
    if SPC.use_opds: fosys.add_optic(SPC.oap8_opd)
        
    fosys2.add_optic(filt, distance=d_oap8_filter)
    if SPC.use_opds: fosys.add_optic(SPC.filter_opd)
        
    fosys2.add_optic(lens_1, distance=d_filter_lens_1_pp1)
    if SPC.use_opds: fosys.add_optic(SPC.lens_opd)
        
    fosys2.add_optic(lens_2, distance=d_lens_1_pp2_lens_2_pp1)
    fosys2.add_optic(fold4, distance=d_lens_2_pp2_fold4)
    
    fosys2.add_optic(SPC.detector, distance=d_fold4_image)

    # Calculate a psf from the first optical system to retrieve the final wavefront at the FPM plane 
    fpm_hdu, wfs_to_fpm = fosys1.calc_psf(wavelength=SPC.wavelength, inwave=SPC.inwave, 
                                          return_final=True, return_intermediates=SPC.return_intermediates)
    inwave2 = copy.deepcopy(wfs_to_fpm[-1]) # copy Wavefront object for use in the post FPM system
    
    if SPC.use_fpm: 
        # super-sample FPM
#         wavefront0 = proper.prop_get_wavefront(wavefront)
#         wavefront0 = ffts( wavefront0, 1 )                # to virtual pupil
#         wavefront0 = trim(wavefront0, n_mft)
#         if np.isscalar(fpm_array):
#             fpm_array = proper.prop_fits_read( fpm_file )
#             fpm_sampling_lamdivD = fpm_sampling_lam0divD * fpm_lam0_m / lambda_m     # FPM sampling at current wavelength in lam/D
#         else:
#             m_per_lamdivD = proper.prop_get_sampling(wavefront) / (pupil_diam_pix / n)
#             fpm_sampling_lamdivD = fpm_array_sampling_m / m_per_lamdivD
#         nfpm = fpm_array.shape[1]
#         wavefront0 = mft2(wavefront0, fpm_sampling_lamdivD, pupil_diam_pix, nfpm, +1)   # MFT to highly-sampled focal plane
#         wavefront0 *= fpm_array
#         wavefront0 = mft2(wavefront0, fpm_sampling_lamdivD, pupil_diam_pix, n, -1)  # MFT to virtual pupil 
#         wavefront0 = ffts( wavefront0, -1 )    # back to normally-sampled focal plane
#         wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
#         wavefront0 = 0
        
        
        # super-sample FPM
        wavefront0 = inwave2.wavefront.get()
        wavefront0 = ffts( wavefront0, 1 )                # to virtual pupil
        
        n_mft = 1400
        wavefront0 = trim(wavefront0, n_mft)
        
        fpm_array = SPC.FPM.amplitude.get()
        
        import misc
        misc.myimshow(fpm_array)
        
        fpm_sampling_lam0divD = 0.1
        fpm_sampling_lamdivD = fpm_sampling_lam0divD * 825e-9 / SPC.wavelength.to(u.m).value
        nfpm = fpm_array.shape[1]
        pupil_diam_pix = 1000
        n = 2048
        wavefront0 = mft2(wavefront0, fpm_sampling_lamdivD, pupil_diam_pix, nfpm, +1)   # MFT to highly-sampled focal plane
        wavefront0 *= fpm_array
        wavefront0 = mft2(wavefront0, fpm_sampling_lamdivD, pupil_diam_pix, n, -1)  # MFT to virtual pupil 
        wavefront0 = ffts( wavefront0, -1 )    # back to normally-sampled focal plane
#         wavefront0 = proper.prop_shift_center(wavefront0)
        
        inwave2.wavefront = cp.array(wavefront0)
        wavefront0 = 0
        
    psf_hdu, wfs_from_fpm = fosys2.calc_psf(wavelength=SPC.wavelength, inwave=inwave2, normalize='none',
                                            return_final=True, return_intermediates=SPC.return_intermediates,)
    
    if SPC.return_intermediates:
        wfs = wfs_to_fpm + wfs_from_fpm
    else: 
        wfs = wfs_from_fpm

    return wfs   
       
    