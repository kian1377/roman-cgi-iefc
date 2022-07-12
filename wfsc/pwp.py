import numpy as np
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import copy

import poppy
if poppy.accel_math._USE_CUPY:
    import cupy as cp
from .import scoob

import misc

def run_pwp(DM, wavelength, Npairs, dark_zone, score_zone, 
            probe_radius=12, probe_offsets=(0,0), probe_axis='alternate', 
            use_jacobian=False, jacobian=None, 
            estimator='batch_process', 
            display_probes=False, return_2d=False):
    # DM is a poppy.ContinuousDeformableMirror objects
    Nacts = DM.surface.shape[0]
    dm_nom = copy.deepcopy(DM.surface) # store the nominal DM actuator commands
    ndz = dark_zone.data[~dark_zone.mask].shape[0]
    print('Total number of pixels within correction zone to be estimated: {:d} pixels'.format(ndz))
    
    dm_cube = np.zeros((1+2*Npairs, Nacts,Nacts))
    
    I_cube = np.zeros((1+2*Npairs, npsf, npsf))
    E_est = np.zeros((ndz,1), dtype=complex)
    I_inco_est = np.zeros((ndz,1))
    
    probe_phase_vec = np.array([0, Npairs])
    for k in range(Npairs-1):
        probe_phase_vec = np.append( probe_phase_vec, probe_phase_vec[-1] - (Npairs-1) )
        probe_phase_vec = np.append( probe_phase_vec, probe_phase_vec[-1] + Npairs )
    probe_phase_vec = probe_phase_vec*np.pi/Npairs
    print('Phase values for each probe in order (multiples of \u03C0): ', probe_phase_vec/np.pi)
    
    bad_axis_vec = ''
    if probe_axis == 'y':
        for _iter in range(2*Npairs): bad_axis_vec += 'y'
        print('Discontinuity axis set to y-axis for all probes.')
    elif probe_axis == 'x':
        for _iter in range(2*Npairs): bad_axis_vec += 'x'
        print('Discontinuity axis set to x-axis for all probes.')
    elif probe_axis in ('alt', 'xy', 'alternate'):
        for i_pair in range(2*Npairs):
            if (i_pair+1) % 4 == 1 or (i_pair+1) % 4 == 2: bad_axis_vec += 'x'
            elif (i_pair+1) % 4 == 3 or (i_pair+1) % 4 == 0: bad_axis_vec += 'y'
        print('Utilizing alternating discontinuity axis in the following order for {:.0f} probes: '.format(2*Npairs) + bad_axis_vec)
    
    DM.set_surface(dm_nom) # Reset DM commands to the unprobed state

    # initialize matrices for intnesities and positive and negative DM probe shapes
    I_pos = np.zeros((Npairs, ndz))
    I_neg = np.zeros((Npairs, ndz))
    dm_pos = np.zeros((Npairs, Nacts, Nacts))
    dm_neg = np.zeros((Npairs, Nacts, Nacts))
    
    psfs, wfs = scoob.run_multi(psf_pixelscale=psf_pixelscale, npsf=npsf, DM=DM, FPM=FPM)
    I0 = wfs[0][-1].intensity
    I0_vec = wfs[0][-1].intensity[~dark_zone.mask]
    I_cube[0,:,:] = I0
    
    I_norm_score = np.mean(I0[~score_zone.mask])
    I_norm_corr = np.mean(I0[~dark_zone.mask])
    print('Mean intensity for normaliztion (Correction ROI / Scoring ROI): {:.3e} / {:.3e} '.format(I_norm_corr, I_norm_score))
    
    I_norm_probe_max = 1e-4
    I_norm_probe = np.min([np.sqrt(np.max(I0_vec)*1e-5), I_norm_probe_max])
    print('Chosen probe intensities: {:.3e}'.format(I_norm_probe))
    
    I_probed_avg = 0 # initialize mean of probed intensity values (mean in the dark-zone)
    i_odd = 0  # Initialize index counters
    i_even = 0
    for i_probe in range(2*Npairs):
        amp = 4*np.pi*wavelength.to(u.m).value*np.sqrt(I_norm_probe) # set the desired amplitude of the probe in the ROI
        probe_cmd = generate_sinc_probe(Nacts, amp=amp, probe_radius=probe_radius, offset=probe_offsets, 
                                        probe_phase=probe_phase_vec[i_probe], bad_axis=bad_axis_vec[i_probe])
        del_dm = probe_cmd
        
        dm = dm_nom + del_dm
        DM.set_surface(dm)
        if display_probes: misc.display_dm( DM, vmax=np.max([ dm.max(), np.abs(dm.min()) ]) )
        psfs, wfs = scoob.run_multi(psf_pixelscale=psf_pixelscale, npsf=npsf, DM=DM, FPM=FPM)

        im = wfs[0][-1].intensity
        im_vec = im[~dark_zone.mask]
        
        i_img = 1+i_probe # index counter for which image
        im_non_neg = im
        im_non_neg[im<0] = 0
        
        I_probed_avg += np.mean(im[~dark_zone.mask]) / (2*Npairs)
        
        I_cube[i_img,:,:] = im
        dm_cube[i_img,:,:] = dm
        
        probe_sign = '-+'
        
        # Assign image to positive or negative probe collection:
        if (i_probe+1) % 2 == 1:  # Odd; for plus probes
            dm_pos[i_odd, :, :] = dm
            I_pos[i_odd, :] = im[~dark_zone.mask]
            i_odd += 1
        elif (i_probe+1) % 2 == 0:  # Even; for minus probes
            dm_neg[i_even, :, :] = dm
            I_neg[i_even,:] = im[~dark_zone.mask]
            i_even += 1
        '''End of for-loop for obtaining intensities for each probe.'''

    amp_sq = (I_pos + I_neg)/2 - np.tile(I0_vec.reshape((1,-1)), (Npairs,1))
    amp_sq[amp_sq < 0] = 0 # set probe amplitude to 0 if it ends up being less than 0 (weird things happen with math)
    amp = np.sqrt(amp_sq)
    isnonzero = np.all(amp,0)
    z_all = ((I_pos-I_neg)/4).T
    amp_sq_2d_cube = np.zeros((Npairs, npsf, npsf))
    for i_probe in range(Npairs):
        amp_sq_2d = np.zeros((npsf, npsf))
        amp_sq_2d[~dark_zone.mask] = amp_sq[i_probe,:]
        amp_sq_2d_cube[i_probe,:,:] = amp_sq_2d
        print('Mean measured I_norm for probe {:d} = {:.3e}'.format(i_probe+1, np.mean(amp_sq_2d[~dark_zone.mask])))
    
    if use_jacobian and jacobian is not None:
        print('Using Jacobian to obtain delta E-Field estimates.')
        del_E_pos = np.zeros_like(I_pos, dtype=complex)
        for i_probe in range(Npairs):
            del_dm = dm_pos[i_probe,:,:] - dm_nom
            del_E_pos = jacobian * np.arange(Nacts**2)
    else: 
        print('Using model to obtain E-Field estimates for probes.')
        # first get the unprobed field based on the model
        DM.set_surface(dm_nom)
        psfs, wfs = scoob.run_multi(psf_pixelscale=psf_pixelscale, npsf=npsf, DM=DM, FPM=FPM)
        E0 = wfs[0][-1].wavefront
        E0_vec = E0[~dark_zone.mask]

        E_pos = np.zeros_like(I_pos, dtype=complex)
        E_neg = np.zeros_like(I_neg, dtype=complex)
        for i_probe in range(Npairs):
            print('\tEstimating for probe pair {:d}.'.format(i_probe))
            # For plus probes:
            DM_pos = copy.deepcopy(DM)
            DM_pos.set_surface(dm_pos[i_probe, :, :])
            
            # For minus probes:
            DM_neg = copy.deepcopy(DM)
            DM_neg.set_surface(dm_neg[i_probe, :, :])

            psfs, wfs = scoob.run_multi(psf_pixelscale=psf_pixelscale, npsf=npsf, DM=[DM_pos, DM_neg], FPM=FPM)
            E_pos[i_probe, :] = wfs[0][-1].wavefront[~dark_zone.mask]
            E_neg[i_probe, :] = wfs[1][-1].wavefront[~dark_zone.mask]
        
        # Create delta E-fields for each probe image.
        # Then create Npairs phase angles.
        del_E_pos = E_pos - np.tile(E0_vec.reshape((1,-1)), (Npairs,1))
        del_E_neg = E_neg - np.tile(E0_vec.reshape((1,-1)), (Npairs,1))
        del_phdm = np.zeros((Npairs, ndz))  # phases
        for i_probe in range(Npairs):
            del_phdm[i_probe, :] = np.arctan2(np.imag(del_E_pos[i_probe, :] - del_E_neg[i_probe, :]), 
                                              np.real(del_E_pos[i_probe, :] - del_E_neg[i_probe, :]) )

    
    if estimator=='batch_process':
        E_est = np.zeros((ndz,), dtype=complex)
        i_zeros = 0 # counter for zeroed out dark zone pixels
        if use_jacobian and jacobian is not None:
            print('Using Jacobian to obtain observation matrix for batch-process.')
            for i_pix in range(ndz): # perform pixel by pixel batch process to estimate E-field using jacobian
                del_E = del_E_pos[i_pix].T
                H = np.array([np.real(del_E), np.imag(del_E)])

                E_pix = np.linalg.pinv(H) @ z_all[i_pix, :]  # Batch processing
                E_est[i_pix] = E_pix[0] + 1j*E_pix[1]
        else: 
            print('Using E-Field values from model to obtain observation matrix.')
            for i_pix in range(ndz): # perform pixel by pixel batch process to estimate E-field using model results
                H = np.zeros([Npairs, 2])  # Observation matrix
                # Leave E_est for a pixel as zero if any probe amp is 0
                if isnonzero[i_pix] == 1:
                    for i_probe in range(Npairs):
                        H[i_probe, :] = amp[i_probe, i_pix] * np.array([np.cos(del_phdm[i_probe, i_pix]), 
                                                                        np.sin(del_phdm[i_probe, i_pix])])
                else:
                    i_zeros += 1
                E_pix = np.linalg.pinv(H) @ z_all[i_pix, :]  # Batch processing
                E_est[i_pix] = E_pix[0] + 1j*E_pix[1]
                
            print('Number of pixels set to 0 in ROI: {:d}'.format(i_zeros))
            E_est[np.abs(E_est)**2 > 1e-2] = 0.0 # be careful with pixels that are too bright

    elif estimator=='kalman_filter': pass # not yet implemented
    
    if return_2d:
        E_est_2d = copy.copy(dark_zone).astype(np.complex128)
        np.place(E_est_2d.data, vals=E_est, mask=~E_est_2d.mask)
        return E_est, E_est_2d
    else:
        return E_est

def generate_sinc_probe(Nacts, amp, probe_radius, probe_phase=0, offset=(0,0), bad_axis='x'):
    print('Generating probe with amplitude={:.3e}, radius={:.1f}, phase={:.3f}, offset=({:.1f},{:.1f}), with discontinuity along '\
          .format(amp, probe_radius, probe_phase, offset[0], offset[1]) + bad_axis + ' axis.')
    xacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[0])/Nacts
    yacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[1])/Nacts
    Xacts,Yacts = np.meshgrid(xacts,yacts)
    if bad_axis=='x': 
        fX = 2*probe_radius
        fY = probe_radius
        omegaY = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaY*Yacts + probe_phase)
    elif bad_axis=='y': 
        fX = probe_radius
        fY = 2*probe_radius
        omegaX = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaX*Xacts + probe_phase) 
    if probe_phase == 0:
        f = 2*probe_radius
        probe_commands = amp * np.sinc(f*Xacts)*np.sinc(f*Yacts)
    return probe_commands

