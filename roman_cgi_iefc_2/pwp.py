import numpy as np
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import copy

import poppy
if poppy.accel_math._USE_CUPY:
    import cupy as cp

    
from . import iefc_utils as iefcu

import misc

def run_pwp(sysi, probes, jacobian, dark_mask):
    nmask = dark_mask.sum()
    
    dm1_ref = sysi.DM1.surface.flatten().get()
    dm2_ref = sysi.DM2.surface.flatten().get()

    dm_commands_p = []
    dm_commands_m = []
    for i in range(len(probes)):
        dm_commands_p.append(np.vstack((probes[i] + dm1_ref, dm2_ref)))
        dm_commands_m.append(np.vstack((-probes[i] + dm1_ref, dm2_ref)))

    sysi.reset_dms()
    wfs_p = sysi.calc_psfs(dm_commands=dm_commands_p) # calculate images for plus probes
    
    sysi.reset_dms()
    wfs_m = sysi.calc_psfs(dm_commands=dm_commands_m) # calculate images for minus probes

    Ip = []
    Im = []
    for i in range(len(probes)):
        Ip.append(wfs_p[i][-1].intensity)
        Im.append(wfs_m[i][-1].intensity)

#         misc.myimshow2(Ip[i], Im[i], lognorm1=True, lognorm2=True)
#         misc.myimshow(Ip[i]-Im[i])
        
    E_probes = cp.zeros((probes.shape[0], 2*nmask))
    I_diff = cp.zeros((probes.shape[0], nmask))
    for i in range(len(probes)):
        E_probe = jacobian.dot(cp.array(probes[i])) # Use jacobian to model probe E-field at the focal plane
        E_probe = E_probe[:nmask] + 1j*E_probe[nmask:]

        E_probes[i, :nmask] = E_probe.real
        E_probes[i, nmask:] = E_probe.imag

        I_diff[i:(i+1), :] = (Ip[i] - Im[i])[dark_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = cp.zeros((nmask,), dtype=cp.complex128)
    for i in range(nmask):
        delI = I_diff[:, i]
        M = 2*cp.array([E_probes[:,i], E_probes[:,i+nmask]]).T

        Minv = iefcu.TikhonovInverse(M, 1e-2)

        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = cp.zeros((sysi.npsf,sysi.npsf), dtype=cp.complex128)
    cp.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    return E_est_2d

def create_sinc_probe(Nacts, amp, probe_radius, probe_phase=0, offset=(0,0), bad_axis='x'):
    print('Generating probe with amplitude={:.3e}, radius={:.1f}, phase={:.3f}, offset=({:.1f},{:.1f}), with discontinuity along '.format(amp, probe_radius, probe_phase, offset[0], offset[1]) + bad_axis + ' axis.')
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

def create_sinc_probes(Npairs, Nacts, probe_amplitude, probe_radius=10, probe_offset=(0,0), display=False):
    
    probe_phases = np.linspace(0, np.pi*(Npairs-1)/Npairs, Npairs)
    
    probes = []
    for i in range(Npairs):
        if i%2==0:
            axis = 'x'
        else:
            axis = 'y'
            
        probe = create_sinc_probe(Nacts, probe_amplitude, probe_radius, probe_phases[i], offset=probe_offset, bad_axis=axis)
        if display:
            misc.myimshow(probe)
            
        probes.append(probe.flatten())
        
    return np.array(probes)
    
    
    
    
    