import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output
import time
import ray

import poppy
if poppy.accel_math._USE_CUPY:
    import cupy as cp

from . import iefc_utils as iefcu

import misc

def build_jacobian(sysi, epsilon, dark_mask):
    print('Building Jacobian.')
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    start = time.time()
    for i, mode in enumerate(modes):
        print('\tCalculating response for mode {:d}. Elapsed time={:.3f} sec.'.format(i,time.time()-start))
        
        dm1_commands = [amps[0]*mode, amps[1]*mode]
        dm_commands = [np.vstack( (dm1_commands[0], np.zeros((sysi.Nact**2))) ),
                       np.vstack( (dm1_commands[1], np.zeros((sysi.Nact**2))) )]
        
        wfs = sysi.calc_psfs(ngpus=1/2, dm_commands=dm_commands)
        
        response = (amps[0]*wfs[0][-1].wavefront + amps[1]*wfs[1][-1].wavefront)
        response /= np.var(amps)
        response = response.flatten()[dark_mask.flatten()]
        
        responses.append(np.concatenate( (response.real, response.imag) ))
    
    if poppy.accel_math._USE_CUPY:
        jacobian = cp.array(responses).T
    else:
        jacobian = np.array(responses).T
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return jacobian

def run_efc_perfect(sysi, efc_matrix, dark_mask, efc_loop_gain=0.5, iterations=5, display=False):
    print('Beginning closed-loop EFC simulation.')
    dm1_command = np.zeros((sysi.Nact, sysi.Nact)) 
    
    commands = []
    efields = []
    
    dm1_ref = sysi.DM1.surface.get()
    dm2_ref = sysi.DM2.surface.get()
    
    start=time.time()
    for i in range(iterations):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        sysi.set_dm1(dm1_ref + dm1_command) 
        
        wfs = sysi.calc_psfs()
        electric_field = wfs[0][-1].wavefront
        
        commands.append(dm1_command)
        efields.append(wfs[0][-1].wavefront)
        
        x = cp.concatenate( (electric_field[dark_mask].real, electric_field[dark_mask].imag) )
        del_dm = efc_matrix.dot(x).reshape(sysi.Nact,sysi.Nact).get()
        
        dm1_command -= efc_loop_gain * del_dm
        
        if display:
            misc.myimshow2(commands[i], abs(efields[i])**2, lognorm2=True)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields

def run_efc_pwp(sysi, efc_matrix, jac, probes, dark_mask, efc_loop_gain=0.5, iterations=5, display=False):
    print('Beginning closed-loop EFC simulation.')
    dm1_command = np.zeros((sysi.Nact, sysi.Nact)) 
    
    commands = []
    efields = []
    images = []
    
    dm1_ref = sysi.DM1.surface.get()
    dm2_ref = sysi.DM2.surface.get()
    
    start=time.time()
    for i in range(iterations):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        sysi.set_dm1(dm1_ref + dm1_command) 
        E_est = run_pwp(sysi, probes, jac, dark_mask)
        
        wfs = sysi.calc_psfs()
        I_exact = wfs[0][-1].intensity
        
        commands.append(dm1_command)
        efields.append(E_est)
        images.append(I_exact)
        
        x = cp.concatenate( (E_est[dark_mask].real, E_est[dark_mask].imag) )
        y = efc_matrix.dot(x)
        
        dm1_command -= (efc_loop_gain * y.reshape(sysi.Nact,sysi.Nact)).get()
        
        if display:
            misc.myimshow2(commands[i], abs(images[i])**2, lognorm2=True)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images

def run_pwp(sysi, probes, jacobian, dark_mask, use_noise=False, display=False):
    nmask = dark_mask.sum()
    
    dm1_ref = sysi.DM1.surface.flatten().get()
    dm2_ref = sysi.DM2.surface.flatten().get()

    dm_commands_p = []
    dm_commands_m = []
    for i in range(len(probes)):
        dm_commands_p.append(np.vstack((probes[i] + dm1_ref, dm2_ref)))
        dm_commands_m.append(np.vstack((-probes[i] + dm1_ref, dm2_ref)))

    sysi.reset_dms()
    wfs_p = sysi.calc_psfs(ngpus=1/3, dm_commands=dm_commands_p) # calculate images for plus probes   
    
    sysi.reset_dms()
    wfs_m = sysi.calc_psfs(ngpus=1/3, dm_commands=dm_commands_m) # calculate images for minus probes
    
    sysi.set_dm1(dm1_ref) # put DMs back in original state
    sysi.set_dm2(dm2_ref)
    
    Ip = []
    Im = []
    for i in range(len(probes)):
        if use_noise:
            Ip.append( sysi.add_noise(wfs_p[i][-1].intensity) )
            Im.append( sysi.add_noise(wfs_m[i][-1].intensity) )
        else:
            Ip.append(wfs_p[i][-1].intensity)
            Im.append(wfs_m[i][-1].intensity)

        if display:
            misc.myimshow2(Ip[i], Im[i], 
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           lognorm1=True, lognorm2=True)
            misc.myimshow(Ip[i]-Im[i], 'Probe {:d} Intensity Difference'.format(i+1))
        
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

