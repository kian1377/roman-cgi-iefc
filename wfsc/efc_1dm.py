import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output
import time
import copy
from importlib import reload

from . import utils
reload(utils)

from cgi_phasec_poppy import misc

def build_jacobian(sysi, epsilon, dark_mask, display=False):
    start = time.time()
    print('Building Jacobian.')
    
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    for i, mode in enumerate(modes):
        if dm_mask[i]==1:
            response = 0
            for amp in amps:
                mode = mode.reshape(sysi.Nact,sysi.Nact)
                
                sysi.add_dm1(amp*mode)
                
                psf = sysi.calc_psf()
                wavefront = psf.wavefront
                response += amp*wavefront/np.var(amps)
                
                sysi.add_dm1(-amp*mode)

            if display:
                misc.myimshow2(cp.abs(response), cp.angle(response))
            
            response = response.flatten().get()[dark_mask.flatten()]
            
        else:
            response = np.zeros((sysi.npsf, sysi.npsf), dtype=np.complex128).flatten()[dark_mask.flatten()]

        responses.append(np.concatenate((response.real, response.imag)))
        
        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1,num_modes,time.time()-start))
        
    jacobian = np.array(responses).T
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return jacobian

def run_pwp(sysi, probes, jacobian, dark_mask, reg_cond=1e-2, use_noise=False, display=False):
    nmask = dark_mask.sum()
    
    dm_ref = sysi.get_dm1()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm1(amp*probe)
            psf = sysi.snap()
                
            if amp==-1: 
                In.append(psf)
            else: 
                Ip.append(psf)
                
            sysi.add_dm1(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, 
                          )
        
    E_probes = np.zeros((probes.shape[0], 2*nmask))
    I_diff = np.zeros((probes.shape[0], nmask))
    for i in range(len(probes)):
        E_probe = jacobian.dot(np.array(probes[i].flatten())) # Use jacobian to model probe E-field at the focal plane
        E_probe = E_probe[:nmask] + 1j*E_probe[nmask:]

        E_probes[i, :nmask] = E_probe.real
        E_probes[i, nmask:] = E_probe.imag

        I_diff[i:(i+1), :] = (Ip[i] - In[i])[dark_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = np.zeros((nmask,), dtype=cp.complex128)
    for i in range(nmask):
        delI = I_diff[:, i]
        M = 2*np.array([E_probes[:,i], E_probes[:,i+nmask]]).T
        Minv = utils.TikhonovInverse(M, reg_cond)

        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    return E_est_2d

def run_efc_pwp(sysi, efc_matrix, jac, probes, dark_mask, efc_loop_gain=0.5, iterations=5, display=False):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    for i in range(iterations):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        sysi.set_dm1(dm_ref + dm_command)
        E_est = run_pwp(sysi, probes, jac, dark_mask)
        I_exact = sysi.snap()
        
        commands.append(sysi.get_dm1())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))
        
        x = np.concatenate( (E_est[dark_mask].real, E_est[dark_mask].imag) )
        y = efc_matrix.dot(x)
        
        dm_command -= efc_loop_gain * y.reshape(sysi.Nact,sysi.Nact)
        
        if display:
            misc.myimshow2(commands[i], abs(images[i])**2, lognorm2=True)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images

def run_efc_perfect(sysi, efc_matrix, dark_mask, efc_loop_gain=0.5, iterations=5, display=False):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')    
    commands = []
    efields = []
    
    start = time.time()
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    for i in range(iterations):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        sysi.set_dm1(dm_ref + dm_command) 
        
        psf = sysi.calc_psf()
        electric_field = psf.wavefront.get()
        
        commands.append(sysi.get_dm1())
        efields.append(copy.copy(electric_field))
        
        x = np.concatenate( (electric_field[dark_mask].real, electric_field[dark_mask].imag) )
        del_dm = efc_matrix.dot(x).reshape(sysi.Nact,sysi.Nact)
        
        dm_command -= efc_loop_gain * del_dm
        
        if display:
            misc.myimshow2(commands[i], abs(efields[i])**2, lognorm2=True)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields

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

def create_sinc_probes(Npairs, Nacts, dm_mask, probe_amplitude, probe_radius=10, probe_offset=(0,0), display=False):
    
    probe_phases = np.linspace(0, np.pi*(Npairs-1)/Npairs, Npairs)
    
    probes = []
    for i in range(Npairs):
        if i%2==0:
            axis = 'x'
        else:
            axis = 'y'
            
        probe = create_sinc_probe(Nacts, probe_amplitude, probe_radius, probe_phases[i], offset=probe_offset, bad_axis=axis)
            
        probes.append(probe*dm_mask)
    
    if display:
        if Npairs==2:
            misc.myimshow2(probes[0], probes[1])
        elif Npairs==3:
            misc.myimshow3(probes[0], probes[1], probes[2])
    
    return np.array(probes)


