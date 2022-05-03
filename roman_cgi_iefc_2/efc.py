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
        
        wfs = sysi.calc_psfs(dm_commands=dm_commands)
        
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

def run_efc(sysi, efc_matrix, dark_mask, efc_loop_gain=0.5, iterations=5):
    print('Beginning closed-loop EFC simulation.')
    current_actuators = np.zeros((sysi.Nact,sysi.Nact)) 
    
    actuators = []
    wavefronts = []
    
    start=time.time()
    for i in range(iterations):
        print('\tRunning iteration {:d}.'.format(i))
        sysi.set_dm(current_actuators) # set the DM surface to poke a particular actuator
        wfs = sysi.calc_psf()
        electric_field = wfs[-1].wavefront
        
        actuators.append(current_actuators)
        wavefronts.append(wfs[-1])
        
        x = cp.concatenate( (electric_field[dark_mask].real, electric_field[dark_mask].imag) )
        y = efc_matrix.dot(x)
        
        current_actuators -= efc_loop_gain * y.reshape(sysi.Nact,sysi.Nact)
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return actuators, wavefronts


