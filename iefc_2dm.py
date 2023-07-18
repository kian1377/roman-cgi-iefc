from math_module import xp, _scipy, ensure_np_array
import utils
import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, DM=1, return_all=False, pca_modes=None, display=False):

    if probe_cube.shape[0]==2:
        differential_operator = xp.array([[-1,1,0,0],
                                          [0,0,-1,1]]) / (2 * probe_amplitude)
    elif probe_cube.shape[0]==3:
        differential_operator = xp.array([[-1,1,0,0,0,0],
                                          [0,0,-1,1,0,0],
                                          [0,0,0,0,-1,1]]) / (2 * probe_amplitude)
    
    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []
    for probe in probe_cube: 
        for amp in amps:
            if DM==1:
                sysi.add_dm1(amp*probe)
                image = sysi.snap()
                images.append(image.flatten())
                sysi.add_dm1(-amp*probe)
            elif DM==2:
                sysi.add_dm2(amp*probe)
                image = sysi.snap()
                images.append(image.flatten())
                sysi.add_dm2(-amp*probe)
            
    images = xp.array(images)
    
    differential_images = differential_operator.dot(images)
    
    if pca_modes is not None:
        differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T
        
    if return_all:
        return differential_images, images
    else:
        return differential_images
    
def calibrate(sysi, 
              control_mask, 
              probe_amplitude, probe_modes, 
              calibration_amplitude, calibration_modes, 
              start_mode=0,
              return_all=False):
    print('Calibrating iEFC...')
    
    response_matrix_1 = []
    response_matrix_2 = []
    if return_all: # be ready to store the full focal plane responses (difference images)
        response_cube_1 = []
        response_cube_2 = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        response_1, response_2 = (0, 0)
        for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
            # DM1: Set the DM to the correct state
            sysi.add_dm1(s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
            diff_ims_1 = take_measurement(sysi, probe_modes, probe_amplitude, DM=1)
            response_1 += s * diff_ims_1 / (2 * calibration_amplitude)
            sysi.add_dm1(-s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact)) # remove the mode

            # DM2: Set the DM to the correct state
            sysi.add_dm2(s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
            diff_ims_2 = take_measurement(sysi, probe_modes, probe_amplitude, DM=1)
            response_2 += s * diff_ims_2 / (2 * calibration_amplitude)
            sysi.add_dm2(-s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact)) 
        
        print("\tCalibrated mode {:d}/{:d} in {:.3f}s".format(ci+1, calibration_modes.shape[0], time.time()-start), end='')
        print("\r", end="")
        
        if probe_modes.shape[0]==2:
            response_matrix_1.append( xp.concatenate([response_1[0, control_mask],
                                                      response_1[1, control_mask]]) )
            response_matrix_2.append( xp.concatenate([response_2[0, control_mask], 
                                                      response_2[1, control_mask]]) )
        elif probe_modes.shape[0]==3:
            response_matrix_1.append( xp.concatenate([response_1[0, control_mask], 
                                                      response_1[1, control_mask],
                                                      response_1[2, control_mask]]) )
            response_matrix_2.append( xp.concatenate([response_2[0, control_mask], 
                                                      response_2[1, control_mask],
                                                      response_2[2, control_mask]]) )
        
        if return_all: 
            response_cube_1.append(response_1)
            response_cube_2.append(response_2)
            
    response_matrix_1 = xp.array(response_matrix_1)
    response_matrix_2 = xp.array(response_matrix_2)
    response_matrix = xp.concatenate((response_matrix_1,response_matrix_2), axis=0) # this is the response matrix to be inverted
    
    if return_all:
        response_cube_1 = xp.array(response_cube_1)
        response_cube_2 = xp.array(response_cube_2)
        response_cube = xp.concatenate((response_cube_1,response_cube_2), axis=0) # this is the response matrix to be inverted
    print()
    print('Calibration complete.')
    
    if return_all:
        return response_matrix.T, xp.array(response_cube)
    else:
        return response_matrix.T
    

def single_iteration(sysi, probe_cube, probe_amplitude, control_matrix, control_mask):
    # Take a measurement
    differential_images = take_measurement(sysi, probe_cube, probe_amplitude)
    
    # Choose which pixels we want to control
    measurement_vector = differential_images[:, control_mask.ravel()].ravel()

    # Calculate the control signal in modal coefficients
    reconstructed_coefficients = control_matrix.dot( measurement_vector )
    
    return reconstructed_coefficients

def run(sysi,
        control_matrix,
        probe_modes, probe_amplitude, 
        calibration_modes,
        control_mask,
        num_iterations=10, 
        loop_gain=0.5, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        plot_radial_contrast=True):
    
    print('Running I-EFC...')
    start = time.time()
    
    Nc = calibration_modes.shape[0]
    
    # The metric
    metric_images = []
    dm1_commands = []
    dm2_commands = []
    
    dm1_ref = sysi.get_dm1()
    dm2_ref = sysi.get_dm2()
    command = 0.0
    dm1_command = 0.0
    dm2_command = 0.0
    for i in range(num_iterations):
        print("\tClosed-loop iteration {:d} / {:d}".format(i+1, num_iterations))
        # Set the current DM state
        sysi.set_dm1(dm1_ref + dm1_command)
        sysi.set_dm2(dm2_ref + dm2_command)
        
        # Take an image to estimate the metrics
        image = sysi.snap()
        
        metric_images.append(copy.copy(image))
        dm1_commands.append(sysi.get_dm1())
        dm2_commands.append(sysi.get_dm2())
        
        delta_coefficients = single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, control_mask)
        command = (1.0-leakage)*command + loop_gain*delta_coefficients
        display(command.shape)
        
        # Reconstruct the full phase from the Fourier modes
        dm1_command = -calibration_modes.T.dot(utils.ensure_np_array(command[:Nc])).reshape(sysi.Nact,sysi.Nact)
        dm2_command = -calibration_modes.T.dot(utils.ensure_np_array(command[Nc:])).reshape(sysi.Nact,sysi.Nact)
        
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            imshows.imshow3(dm1_commands[i], dm2_commands[i], image, 
                               'DM1', 'DM2', 'Image: Iteration {:d}'.format(i+1),
                               lognorm3=True, vmin3=1e-11, pxscl3=sysi.psf_pixelscale_lamD)
            
            if plot_radial_contrast:
                utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=50)
#     /1.54100337e+11
    metric_images = xp.array(metric_images)
    dm1_commands = xp.array(dm1_commands)
    dm2_commands = xp.array(dm2_commands)
    print('I-EFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm1_commands, dm2_commands





