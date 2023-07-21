from wfsc_tests.math_module import xp
from wfsc_tests import utils, imshows
# from wfsc_tests import imshows
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, return_all=False, pca_modes=None):

    if probe_cube.shape[0]==2:
        differential_operator = xp.array([[-1,1,0,0],
                                          [0,0,-1,1]]) / (2 * probe_amplitude )
    elif probe_cube.shape[0]==3:
        differential_operator = xp.array([[-1,1,0,0,0,0],
                                          [0,0,-1,1,0,0],
                                          [0,0,0,0,-1,1]]) / (2 * probe_amplitude )
    
    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []
    for probe in probe_cube: 
        for amp in amps:
            sysi.add_dm1(amp*probe)
            image = sysi.snap()
            images.append(image.flatten())
            sysi.add_dm1(-amp*probe)
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
    print('Calibrating I-EFC...')
    if return_all:
        response_cube = []
    response_matrix = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        response = 0
        for s in [-1, 1]:
            # Set the DM to the correct state
            sysi.add_dm1(s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
            differential_images = take_measurement(sysi, probe_modes, probe_amplitude, return_all=False)
            sysi.add_dm1(-s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))

            response += s * differential_images / (2 * calibration_amplitude)
        print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1+start_mode, calibration_modes.shape[0], 
                                                                time.time()-start))
        if return_all: 
            response_cube.append(response)
        response_matrix.append(xp.concatenate([response[0, control_mask], response[1, control_mask]])) # masked response for each probe mode 
    print('Calibration complete.')
    
    if return_all:
        return xp.array(response_matrix).T, xp.array(response_cube)
    else:
        return xp.array(response_matrix).T

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
        thresh=None,
        exp_time_gain=None,
        plot_current=True,
        plot_all=False,
        contrast_parameters=None,
        plot_radial_contrast=True):
    print('Running I-EFC...')
    start = time.time()
    
    metric_images = []
    dm_commands = []
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros_like(dm_ref) 
    command = 0.0
    for i in range(num_iterations+1):
        print("\tClosed-loop iteration {:d} / {:d}".format(i, num_iterations))
        # Set the current DM state
        sysi.set_dm1(dm_ref + dm_command)
        
        # Take an image to estimate the metrics
        image = sysi.snap()
        if thresh is not None and exp_time_gain is not None:
            mean_readout = np.mean(image[utils.ensure_np_array(control_mask)])
            if mean_readout<thresh:
                print('Mean readout was below the threshold, adjusting exposure time by factor of {:.2f}'.format(exp_time_gain))
                sysi.exp_time *= exp_time_gain
            
        metric_images.append(copy.copy(image))
        dm_commands.append(sysi.get_dm1())
        
        delta_coefficients = -single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, control_mask)
        command = (1.0-leakage)*command + loop_gain*delta_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        dm_command = calibration_modes.T.dot(utils.ensure_np_array(command)).reshape(sysi.Nact,sysi.Nact)
        
        if plot_current: 
            if contrast_parameters is not None:
                max_ref = contrast_parameters['max_ref']
                uo_exp_time = contrast_parameters['unocculted_exp_time']
                contrast_map = image/max_ref * uo_exp_time/sysi.exp_time
                image = contrast_map
                
            if not plot_all: clear_output(wait=True)
            imshows.imshow2(dm_commands[i], utils.ensure_np_array(image),
                            'DM Command', 'Image: Iteration {:d}'.format(i),
                            pxscl2=sysi.psf_pixelscale_lamD, lognorm2=True,)
            
            if plot_radial_contrast:
                    utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=30)
                    
    print('I-EFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm_commands



