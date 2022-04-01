import numpy as np
import cupy as cp
import poppy
if poppy.accel_math._USE_CUPY:
    from cupyx.scipy.sparse import linalg as sLA
else:
    from scipy.sparse import linalg as sLA

import time
import copy
from astropy.io import fits

from . import iefc_utils
from .poppy_roman_cgi_phasec import cgi
import misc

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, DM=1, return_all=False, pca_modes=None):
    if poppy.accel_math._USE_CUPY:
        differential_operator = cp.array([ [-1,1,0,0] , [0,0,-1,1] ]) / (2 * probe_amplitude * sysi.texp)
    else:
        differential_operator = np.array([ [-1,1,0,0] , [0,0,-1,1] ]) / (2 * probe_amplitude * sysi.texp)

    if DM==1:
        dm1_commands = [-1.0*probe_amplitude*probe_cube[0], 1.0*probe_amplitude*probe_cube[0],
                        -1.0*probe_amplitude*probe_cube[1], 1.0*probe_amplitude*probe_cube[1]]
        dm2_commands = [np.zeros((sysi.Nact**2)), np.zeros((sysi.Nact**2)), 
                        np.zeros((sysi.Nact**2)), np.zeros((sysi.Nact**2))]
    elif DM==2:
        dm2_commands = [-1.0*probe_amplitude*probe_cube[0], 1.0*probe_amplitude*probe_cube[0],
                        -1.0*probe_amplitude*probe_cube[1], 1.0*probe_amplitude*probe_cube[1]]
        dm1_commands = [np.zeros((sysi.Nact**2)), np.zeros((sysi.Nact**2)), 
                        np.zeros((sysi.Nact**2)), np.zeros((sysi.Nact**2))]
    dm_commands = []
    for i in range(len(dm1_commands)):
        dm_commands.append(np.vstack((dm1_commands[i], dm2_commands[i])))
    wfs = sysi.calc_psfs(dm_commands=dm_commands)
    
    images=[]
    for i,wf in enumerate(wfs): 
        images.append(wf[0].intensity.flatten())
        
    if poppy.accel_math._USE_CUPY:
        images = cp.array(images)
    else:
        images = np.array(images)
    differential_images = differential_operator.dot(images)
    
    if pca_modes is not None:
        differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T

    if return_all:
        return differential_images, images
    else:
        return differential_images
    
def calibrate(sysi, probe_amplitude, probe_modes, calibration_amplitude, calibration_modes, start_mode=0):
    print('Calibrating I-EFC...')
    
    slopes_1 = []
    slopes_2 = []
    images_1 = []
    images_2 = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        try:
            slope1, slope2 = (0, 0)
            for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
                # DM1: Set the DM to the correct state
                sysi.add_dm1(s * calibration_amplitude * calibration_mode)
                differential_images_1, single_images_1 = take_measurement(sysi, probe_modes, probe_amplitude, DM=1,
                                                                          return_all=True)
                
                images_1.append(single_images_1)
                slope1 += s * differential_images_1 / (2 * calibration_amplitude)
                
                sysi.add_dm1(-s * calibration_amplitude * calibration_mode) # remove the calibrated mode from DM1
                
                # DM2: Set the DM to the correct state
                sysi.add_dm2(s * calibration_amplitude * calibration_mode)
                differential_images_2, single_images_2 = take_measurement(sysi, probe_modes, probe_amplitude, DM=1, 
                                                                          return_all=True)
                
                images_2.append(single_images_2)
                slope2 += s * differential_images_2 / (2 * calibration_amplitude)
                
                sysi.add_dm2(-s * calibration_amplitude * calibration_mode) # remove the calibrated mode from DM2
                
            print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1+start_mode, calibration_modes.shape[0], 
                                                                    time.time()-start))
            slopes_1.append(slope1)
            slopes_2.append(slope2)
        except KeyboardInterrupt: 
            print('Calibration interrupted.')
            break
    
    if poppy.accel_math._USE_CUPY:
        slopes_1 = cp.array(slopes_1)
        slopes_2 = cp.array(slopes_2)
        images_1 = cp.array(images_1)
        images_2 = cp.array(images_2)
        
        slopes = cp.concatenate((slopes_1,slopes_2), axis=0) # this is the response cube
        images = cp.concatenate((images_1,images_2), axis=0) # this is the calibration cube
    else:
        slopes_1 = np.array(slopes_1)
        slopes_2 = np.array(slopes_2)
        images_1 = np.array(images_1)
        images_2 = np.array(images_2)

        slopes = np.concatenate((slopes_1,slopes_2), axis=0) # this is the response cube
        images = np.concatenate((images_1,images_2), axis=0) # this is the calibration cube
    
    print('Calibration complete.')
    return slopes, images

# def WeightedLeastSquares(A, W, rcond=1e-15):
#     cov = A.T.dot(W.dot(A))
#     if poppy.accel_math._USE_CUPY:
#         return cp.linalg.inv(cov + rcond * cp.diag(cov).max() * cp.eye(A.shape[1])).dot( A.T.dot(W) )
#     else:
#         return np.linalg.inv(cov + rcond * np.diag(cov).max() * np.eye(A.shape[1])).dot( A.T.dot(W) )

# def TikhonovInverse(A, rcond=1e-15):
#     if poppy.accel_math._USE_CUPY:
#         U, s, Vt = np.linalg.svd(A, full_matrices=False)
#     else:
#         U, s, Vt = np.linalg.svd(A, full_matrices=False)
#     s_inv = s/(s**2 + (rcond * s.max())**2)
#     return (Vt.T * s_inv).dot(U.T)

def construct_control_matrix(response_matrix, weight_map, rcond1=1e-2, rcond2=1e-2, WLS=True, pca_modes=None):
    weight_mask = weight_map>0
    
    # Invert the matrix with an SVD and Tikhonov regularization
    masked_matrix = response_matrix[:, :, weight_mask].reshape((response_matrix.shape[0], -1)).T
    
    # Add the extra PCA modes that are fitted
    if pca_modes is not None:
        double_pca_modes = np.concatenate( (pca_modes[:, weight_mask], pca_modes[:, weight_mask]), axis=1).T
        masked_matrix = np.hstack((masked_matrix, double_pca_modes))
    
    nmodes = int(response_matrix.shape[0]/2)
    if WLS:  
        print('Using Weighted Least Squares ')
        Wmatrix = np.diag(np.concatenate((weight_map[weight_mask], weight_map[weight_mask])))
        control_matrix_1 = iefc_utils.WeightedLeastSquares(masked_matrix[:,:nmodes], Wmatrix, rcond=rcond1)
        control_matrix_2 = iefc_utils.WeightedLeastSquares(masked_matrix[:,nmodes:], Wmatrix, rcond=rcond2)
    else: 
        print('Using Tikhonov Inverse')
        control_matrix_1 = iefc_utils.TikhonovInverse(masked_matrix[:,:nmodes], rcond=rcond1)
        control_matrix_2 = iefc_utils.TikhonovInverse(masked_matrix[:,nmodes:], rcond=rcond2)
    control_matrix = np.concatenate((control_matrix_1, control_matrix_2), axis=0)
    
    if pca_modes is not None:
        # Return the control matrix minus the pca_mode coefficients
        return control_matrix[0:-pca_modes.shape[0]]
    else:
        return control_matrix
    
def single_iteration(sysi, probe_cube, probe_amplitude, control_matrix, pixel_mask_dark_hole):
    # Take a measurement
    differential_images = take_measurement(sysi, probe_cube, probe_amplitude)
    
    # Choose which pixels we want to control
    measurement_vector = differential_images[:, pixel_mask_dark_hole].ravel()
    
    # Calculate the control signal in modal coefficients
    reconstructed_coefficients = control_matrix.dot( measurement_vector )
    
    return reconstructed_coefficients
    

def run(sysi, control_matrix, probe_modes, probe_amplitude, calibration_modes, weights,
        num_iterations=10, gain=-0.5, leakage=0.0,
        display=False):
    nmodes = calibration_modes.shape[0]
    
    # The metric
    metric_images = []
    dm1_commands = []
    dm2_commands = []
    commands = 0.0
    
    dm1_ref = sysi.DM1.surface.get()
    dm2_ref = sysi.DM2.surface.get()
    
    print('Running I-EFC...')
    start = time.time()
    for i in range(num_iterations):
        print("\tClosed-loop iteration {:d} / {:d}".format(i+1, num_iterations))
        delta_coefficients = single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, weights.flatten()>0)
        commands = (1.0-leakage) * commands + gain * delta_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        dm1_command = calibration_modes.T.dot(commands[:nmodes].get()).reshape(sysi.Nact,sysi.Nact)
        dm2_command = calibration_modes.T.dot(commands[nmodes:].get()).reshape(sysi.Nact,sysi.Nact)
        
        # Set the current DM state
        sysi.set_dm1(dm1_ref + dm1_command)
        sysi.set_dm2(dm2_ref + dm2_command)
        
        # Take an image to estimate the metrics
        image = sysi.calc_psf()[-1]
        metric_images.append(image)
        dm1_commands.append(copy.copy(sysi.DM1.surface))
        dm2_commands.append(copy.copy(sysi.DM2.surface))
        
        if display: 
            misc.myimshow2(dm1_command, dm2_command)
            misc.myimshow(image.intensity, lognorm=True)
    print('I-EFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm1_commands, dm2_commands









