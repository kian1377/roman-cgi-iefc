import numpy as np
import cupy as cp
import poppy
if poppy.accel_math._USE_CUPY:
    from cupyx.scipy.sparse import linalg as sLA
else:
    from scipy.sparse import linalg as sLA

import time
from astropy.io import fits
import copy

from . import iefc_utils
from .poppy_roman_cgi_phasec import cgi
import misc

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, DM=1, return_all=False, pca_modes=None):
    if poppy.accel_math._USE_CUPY:
        differential_operator = cp.array([[-1,1,0,0,0,0],
                                          [0,0,-1,1,0,0],
                                          [0,0,0,0,-1,1]]) / (2 * probe_amplitude * sysi.texp)
    else:
        differential_operator = np.array([ [-1,1,0,0] , [0,0,-1,1] ]) / (2 * probe_amplitude * sysi.texp)

    if DM==1:
        dm1_commands = [-1.0*probe_amplitude*probe_cube[0], 1.0*probe_amplitude*probe_cube[0],
                        -1.0*probe_amplitude*probe_cube[1], 1.0*probe_amplitude*probe_cube[1],
                        -1.0*probe_amplitude*probe_cube[2], 1.0*probe_amplitude*probe_cube[2]]
        dm2_commands = [np.zeros((sysi.Nact**2)), np.zeros((sysi.Nact**2)), 
                        np.zeros((sysi.Nact**2)), np.zeros((sysi.Nact**2)),
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
    
def calibrate(sysi, probe_amplitude, probe_modes, calibration_amplitude, calibration_modes, DM=1, start_mode=0):
    print('Calibrating I-EFC...')
    slopes = [] # this becomes the response cube
    images = [] # this becomes the calibration cube
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        try:
            slope = 0
            # We need a + and - probe to estimate the jacobian
            for s in [-1, 1]:
                if DM==1:
                    add_dm = sysi.add_dm1
                elif DM==2:
                    add_dm = sysi.add_dm2
                    
                # Set the DM to the correct state
                add_dm(s * calibration_amplitude * calibration_mode)
                differential_images, single_images = take_measurement(sysi, probe_modes, probe_amplitude, 
                                                                      DM=1, return_all=True)

                slope += s * differential_images / (2 * calibration_amplitude)
                images.append(single_images)

                # Remove the calibrated mode
                add_dm(-s * calibration_amplitude * calibration_mode)
            print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1+start_mode, calibration_modes.shape[0], 
                                                                    time.time()-start))
            slopes.append(slope)
        except KeyboardInterrupt: 
            print('Calibration interrupted.')
            break
    print('Calibration complete.')
    
    if poppy.accel_math._USE_CUPY:
        return cp.array(slopes), cp.array(images)
    else:
        return np.array(slopes), np.array(images)

# def WeightedLeastSquares(A, W, rcond=1e-15):
#     cov = A.T.dot(W.dot(A))
#     if poppy.accel_math._USE_CUPY:
#         return cp.linalg.inv(cov + rcond * cp.diag(cov).max() * cp.eye(A.shape[1])).dot( A.T.dot(W) )
#     else:
#         return np.linalg.inv(cov + rcond * np.diag(cov).max() * np.eye(A.shape[1])).dot( A.T.dot(W) )

# def TikhonovInverse(A, rcond=1e-15):
#     U, s, Vt = np.linalg.svd(A, full_matrices=False)
#     s_inv = s/(s**2 + (rcond * s.max())**2)
#     return (Vt.T * s_inv).dot(U.T)

def construct_control_matrix(response_matrix, weight_map, rcond=1e-2, WLS=True, pca_modes=None):
    weight_mask = weight_map>0
    
    # Invert the matrix with an SVD and Tikhonov regularization
    masked_matrix = response_matrix[:, :, weight_mask].reshape((response_matrix.shape[0], -1)).T
    
    # Add the extra PCA modes that are fitted
    if pca_modes is not None:
        double_pca_modes = np.concatenate( (pca_modes[:, weight_mask], pca_modes[:, weight_mask]), axis=1).T
        masked_matrix = np.hstack((masked_matrix, double_pca_modes))
    
    nmodes = int(response_matrix.shape[0])
    if WLS:  
        print('Using Weighted Least Squares ')
        if poppy.accel_math._USE_CUPY:
            Wmatrix = cp.diag(cp.concatenate((weight_map[weight_mask], weight_map[weight_mask], weight_map[weight_mask])))
        else:
            Wmatrix = np.diag(np.concatenate((weight_map[weight_mask], weight_map[weight_mask], weight_map[weight_mask])))
        control_matrix = iefc_utils.WeightedLeastSquares(masked_matrix[:,:nmodes], Wmatrix, rcond=rcond)
    else: 
        print('Using Tikhonov Inverse')
        control_matrix = iefc_utils.TikhonovInverse(masked_matrix[:,:nmodes], rcond=rcond)
    
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
    # The metric
    metric_images = []
    dm_commands = []
    command = 0.0
    
    dm1_ref = sysi.DM1.surface.get()
    dm2_ref = sysi.DM2.surface.get()
    
    print('Running I-EFC...')
    start = time.time()
    for i in range(num_iterations):
        print("\tClosed-loop iteration {:d} / {:d}".format(i+1, num_iterations))
        delta_coefficients = single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, weights.flatten()>0)
        command = (1.0-leakage) * command + gain * delta_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        dm_command = calibration_modes.T.dot(command.get()).reshape(sysi.Nact,sysi.Nact)

        # Set the current DM state
        sysi.set_dm1(dm1_ref + dm_command)

        # Take an image to estimate the metrics
        image = sysi.calc_psf()[-1]
        metric_images.append(image)
        dm_commands.append(copy.copy(sysi.DM1.surface))
        
        if display: misc.myimshow2(dm_command.reshape(sysi.Nact,sysi.Nact), image.intensity, lognorm2=True)
    print('I-EFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm_commands





