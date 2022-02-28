import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy import interpolate
import time
from datetime import date
import astropy.io.fits as pyfits
from scipy.sparse import linalg as sLA

from .import cgi
import misc

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, DM=1, return_all=False, pca_modes=None):
    differential_operator = np.array([ [-1,1,0,0] , [0,0,-1,1] ]) / (2 * probe_amplitude * sysi.texp)

    if DM==1:
        dm1_commands = [-1.0*probe_amplitude*probe_cube[0].reshape(sysi.Nact,sysi.Nact),
                        1.0*probe_amplitude*probe_cube[0].reshape(sysi.Nact,sysi.Nact),
                        -1.0*probe_amplitude*probe_cube[1].reshape(sysi.Nact,sysi.Nact),
                        1.0*probe_amplitude*probe_cube[1].reshape(sysi.Nact,sysi.Nact)]
        dm2_commands = [np.zeros((48,48)), np.zeros((48,48)), np.zeros((48,48)), np.zeros((48,48))]
    elif DM==2:
        dm2_commands = [-1.0*probe_amplitude*probe_cube[0].reshape(sysi.Nact,sysi.Nact),
                        1.0*probe_amplitude*probe_cube[0].reshape(sysi.Nact,sysi.Nact),
                        -1.0*probe_amplitude*probe_cube[1].reshape(sysi.Nact,sysi.Nact),
                        1.0*probe_amplitude*probe_cube[1].reshape(sysi.Nact,sysi.Nact)]
        dm1_commands = [np.zeros((48,48)), np.zeros((48,48)), np.zeros((48,48)), np.zeros((48,48))]
    
    psfs = sysi.calc_psfs(dm1_commands, dm2_commands)
    images=[]
    for i,psf in enumerate(psfs): images.append(psf.flatten())
        
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
    
    slopes_1 = np.array(slopes_1)
    slopes_2 = np.array(slopes_2)
    images_1 = np.array(images_1)
    images_2 = np.array(images_2)

    slopes = np.concatenate((slopes_1,slopes_2), axis=0) # this is the response cube
    images = np.concatenate((images_1,images_2), axis=0) # this is the calibration cube
    
    print('Calibration complete.')
    return slopes, images

# def calibrate_2_probe(sysi, probe_amplitude, probe_modes_1, probe_modes_2
#                            calibration_amplitude, calibration_modes, start_mode=0):
#     print('Calibrating I-EFC...')
    
#     slopes_11, slopes_12, slopes_21, slopes_22 = ([], [], [], [])
#     images_11, images_12, images_21, images_22 = ([], [], [], [])
    
#     # Loop through all modes that you want to control
#     start = time.time()
#     for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
#         try:
#             slope11, slope12, slope21, slope22 = (0, 0, 0, 0) # 11 means Probe set 1 for DM 1
#             for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
#                 # DM1: Set the DM to the correct state
#                 sysi.add_dm1(s * calibration_amplitude * calibration_mode)
#                 differential_images_11, single_images_1 = take_measurement(sysi, probe_modes_1, probe_amplitude, DM=1,
#                                                                           return_all=True)
                
#                 images_11.append(single_images_1)
#                 slope11 += s * differential_images_1 / (2 * calibration_amplitude)
                
#                 sysi.add_dm1(-s * calibration_amplitude * calibration_mode) # remove the calibrated mode from DM1
                
#                 # DM2: Set the DM to the correct state
#                 sysi.add_dm2(s * calibration_amplitude * calibration_mode)
#                 differential_images_2, single_images_2 = take_measurement(sysi, probe_modes_1, probe_amplitude, DM=1, 
#                                                                           return_all=True)
                
#                 images_12.append(single_images_2)
#                 slope12 += s * differential_images_2 / (2 * calibration_amplitude)
                
#                 sysi.add_dm2(-s * calibration_amplitude * calibration_mode) # remove the calibrated mode from DM2
                
                
#                 # Now repeat for the second set of probes
                
#                 # DM1: Set the DM to the correct state
#                 sysi.add_dm1(s * calibration_amplitude * calibration_mode)
#                 differential_images, single_images = take_measurement(sysi, probe_modes_2, probe_amplitude, DM=1,
#                                                                           return_all=True)
                
#                 images_11.append(single_images)
#                 slope21 += s * differential_images / (2 * calibration_amplitude)
                
#                 sysi.add_dm1(-s * calibration_amplitude * calibration_mode) # remove the calibrated mode from DM1
                
#                 # DM2: Set the DM to the correct state
#                 sysi.add_dm2(s * calibration_amplitude * calibration_mode)
#                 differential_images, single_images = take_measurement(sysi, probe_modes_2, probe_amplitude, DM=1, 
#                                                                           return_all=True)
                
#                 images_22.append(single_images)
#                 slope22 += s * differential_images / (2 * calibration_amplitude)
                
#                 sysi.add_dm2(-s * calibration_amplitude * calibration_mode) # remove the calibrated mode from DM2
                
                
#             print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1+start_mode, calibration_modes.shape[0], 
#                                                                     time.time()-start))
#             slopes_11.append(slope11)
#             slopes_12.append(slope12)
#             slopes_21.append(slope21)
#             slopes_22.append(slope22)
#         except KeyboardInterrupt: 
#             print('Calibration interrupted.')
#             break
    
#     slopes_11, slopes_12, slopes_21, slopes_22 = (np.array(slopes_11), np.array(slopes_12), np.array(slopes_21), np.array(slopes_22))
#     images_11, images_12, images_21, images_22 = (np.array(images_11), np.array(images_12), np.array(images_21), np.array(images_22)

#     slopes = np.concatenate((slopes_11,slopes_12,slopes_21,slopes_22), axis=0) # this is the response cube
#     images = np.concatenate((images_11,images_12,images_21,images_22), axis=0) # this is the calibration cube
    
#     print('Calibration complete.')
#     return slopes, images

def WeightedLeastSquares(A, W, rcond=1e-15):
    cov = A.T.dot(W.dot(A))
    return np.linalg.inv(cov + rcond * np.diag(cov).max() * np.eye(A.shape[1])).dot( A.T.dot(W) )

def TikhonovInverse(A, rcond=1e-15):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

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
        control_matrix_1 = WeightedLeastSquares(masked_matrix[:,:nmodes], Wmatrix, rcond=rcond1)
        control_matrix_2 = WeightedLeastSquares(masked_matrix[:,nmodes:], Wmatrix, rcond=rcond2)
    else: 
        print('Using Tikhonov Inverse')
        control_matrix_1 = TikhonovInverse(masked_matrix[:,:nmodes], rcond=rcond1)
        control_matrix_2 = TikhonovInverse(masked_matrix[:,nmodes:], rcond=rcond2)
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
    








