import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy import interpolate
import time
from datetime import date
import astropy.io.fits as pyfits
from scipy.sparse import linalg as sLA

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
    differential_operator = np.array([ [-1,1,0,0] , [0,0,-1,1] ]) / (2 * probe_amplitude * sysi.texp)
    
#     # Measure the response
#     images = []
#     for probe in probe_cube:
#         for s in [-1.0, 1.0]:
#             # Add the probe to the DM
# #             system_interface.add_dm(s * probe_amplitude * probe)
#             sysi.add_dm1(s * probe_amplitude * probe)
            
#             # Measure the response
# #             sysi.send_dm()
# #             image = system_interface.snap()
#             image = sysi.calc_psf()
#             images.append(image.flatten())
            
#             # Remove the probe from the DM
# #             system_interface.add_dm(-s * probe_amplitude * probe)
#             sysi.add_dm1(-s * probe_amplitude * probe)

    dm1_commands = [-1.0*probe_amplitude*probe_cube[0].reshape(sysi.Nact,sysi.Nact),
                    1.0*probe_amplitude*probe_cube[0].reshape(sysi.Nact,sysi.Nact),
                    -1.0*probe_amplitude*probe_cube[1].reshape(sysi.Nact,sysi.Nact),
                    1.0*probe_amplitude*probe_cube[1].reshape(sysi.Nact,sysi.Nact)]
    dm2_commands = [np.zeros((48,48)), np.zeros((48,48)), np.zeros((48,48)), np.zeros((48,48))]
    wfs = sysi.calc_psfs(dm1_commands, dm2_commands)
    
    images=[]
    for i,wf in enumerate(wfs): images.append(wf.intensity.flatten())
    
#     system_interface.send_dm()
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
    slopes = []
    images = []
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        try:
            slope = 0
            # We need a + and - probe to estimate the jacobian
            for s in [-1, 1]:
                # Set the DM to the correct state
#                 system_interface.add_dm(s * calibration_amplitude * calibration_mode)
                sysi.add_dm1(s * calibration_amplitude * calibration_mode)
                differential_images, single_images = take_measurement(sysi, probe_modes, probe_amplitude, return_all=True)
                
                slope += s * differential_images / (2 * calibration_amplitude)
                images.append(single_images)
                
                # Remove the calibrated mode
#                 system_interface.add_dm(-s * calibration_amplitude * calibration_mode)
                sysi.add_dm1(-s * calibration_amplitude * calibration_mode)
            print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1+start_mode, calibration_modes.shape[0], 
                                                                    time.time()-start))
            slopes.append(slope)
        except KeyboardInterrupt: 
            print('Calibration interrupted.')
            break
    print('Calibration complete.')
    return np.array(slopes), np.array(images)

def single_iteration(sysi, probe_cube, probe_amplitude, control_matrix, pixel_mask_dark_hole):
    # Take a measurement
    differential_images = take_measurement(sysi, probe_cube, probe_amplitude)
    
    # Choose which pixels we want to control
    measurement_vector = differential_images[:, pixel_mask_dark_hole].ravel()
    
    # Calculate the control signal in modal coefficients
    reconstructed_coefficients = control_matrix.dot( measurement_vector )
    
    return reconstructed_coefficients

def create_fourier_modes(xfp, mask, Nact=34, use_both=True):
    print("Createing Fourier modes: ", mask.shape)
    intp = interpolate.interp2d(xfp, xfp, mask)
    
    # This creates the grid and frequencies
    xs = np.linspace(-0.5, 0.5, Nact) * (Nact-1)
    x, y = np.meshgrid(xs, xs)
    x = x.ravel()
    y = y.ravel()
    
    # Create the fourier frequencies. An odd number of modes is preferred for symmetry reasons.
    if Nact % 2 == 0: 
        fxs = np.fft.fftshift( np.fft.fftfreq(Nact+1) )
    else:
        fxs = np.fft.fftshift( np.fft.fftfreq(Nact) )
        
    fx, fy = np.meshgrid(fxs, fxs)
    
    # Select all Fourier modes of interest based on the dark hole mask and remove the piston mode
    mask2 = intp(fxs * Nact, fxs * Nact) * (((fx!=0) + (fy!=0)) > 0) > 0
    
    fx = fx.ravel()[mask2.ravel()]
    fy = fy.ravel()[mask2.ravel()]
    
    # The modes can rewritten to a single (np.outer(x, fx) + np.outer(y, fy))
    if use_both:
        M1 = [np.cos(2 * np.pi * (fi[0] * x + fi[1] * y)) for fi in zip(fx, fy)]
        M2 = [np.sin(2 * np.pi * (fi[0] * x + fi[1] * y)) for fi in zip(fx, fy)]
        
        # Normalize the modes
        M = np.array(M1+M2)
    else:
        M = np.array([np.sin(2 * np.pi * (fi[0] * x + fi[1] * y)) for fi in zip(fx, fy)])
        
    M /= np.std(M, axis=1, keepdims=True)
    return M, fx, fy

def create_probe_poke_modes(Nact, indx0, indy0, indx1, indy1):
    probe_modes = np.zeros((2, Nact, Nact))
    probe_modes[0, indy0, indx0] = 1
    probe_modes[1, indy1, indx1] = 1
    probe_modes = probe_modes.reshape((2, -1))
    return probe_modes


def remove_k_pca_modes(data_cube, k=2):
    ''' This function removes the k strongest PCA modes.'''
    temp_cube = data_cube.reshape((-1, data_cube.shape[-1]))
    
    U, s, V = sLA.svds(temp_cube, k=k)
    filtered_data = temp_cube - (V.T.dot( V.dot(temp_cube.T) )).T
    return filtered_data.reshape(data_cube.shape), V

def WeightedLeastSquares(A, W, rcond=1e-15):
    cov = A.T.dot(W.dot(A))
    return np.linalg.inv(cov + rcond * np.diag(cov).max() * np.eye(A.shape[1])).dot( A.T.dot(W) )

def TikhonovInverse(A, rcond=1e-15):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def construct_control_matrix(response_matrix, weight_map, rcond=1e-2, pca_modes=None):
    weight_mask = weight_map>0
    
    # Invert the matrix with an SVD and Tikhonov regularization
    masked_matrix = response_matrix[:, :, weight_mask].reshape((response_matrix.shape[0], -1)).T
    
    # Add the extra PCA modes that are fitted
    if pca_modes is not None:
        double_pca_modes = np.concatenate( (pca_modes[:, weight_mask], pca_modes[:, weight_mask]), axis=1).T
        masked_matrix = np.hstack((masked_matrix, double_pca_modes))
        
    Wmatrix = np.diag(np.concatenate((weight_map[weight_mask], weight_map[weight_mask])))
    control_matrix = WeightedLeastSquares(masked_matrix, Wmatrix, rcond=rcond)
    
    if pca_modes is not None:
        # Return the control matrix minus the pca_mode coefficients
        return control_matrix[0:-pca_modes.shape[0]]
    else:
        return control_matrix


def create_rect_patch(rect_params):
    rect_patch = Rectangle((rect_params['x0']-rect_params['w']/2, rect_params['y0']-rect_params['h']/2), 
                           rect_params['w'], rect_params['h'], color='c', fill=False)
    return rect_patch






