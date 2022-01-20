"""
capture frames while changing DM settings with Olivier's probe cube

2021-02-02
   initial attempt
   no burst mode because ipc server
2021-03-18
   adopted from olivier_probes_20210202_script.py
2021-03-30
   adopted from sebastiaan_probes_example_script.
2021-04-09
   After several iterations the code is working in simulation. This validates the logic of the functions.
2021-04-14
	Remove all references to the PIAA testbed and make it agnostic to the actual system
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from datetime import date
import astropy.io.fits as pyfits
from scipy.sparse import linalg as sLA

def remove_k_pca_modes(data_cube, k=2):
	''' This function removes the k strongest PCA modes.
	'''
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

def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
	differential_operator = np.array([[-1,1,0,0],[0,0,-1,1]]) / (2 * probe_amplitude * system_interface.texp)

	# Measure the response
	images = []
	for probe in probe_cube:
		for s in [-1.0, 1.0]:
			# Add the probe to the DM
			system_interface.add_dm(s * probe_amplitude * probe)

			# Measure the response
			system_interface.send_dm()
			image = system_interface.snap()
			images.append(image)
			
			# Remove the probe from the DM
			system_interface.add_dm(-s * probe_amplitude * probe)
	
	system_interface.send_dm()

	images = np.array(images)
	differential_images = differential_operator.dot(images)
	
	if pca_modes is not None:
		differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T

	if return_all:
		return differential_images, images
	else:
		return differential_images

def IEFC_single_iteration(system_interface, probe_cube, probe_amplitude, control_matrix, pixel_mask_dark_hole):
	
	# Take a measurement
	differential_images = take_measurement(system_interface, probe_cube, probe_amplitude)

	# Choose which pixels we want to control
	measurement_vector = differential_images[:, pixel_mask_dark_hole].ravel()

	# Calculate the control signal in modal coefficients
	reconstructed_coefficients = control_matrix.dot( measurement_vector )

	return reconstructed_coefficients

def take_background(system_interface, exposure_times, num_images):
	current_texp = system_interface.texp
	
	system_interface.log('Taking new background images.')
	system_interface.set_shutter(False)
	
	for ti, ni in zip(exposure_times, num_images):
		system_interface.log('Taking {:d} exposures with {:g} exposure time.'.format(ni, ti))
		system_interface.texp = ti
		system_interface.set_mode('background_{:g}'.format(ti).replace('.', 'p'))

		for i in range(ni):
			system_interface.snap()
	
	system_interface.set_mode('run')
	system_interface.set_shutter(True)
	system_interface.texp = current_texp
	
	
def run_IEFC_calibration(system_interface, probe_amplitude, probe_modes, calibration_amplitude, calibration_modes, start_mode=0):
	
	system_interface.log('Taking a new calibration.')
	system_interface.set_mode('calib')
	system_interface.update_run_number()
	system_interface.log('Update run number to {:d}.'.format(system_interface.run_number))
		
	slopes = []
	images = []
	# Loop through all modes that you want to control
	for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
		if ci % 5 == 0:
			print("Calibrating mode {:d} / {:d}".format(ci+1+start_mode, calibration_modes.shape[0]))
		
		try:
			slope = 0
			# We need a + and - probe to estimate the jacobian
			for s in [-1, 1]:
				# Set the DM to the correct state
				system_interface.add_dm(s * calibration_amplitude * calibration_mode)
				
				differential_images, single_images = take_measurement(system_interface, probe_modes, probe_amplitude, return_all=True)
				slope += s * differential_images / (2 * calibration_amplitude)
				images.append(single_images)
				
				# Remove the calibrated mode
				system_interface.add_dm(-s * calibration_amplitude * calibration_mode)
			
			slopes.append(slope)
		except KeyboardInterrupt:
			break
			
	system_interface.send_dm()
	system_interface.log('Calibration done. Took {:d} images.'.format(len(images)))
	system_interface.set_mode('run')
	
	return np.array(slopes), np.array(images)

def create_dshaped_focal_plane_mask(x, y, inner_radius, outer_radius, edge_position=0, direction='-y'):

	r = np.hypot(x, y)
	mask = (r < outer_radius) * (r > inner_radius)
	
	if direction == '+x':
		mask *= (x > edge_position)
	elif direction == '-x':
		mask *= (x < -edge_position)
	elif direction == '+y':
		mask *= (y > edge_position)
	elif direction == '-y':
		mask *= (y < -edge_position)

	return mask

def create_box_focal_plane_mask(x, y, x0, y0, width, height):

	mask = ( abs(x - x0) < width/2 ) * ( abs(y - y0) < height/2 )

	return mask > 0

def load_calibration_files(path, date, calibration_runs, number_of_modes):
	images = []

	for cal_run, ni in zip(calibration_runs, number_of_modes):
		subdir = 'iefc_{:s}_run_{:d}_{:s}'.format(date, cal_run, 'calib')
		
		for i in range(ni):
			for j in range(8):
				filename = path + subdir + '/piaa.{:0>6d}.fits'.format(j + 8 * i + 1)
				image = pyfits.getdata(filename)
				images.append(image.ravel())
				
	return np.array(images)

def reconstruct_response(image_cube, texp, probe_amplitude, calibration_amplitude):
	differential_operator = np.array([[-1,1,0,0],[0,0,-1,1]]) / (2 * probe_amplitude * texp)
	number_of_modes = image_cube.shape[0] // 8
	
	slopes = []
	for i in range(number_of_modes):
		
		minus_response = differential_operator.dot(image_cube[(8*i):(4+8*i)])
		plus_response = differential_operator.dot(image_cube[(4+8*i):(8+8*i)])
	
		slope = (plus_response - minus_response) / (2 * calibration_amplitude)
		slopes.append(slope)

	return np.array(slopes)

'''
def write_fits(data, filename, header={}, reference_image=None):
	print('writing image cube to: ', filename)
	hducube = pyfits.PrimaryHDU(data)
	hducube.header.update(header)

	if reference_image is not None:
		hduref  = pyfits.ImageHDU(reference_image)
		pyfits.HDUList([hducube, hduref]).writeto(filename)
	else:
		pyfits.HDUList([hducube,]).writeto(filename)
'''

def create_fourier_modes(xfp, mask, Nact=34, use_both=True):
	'''
		Parameters
		xfp - array_like
			The one-dimensional focal plane coordinates (in lambda/D)

		mask - array_like
			The focal plane dark hole mask

		Nact - int
			The number of actuators across the DM
		Return
			The mode matrix
	'''
	#print("create_fourier_modes: ", mask.shape)
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

	# Select all Fourier modes of interest based on the dark hole mask
	# and remove the piston mode
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

def create_probe_sinc_modes(Nact, iwa, owa, width):

	xs = np.linspace(-0.5, 0.5, Nact) * 34/32
	x, y = np.meshgrid(xs, xs)

	cx = 2 * np.pi * 0
	cy = 2 * np.pi * (owa + iwa)/2
	wx = width + 2
	wy = (owa-iwa + 2)

	mode1 = np.sinc(wx * x) * np.sinc(wy * y) * np.cos(cx * x + 3 * np.pi/4) * np.cos(cy * y + 3 * np.pi/4)
	mode1 /= np.std(mode1)

	mode2 = np.sinc(wx * x) * np.sinc(wy * y) * np.cos(cx * x + np.pi/4) * np.cos(cy * y + np.pi/4)
	mode2 /= np.std(mode2)

	return np.array([mode1.ravel(), mode2.ravel])