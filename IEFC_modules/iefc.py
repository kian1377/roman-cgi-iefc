import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from datetime import date
import astropy.io.fits as pyfits
from scipy.sparse import linalg as sLA

def 

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
    
def calibrate_iefc(probe_amplitude, probe_modes, calibration_amplitude, calibration_modes, start_mode=0):
    print('Calibrating I-EFC...')
    slopes = []
    images = []
    # Loop through all modes that you want to control
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        if ci % 5 == 0: print("\tCalibrating mode {:d} / {:d}".format(ci+1+start_mode, calibration_modes.shape[0]))
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
    print('Calibration complete.')
    
    return np.array(slopes), np.array(images)




















