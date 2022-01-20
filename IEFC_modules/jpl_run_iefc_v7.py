from Simulation.iefc_functions import load_calibration_files
import numpy as np
from matplotlib import pyplot as plt
import astropy.io.fits as pyfits
import importlib
import iefc_functions
import piaa_interface

### RUN STARTUP SCRIPT ###
# first align source, cmc
# lyot in
# cmc in
# vflat is defined in startup

# Create a grid for the focal plane
Nact = 34	# Number of actuators across the DM
q = 5	# pixels per lambda / D
num_pixels = 216
xs = np.linspace(-0.5, 0.5, num_pixels) * num_pixels / q
x, y = np.meshgrid(xs, xs)

### Set the testbed settings
#gain_path = '/home/dmarx/HCIT/PIAA/hcim_testbed_run100/dms/25CW011029/dmgains_20200130.fits
gain_path = '/proj/piaacmc/DM/25CW011029/dmgains_20200130.fits'

tbi = piaa_interface.PIAATestbedInterface(tb, dmcoef, lys, vflat, gain_path)
tbi.texp = 1e-2
tbi.delay = 1.0

## Control the source
SetBand(2)

##
# Create the mask that is used to select which region to make dark.
dark_hole_options = {
    'x0' : 0,
    'y0' : -6,
    'w' : 10,
    'h' : 8
}
dark_hole_mask = iefc_functions.create_box_focal_plane_mask(x, y, dark_hole_options['x0'], dark_hole_options['y0'], dark_hole_options['w'], dark_hole_options['h']).ravel()

#Create the mask that is used to select which region to make dark.
control_region_options = {
    'x0' : 0,
    'y0' : -8,
    'w' : 10,
    'h' : 10
}
control_region_mask = iefc_functions.create_box_focal_plane_mask(x, y, control_region_options['x0'], control_region_options['y0'], control_region_options['w'], control_region_options['h']).ravel()
full_mask = control_region_mask * (abs(y).ravel()>0.5)

# Create the fourier modes
fourier_modes, fx, fy = iefc_functions.create_fourier_modes(xs, control_region_mask.reshape((num_pixels, num_pixels)), Nact)

# Create the probe cube
probe_modes = iefc_functions.create_probe_poke_modes(Nact, Nact//2, Nact//4, Nact//2, Nact//4-1)

# Calibration settings
um = 1e-6
wavelength = 0.635 * um
probe_amplitude = 0.05 * wavelength
calibration_amplitude = 0.006 * wavelength

# Background and detector calibration
iefc_functions.take_background(tbi, [tbi.texp,], [10,])

# Calibrate the IEFC method
response_cube, calibration_cube = iefc_functions.run_IEFC_calibration(tbi, probe_amplitude, probe_modes, calibration_amplitude, fourier_modes, start_mode=0)

# Filter the calibration files by subtracting PCA modes
npca = 3
filtered_response_cube, pca_modes = iefc_functions.remove_k_pca_modes(response_cube, k=npca)

load_calibration = False
if load_calibration:
	response_cube = 0
	calibration_cube = 0
	path = ''
	calibration_cube = iefc_functions.load_calibration_files(path, tbi.date, [1,2], [660, 1052-660])
	response_cube = iefc_functions.reconstruct_response(calibration_cube, 5e-3, probe_amplitude, calibration_amplitude)


# Create the control matrix
relative_weight = 0.99
weights = dark_hole_mask * relative_weight + (1 - relative_weight) * full_mask

control_matrix = iefc_functions.construct_control_matrix(filtered_response_cube, weights, rcond=3e-4, pca_modes=pca_modes)

# The metric
metric_images = []
command = 0.0

## Run IEFC for num_iterations
num_iterations = 50
gain = -0.8
leakage = 0.0

for i in range(num_iterations):
    print("Closed-loop iteration {:d} / {:d}".format(i+1, num_iterations))
    delta_coefficients = iefc_functions.IEFC_single_iteration(tbi, probe_modes, probe_amplitude, control_matrix, weights>0)
    command = (1.0-leakage) * command + gain * delta_coefficients
    # Reconstruct the full phase from the Fourier modes
    voltage_command = np.einsum('ij, i->j', fourier_modes, command)
    # Set the current DM state
    tbi.set_dm(voltage_command)
    ret = tbi.send_dm()
    # Take an image to estimate the metrics
    metric_images.append(tbi.snap())
    pass



tbi.write_log_to_file('iefc_run_log')