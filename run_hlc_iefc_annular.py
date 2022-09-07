import numpy as np
import cupy as cp
import astropy.units as u
from astropy.io import fits
from matplotlib.patches import Rectangle, Circle

import os, shutil
from pathlib import Path
from IPython.display import clear_output
from importlib import reload

import logging, sys
poppy_log = logging.getLogger('poppy')
poppy_log.setLevel('INFO')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

poppy_log.disabled = True

import warnings
warnings.filterwarnings("ignore")

import cgi_phasec_poppy as cgi
from cgi_phasec_poppy import misc

from wfsc import iefc_2dm as iefc
from wfsc import utils

# Set the name for this iteration of simulation
data_set_name = 'hlc_annular_iefc_sim_v1'

iefc_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')

if os.path.exists(str(iefc_dir/data_set_name)):
    shutil.rmtree(str(iefc_dir/data_set_name))
os.mkdir(str(iefc_dir/data_set_name))

dm_dir = cgi.data_dir/'dm-acts'

dm1_flat = 2*fits.getdata(dm_dir/'flatmaps'/'hlc_flattened_dm1.fits')
dm2_flat = 2*fits.getdata(dm_dir/'flatmaps'/'hlc_flattened_dm2.fits')

hlc = cgi.CGI(cgi_mode='hlc', 
              use_fpm=True,
              use_pupil_defocus=False, 
              polaxis=0,
              use_opds=True,
              dm1_ref=dm1_flat, dm2_ref=dm2_flat,
             )
hlc.show_dms()

npsf = hlc.npsf
Nact = hlc.Nact

reload(utils)
xfp = np.linspace(-0.5, 0.5, npsf) * npsf * hlc.psf_pixelscale_lamD
xf,yf = np.meshgrid(xfp,xfp)

edge = 1
iwa = 3
owa = 6
rot = 0

# Create the dark-hole mask.
dh_params = {
    'inner_radius' : iwa,
    'outer_radius' : owa,
    'edge_position' : edge,
    'direction' : '+x',
    'rotation':rot,
    'full':True,
}
dh_mask = utils.create_annular_focal_plane_mask(xf, yf, dh_params).ravel()

# Create the control region mask.
control_params = {
    'inner_radius' : iwa-0.2,
    'outer_radius' : owa+0.7,
    'edge_position' : edge,
    'rotation':rot,
    'full':True,
}
control_mask = utils.create_annular_focal_plane_mask(xf, yf, control_params).ravel()

relative_weight = 0.95
weights = dh_mask * relative_weight + (1 - relative_weight) * control_mask

# Create probe and calibration modes
probe_modes = iefc.create_probe_poke_modes(Nact, 
                                           xinds=[Nact//4, Nact//4+1],
                                           yinds=[Nact//4, Nact//4], 
                                           display=True)

calibration_modes, fx, fy = iefc.create_fourier_modes(xfp, 
                                                      control_mask.reshape((npsf,npsf)), 
                                                      Nact, 
                                                      circular_mask=False)
calibration_modes[:] *= hlc.dm_mask.flatten()

nmodes = calibration_modes.shape[0]
print('Calibration modes required: {:d}'.format(nmodes))

calibration_amplitude = 0.006 * hlc.wavelength_c.to(u.m).value
probe_amplitude = 0.05 * hlc.wavelength_c.to(u.m).value

n_calibrations = 10
n_iefc_iterations_per_calib = 2

response_cubes = []
images = []
dm1_acts = []
dm2_acts = []
for i in range(n_calibrations):
    
    response_cube, calibration_cube = iefc.calibrate(hlc, probe_amplitude, probe_modes, 
                                                     calibration_amplitude, calibration_modes, start_mode=0)
    
    control_matrix = iefc.construct_control_matrix(response_cube, 
                                                   weights.flatten(), 
                                                   rcond1=1e-2,
                                                   rcond2=1e-2,
                                                   nprobes=probe_modes.shape[0], pca_modes=None)
    
    ims, dm1_commands, dm2_commands = iefc.run(hlc, 
                                                  control_matrix, 
                                                  probe_modes, 
                                                  probe_amplitude, 
                                                  calibration_modes, 
                                                  weights, 
                                                  num_iterations=n_iefc_iterations_per_calib, 
                                                  gain=-0.5, leakage=0.0,
                                                  display=False)
    
    response_cubes.append(response_cube)
    images.append(ims)
    dm1_acts.append(dm1_commands)
    dm2_acts.append(dm2_commands)

misc.save_pickle(iefc_dir/data_set_name/'reponse_cubes', reponse_cubes)
misc.save_pickle(iefc_dir/data_set_name/'images', images)
misc.save_pickle(iefc_dir/data_set_name/'dm1_acts', dm1_acts)
misc.save_pickle(iefc_dir/data_set_name/'dm2_acts', dm2_acts)

    
    