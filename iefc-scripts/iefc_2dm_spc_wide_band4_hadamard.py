import sys
sys.path.append("..")

import numpy as np
import scipy

import cupy as cp
import cupyx.scipy

import astropy.units as u
from astropy.io import fits
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
import copy 

from IPython.display import clear_output, display, HTML
display(HTML("<style>.container { width:90% !important; }</style>")) # just making the notebook cells wider

from datetime import datetime
today = int(datetime.today().strftime('%Y%m%d'))

from importlib import reload
import time

import logging, sys
poppy_log = logging.getLogger('poppy')
poppy_log.setLevel('DEBUG')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

poppy_log.disabled = True

import warnings
warnings.filterwarnings("ignore")

import cgi_phasec_poppy

import ray
if not ray.is_initialized():
    ray.init(log_to_driver=False)
    
from math_module import xp, _scipy, ensure_np_array
import iefc_2dm 
import utils
from imshows import *

data_dir = iefc_2dm.iefc_data_dir
response_dir = data_dir/'response-data'

dm1_flat = fits.getdata('../spc_wide_band4_flattened_dm1.fits')
dm2_flat = fits.getdata('../spc_wide_band4_flattened_dm2.fits')

nwaves = 5
wavelength_c = 825e-9*u.m
bandwidth = 10/100
minwave = wavelength_c * (1 - bandwidth/2)
maxwave = wavelength_c * (1 + bandwidth/2)
wavelengths = np.linspace( minwave, maxwave, nwaves )


mode = cgi_phasec_poppy.cgi.CGI(cgi_mode='spc-wide', npsf=150,
                                use_pupil_defocus=True, 
                                use_opds=True,
                                dm1_ref=2*dm1_flat, 
                                # dm2_ref=dm2_flat,
                                )

mode.wavelengths = wavelengths
mode.exp_times_list = None
mode.use_fpm = False
raw_im = mode.snap()

mode.Imax_ref = raw_im.get().max()
ref_unocc_im = mode.snap()

mode.use_fpm = True
ref_im = mode.snap()

control_mask = utils.create_annular_focal_plane_mask(mode, inner_radius=5.4, outer_radius=20.6, edge=None)
mean_ni = xp.mean(ref_im[control_mask])
imshow3(ref_unocc_im, ref_im, ref_im*control_mask, 
        f'Reference/Initial State', 
        'Normalized Reference Image',
        f'Mean NI: {mean_ni:.2e}',
        lognorm=True,
        save_fig='test_bb_ims.png',
        )

reload(utils)
probe_amp = 20e-9
# probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.2, shift=[(-12,6), (12,6), (0,-12)], nprobes=3, plot=True)
# probe_modes = utils.create_poke_probes([(10,34), (38,34), (24,10)], plot=True)
# probe_modes = utils.create_poke_probes([(11,31), (36,31), (23,9)], plot=True)
probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.25,
                                          shift=[(-12,7), (12,7),(0,-14), (0,0)], nprobes=3,
                                          use_weighting=True)

# imshow3(probe_modes[0], probe_modes[1], probe_modes[2], save_fig='test_probes.png')
utils.save_fits(response_dir/f'test_bb_probes_{today}.fits', probe_modes)

calib_amp = 5e-9
calib_modes = utils.create_hadamard_modes(mode.dm_mask, ndms=2)

# date = 20240129
# response_matrix = xp.array(fits.getdata(response_dir/f'spc_wide_825_had_modes_response_matrix_{date}.fits'))
# dm_response = xp.sqrt(xp.mean(response_matrix.dot(xp.array(calib_modes))**2, axis=0))
# dm1_response = dm_response[:mode.Nact**2].reshape(mode.Nact, mode.Nact)/dm_response.max()
# dm2_response = dm_response[mode.Nact**2:].reshape(mode.Nact, mode.Nact)/dm_response.max()
# print(dm_response.max())
# imshow2(dm1_response, dm2_response, 
#         'RMS Response of DM1\nActuators with Hadamard Modes', 'RMS Response of DM2\nActuators with Hadamard Modes',
#         lognorm=True, vmin1=0.01, vmin2=0.01,
#         )
# dm_mask = dm1_response>9e-2
# calib_modes = utils.create_hadamard_modes(ensure_np_array(dm_mask), ndms=2)
# print(calib_modes.shape)

response_matrix, response_cube, calib_amps = iefc_2dm.calibrate(mode, 
                                                                control_mask,
                                                                probe_amp, probe_modes, 
                                                                calib_amp, calib_modes, 
                                                                return_all=True, 
                                                                # plot_responses=False,
                                                                )

utils.save_fits(response_dir/f'spc_wide_band4_had_modes_response_matrix_{today}.fits', response_matrix)

# iefc_2dm_spc_wide_band4_had_modes.py









