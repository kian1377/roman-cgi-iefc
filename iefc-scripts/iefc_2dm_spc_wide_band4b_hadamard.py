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
from IPython.display import clear_output, display, HTML
display(HTML("<style>.container { width:90% !important; }</style>")) # just making the notebook cells wider
from datetime import datetime
today = int(datetime.today().strftime('%Y%m%d'))

from importlib import reload
import time


import ray
if not ray.is_initialized():
    ray.init(log_to_driver=False)
    

import logging, sys
poppy_log = logging.getLogger('poppy')
poppy_log.setLevel('DEBUG')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

poppy_log.disabled = True

import warnings
warnings.filterwarnings("ignore")

import cgi_phasec_poppy
    
from math_module import xp, ensure_np_array
import iefc_2dm 
import utils
from imshows import *

data_dir = iefc_2dm.iefc_data_dir
response_dir = data_dir/'response-data'

dm1_flat = fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'spc_wide_band4_flattened_dm1.fits')
dm2_flat = fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'spc_wide_band4_flattened_dm2.fits')

wavelength_c = 825e-9*u.m

nwaves = 3
bandwidth = 3.6/100
minwave = wavelength_c * (1 - bandwidth/2)
maxwave = wavelength_c * (1 + bandwidth/2)
wavelengths = np.linspace( minwave, maxwave, nwaves )

reload(cgi_phasec_poppy.cgi)
reload(cgi_phasec_poppy.parallelized_cgi)

rayCGI = ray.remote(cgi_phasec_poppy.cgi.CGI) # make a ray actor class from the original CGI class  

kwargs = {
    'cgi_mode':'spc-wide',
    'npsf':150,
    'use_pupil_defocus':True,
    'use_opds':True,
    'polaxis':0,
}

actors = []
for i in range(nwaves):
    actors.append(rayCGI.options(num_cpus=2, num_gpus=1/8).remote(**kwargs))
    actors[i].setattr.remote('wavelength', wavelengths[i])

mode = cgi_phasec_poppy.parallelized_cgi.ParallelizedCGI(actors=actors, 
                                                         dm1_ref=2*dm1_flat, 
                                                        #  dm2_ref=dm2_flat, 
                                                         )
mode.set_actor_attr('use_fpm',False)
raw_im = mode.snap(quiet=False)

mode.Imax_ref = xp.max(raw_im.max())
ref_unocc_im = mode.snap(quiet=False)
imshow2(raw_im, ref_unocc_im, '', f'{xp.max(ref_unocc_im):.2f}', 
        lognorm=True, vmin1=mode.Imax_ref*1e-6, vmin2=1e-6)

mode.set_actor_attr('use_fpm',True)
ref_im = mode.snap(quiet=False)

control_mask = utils.create_annular_focal_plane_mask(mode, inner_radius=5.4, outer_radius=20.6, edge=None)
mean_ni = xp.mean(ref_im[control_mask])
imshow3(ref_im*mode.Imax_ref, ref_im, ref_im*control_mask, 
        f'Reference/Initial State: {xp.max(ref_im*mode.Imax_ref):.2e}', 
        'Normalized Reference Image',
        f'Mean NI: {mean_ni:.2e}',
        lognorm=True,)



probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.2,
                                          shift=[(-12,7), (12,7),(0,-14), (0,0)], nprobes=3,
                                           use_weighting=True)
imshow3(probe_modes[0], probe_modes[1], probe_modes[2])

calib_modes = utils.create_hadamard_modes(mode.dm_mask, ndms=2)
Nmodes = calib_modes.shape[0]
print(calib_modes.shape)
i = 8
imshow2(calib_modes[i,:mode.Nact**2].reshape(mode.Nact,mode.Nact), calib_modes[i+mode.Nacts,mode.Nact**2:].reshape(mode.Nact,mode.Nact))

probe_amp = 20e-9
calib_amp = 5e-9

response_matrix, response_cube, calib_amps = iefc_2dm.calibrate(mode, 
                                                                control_mask,
                                                                probe_amp, probe_modes, 
                                                                calib_amp, calib_modes, 
                                                                return_all=True, 
            #                                                     plot_responses=False,
                                                                )

utils.save_fits(response_dir/f'spc_wide_band4b_had_modes_response_matrix_{today}.fits', response_matrix)

# iefc_2dm_spc_wide_band4b_hadamard.py









