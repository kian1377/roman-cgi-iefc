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

dm1_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm1.fits')
dm2_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm2.fits')

reload(cgi_phasec_poppy)
mode = cgi_phasec_poppy.cgi.CGI(cgi_mode='hlc', npsf=60,
                                dm1_ref=dm1_flat,
                                dm2_ref=dm2_flat,
                              use_pupil_defocus=True, 
                              use_opds=True)

mode.use_fpm = False
ref_unocc_im = mode.snap()
imshow1(ref_unocc_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

mode.Imax_ref = ref_unocc_im.get().max()
mode.use_fpm = True

ref_im = mode.snap()
imshow1(ref_im, 'HLC Initial Coronagraphic Image\nfor iEFC',
        pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True, vmin=1e-11)



reload(utils)
roi1 = utils.create_annular_focal_plane_mask(mode, inner_radius=3, outer_radius=9, edge=0.5, plot=True)
roi2 = utils.create_annular_focal_plane_mask(mode, inner_radius=2.8, outer_radius=9.7, edge=0.5, plot=True)
roi3 = utils.create_annular_focal_plane_mask(mode, inner_radius=3.2, outer_radius=6, edge=0.5, plot=True)

relative_weight_1 = 0.9
relative_weight_2 = 0.4
weight_map = roi3 + relative_weight_1*(roi1*~roi3) + relative_weight_2*(roi2*~roi1*~roi3)
control_mask = weight_map>0
imshow2(weight_map, control_mask*ref_im, lognorm2=True)
mean_ni = xp.mean(ref_im[control_mask])
print(mean_ni)


probe_amp = 2.5e-8
probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.5, shift=(0,14), nprobes=3, plot=True)


calib_amp = 5e-9
calib_modes = utils.create_all_poke_modes(mode.dm_mask)

response_matrix, response_cube = iefc_2dm.calibrate(mode, 
                                                    control_mask,
                                                    probe_amp, probe_modes, 
                                                    calib_amp, calib_modes, 
                                                    return_all=True)

utils.save_fits(response_dir/f'hlc_iefc_2dm_poke_response_matrix_{today}.fits', ensure_np_array(response_matrix))
utils.save_fits(response_dir/f'hlc_iefc_2dm_poke_response_cube_{today}.fits', ensure_np_array(response_cube))











