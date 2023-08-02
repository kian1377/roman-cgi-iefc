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

imshow2(dm1_flat/2, dm2_flat/2, 'DM1 Initial State', 'DM2 Initial State', cmap1='viridis', cmap2='viridis')


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
roi2 = utils.create_annular_focal_plane_mask(mode, inner_radius=2.5, outer_radius=9.7, edge=0.5, plot=True)
roi3 = utils.create_annular_focal_plane_mask(mode, inner_radius=3.2, outer_radius=6, edge=0.5, plot=True)

relative_weight_1 = 0.9
relative_weight_2 = 0.4
weight_map = roi3 + relative_weight_1*(roi1*~roi3) + relative_weight_2*(roi2*~roi1*~roi3)
control_mask = weight_map>0
imshow2(weight_map, control_mask*ref_im, lognorm2=True)
mean_ni = xp.mean(ref_im[control_mask])
print(mean_ni)

# choose probe modes
probe_amp = 2.5e-8
probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.5, shift=(14,0), nprobes=2, plot=True)

# choose calibration modes 
calib_amp = 5e-9
calib_modes = utils.create_all_poke_modes(mode.dm_mask)

Ncalibs = 6
Nitr = 5
for i in range(Ncalibs):
    print('Calibrations {:d}/{:d}'.format(i+1, Ncalibs))
    response_matrix, response_cube = iefc_2dm.calibrate(mode, 
                                                         control_mask,
                                                         probe_amp, probe_modes, 
                                                         calib_amp, calib_modes, 
                                                         return_all=True)
    
    utils.save_fits(response_dir/f'hlc_iefc_2dm_response_matrix_{i+1}_{Ncalibs}_{today}.fits', ensure_np_array(response_matrix))
    utils.save_fits(response_dir/f'hlc_iefc_2dm_response_cube_{i+1}_{Ncalibs}_{today}.fits', ensure_np_array(response_cube))
    
    reg_conds = [(1e-2,3), (1e-1,2)]
    reg_fun = utils.WeightedLeastSquares
    reg_kwargs = {
        'weight_map':weight_map,
        'nprobes':probe_modes.shape[0],
    }

    images, dm1_commands, dm2_commands = iefc_2dm.run_varying_regs(mode, 
                                                                   reg_fun,
                                                                   reg_conds,
                                                                   reg_kwargs,
                                                                   response_matrix,
                                                                   probe_modes, 
                                                                   probe_amp, 
                                                                   calib_modes,
                                                                   control_mask,
                                                                   num_iterations=Nitr, 
                                                                   loop_gain=0.5, 
                                                                   leakage=0.0,
                                                                   plot_current=False,
                                                                   plot_radial_contrast=False,
                                                                  )
    
    utils.save_fits(data_dir/'images'/f'hlc_iefc_2dm_images_{i+1}_{Ncalibs}_{today}.fits', ensure_np_array(images))
    utils.save_fits(data_dir/'dm-commands'/f'hlc_iefc_2dm_dm1_{i+1}_{Ncalibs}_{today}.fits', ensure_np_array(dm1_commands))
    utils.save_fits(data_dir/'dm-commands'/f'hlc_iefc_2dm_dm2_{i+1}_{Ncalibs}_{today}.fits', ensure_np_array(dm2_commands))


















