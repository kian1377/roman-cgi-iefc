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

# import wfsc_tests as wfsc
# wfsc.math_module.update_np(cp)
# wfsc.math_module.update_scipy(cupyx.scipy)
# from wfsc_tests.math_module import xp, _scipy, ensure_np_array

import cgi_phasec_poppy

from math_module import xp
import iefc_2dm 
import utils
from imshows import *

data_dir = iefc_2dm.iefc_data_dir
response_dir = data_dir/'response-data'

dm1_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm1.fits')
dm2_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm2.fits')

imshow2(dm1_flat/2, dm2_flat/2, 'DM1 Initial State', 'DM2 Initial State', cmap1='viridis', cmap2='viridis')

# INITIALIZE THE MODE
mode = cgi_phasec_poppy.cgi.CGI(cgi_mode='hlc', npsf=60,
                              use_pupil_defocus=True, 
                              use_opds=True)

mode.set_dm1(dm1_flat)
mode.set_dm2(dm2_flat)

mode.use_fpm = False
ref_unocc_im = mode.snap()
imshow1(ref_unocc_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

mode.Imax_ref = ref_unocc_im.get().max()
mode.use_fpm = True

ref_im = mode.snap()
imshow1(ref_im, 'HLC Initial Coronagraphic Image\nfor iEFC',
        pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True, vmin=1e-11)


# CREATE THE DARK_HOLE MASK
xfp = (xp.linspace(-mode.npsf/2, mode.npsf/2-1, mode.npsf) + 1/2)*mode.psf_pixelscale_lamD
fpx,fpy = xp.meshgrid(xfp,xfp)
  
iwa = 3
owa = 9
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi1 = utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 2.8
owa = 9.7
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi2 = utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 3.2
owa = 6
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi3 = utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

relative_weight_1 = 0.9
relative_weight_2 = 0.2
weight_map = roi3 + relative_weight_1*(roi1*~roi3) + relative_weight_2*(roi2*~roi1*~roi3)
control_mask = weight_map>0
imshow1(weight_map)
misc.save_fits(iefc_dir/'response-data'/f'hlc_iefc_2dm_weight_map_{date}.fits', ensure_np_array(weight_map))

# MAKE THE PROBE MODES
probe_amp = 2.5e-8
fourier_modes, fs = utils.select_fourier_modes(mode, control_mask*(fpx>0), fourier_sampling=0.5) 
probe_modes = utils.create_fourier_probes(fourier_modes, shift_cos=(10,10), shift_sin=(-10,-10), plot=True)

# MAKE THE CALIBRATION MODES
calib_amp = 2.5e-9

calib_modes = xp.zeros((mode.Nacts, mode.Nact, mode.Nact))
count=0
for i in range(mode.Nact):
    for j in range(mode.Nact):
        if mode.dm_mask[i,j]:
            calib_modes[count, i,j] = 1
            count+=1
            
calib_modes = calib_modes[:,:].reshape(mode.Nacts, mode.Nact**2)

# CALCULATE THE RESPONSE MATRIX
response_matrix, response_cube = iefc_2dm.calibrate(mode, 
                                                    control_mask,
                                                    probe_amp, probe_modes, 
                                                    calib_amp, utils.ensure_np_array(calib_modes), 
                                                    return_all=True)

misc.save_fits(iefc_dir/'response-data'/f'hlc_iefc_2dm_poke_response_matrix_{today}.fits', ensure_np_array(response_matrix))
misc.save_fits(iefc_dir/'response-data'/f'hlc_iefc_2dm_poke_response_cube_{today}.fits', ensure_np_array(response_cube))



