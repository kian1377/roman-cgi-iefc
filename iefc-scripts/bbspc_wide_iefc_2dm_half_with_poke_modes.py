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

import ray
if not ray.is_initialized():
    ray.init(log_to_driver=False)

from math_module import xp, ensure_np_array
import iefc_2dm 
import utils
from imshows import *

data_dir = iefc_2dm.iefc_data_dir
response_dir = data_dir/'response-data'

# Make the spectrum and create the actors and broadband mode
wavelength_c = 825e-9*u.m

nlam = 5
bandwidth = 0.1
minlam = wavelength_c * (1 - bandwidth/2)
maxlam = wavelength_c * (1 + bandwidth/2)
wavelengths = np.linspace( minlam, maxlam, nlam )

from astropy.constants import h, c, k_B, R_sun

uma47 = cgi_phasec_poppy.source_flux.SOURCE(wavelengths=wavelengths,
                                            temp=5887*u.K,
                                            distance=14.06*u.parsec,
                                            diameter=2*1.172*R_sun)

uma47.plot_spectrum()

source_fluxes = uma47.calc_fluxes()
print(source_fluxes)


rayCGI = ray.remote(cgi_phasec_poppy.cgi.CGI) # make a ray actor class from the original CGI class  

kwargs = {
    'cgi_mode':'spc-wide',
    'npsf':148,
    'use_pupil_defocus':True,
    'use_opds':True,
    'polaxis':10,
}

actors = []
for i in range(nlam):
    actors.append(rayCGI.options(num_cpus=2, num_gpus=1/nlam).remote(**kwargs))
    actors[i].setattr.remote('wavelength', wavelengths[i])
    actors[i].setattr.remote('source_flux', source_fluxes[i])
    
mode = cgi_phasec_poppy.parallelized_cgi.ParallelizedCGI(actors=actors)

mode.set_actor_attr('use_fpm',False)
ref_unocc_im = mode.snap()
imshow1(ref_unocc_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True, 
        display_fig=False, 
        save_fig='figures/bbspc_wfov_unocculted.png')

max_ref = ref_unocc_im.get().max()
display(max_ref)

mode.set_actor_attr('use_fpm',True)
mode.Iref = max_ref
ref_im = mode.snap()
imshow1(ref_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True,
        display_fig=False, 
        save_fig='figures/bbspc_wfov_occulted_initial_state.png')
    
# create the weight map
xfp = (xp.linspace(-mode.npsf/2, mode.npsf/2-1, mode.npsf) + 1/2)*mode.psf_pixelscale_lamD
fpx,fpy = xp.meshgrid(xfp,xfp)
  
iwa = 6
owa = 20
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
#         'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi1 = utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 5.4
owa = 20.6
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
#         'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi2 = utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 6
owa = 11
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
#         'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi3 = utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

relative_weight_1 = 0.9
relative_weight_2 = 0.2
weight_map = roi3 + relative_weight_1*(roi1*~roi3) + relative_weight_2*(roi2*~roi1*~roi3)
control_mask = weight_map>0
imshow1(weight_map, display_fig=False, save_fig='bbspc_wfov_weight_map.png')
utils.save_fits(response_dir/f'bbspc_wfov_iefc_2dm_weight_map_bw0.1_{today}.fits', ensure_np_array(weight_map))

# Create the poke calibration modes
calib_amp = 2.5e-9

calib_modes = xp.zeros((mode.Nacts, mode.Nact, mode.Nact))
count=0
for i in range(mode.Nact):
    for j in range(mode.Nact):
        if mode.dm_mask[i,j]:
            calib_modes[count, i,j] = 1
            count+=1
            
calib_modes = calib_modes[:,:].reshape(mode.Nacts, mode.Nact**2)

# create the probe modes, also using pokes
probe_amp = 2.5e-8

# poke_probes = wfsc.utils.create_probe_poke_modes(Nact, [(Nact//2, Nact//5), (Nact//2, Nact//5+2)], plot=True)
probe_modes = utils.create_probe_poke_modes(mode.Nact,  
                                            [(mode.Nact//5+2, mode.Nact//3), (mode.Nact//5+1, mode.Nact//3)])

# Calculate response matrix
response_matrix, response_cube = iefc_2dm.calibrate(mode, 
                                                         control_mask,
                                                         probe_amp, probe_modes, 
                                                         calib_amp, ensure_np_array(calib_modes), 
                                                         return_all=True)


utils.save_fits(response_dir/f'bbspc_wfov_iefc_2dm_poke_response_matrix_bw0.1_{today}.fits', 
                ensure_np_array(response_matrix))
utils.save_fits(response_dir/f'bbspc_wfov_iefc_2dm_poke_response_cube_bw0.1_{today}.fits',
                ensure_np_array(response_cube))




