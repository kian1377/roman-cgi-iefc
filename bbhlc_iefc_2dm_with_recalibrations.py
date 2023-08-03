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

dm1_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm1.fits')
dm2_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm2.fits')

reload(cgi_phasec_poppy.source_flux)

wavelength_c = 575e-9*u.m

nlam = 3
bandwidth = 2.6/100
minlam = wavelength_c * (1 - bandwidth/2)
maxlam = wavelength_c * (1 + bandwidth/2)
wavelengths = np.linspace( minlam, maxlam, nlam )
print(wavelengths)
from astropy.constants import h, c, k_B, R_sun

zpup = cgi_phasec_poppy.source_flux.SOURCE(wavelengths=wavelengths,
                                            temp=40000*u.K,
                                            distance=300*u.parsec,
                                            diameter=2*14*R_sun,
                                            name='$\zeta$ Pup', 
                                            lambdas=np.linspace(10, 1000, 19801)*u.nm,
                                           )

zpup.plot_spectrum()
# zpup.plot_spectrum_ph()

source_fluxes = zpup.calc_fluxes()
print(source_fluxes)
total_flux = np.sum(source_fluxes)
print(total_flux)


reload(cgi_phasec_poppy.cgi)
reload(cgi_phasec_poppy.parallelized_cgi)

rayCGI = ray.remote(cgi_phasec_poppy.cgi.CGI) # make a ray actor class from the original CGI class  

wavelength_c = 575e-9*u.m

kwargs = {
    'cgi_mode':'hlc',
    'use_pupil_defocus':True,
    'use_opds':True,
    'polaxis':10,
}

actors = []
for i in range(nlam):
    actors.append(rayCGI.options(num_cpus=2, num_gpus=1/nlam).remote(**kwargs))
    actors[i].setattr.remote('wavelength', wavelengths[i])
    actors[i].setattr.remote('source_flux', source_fluxes[i])
    
    
reload(cgi_phasec_poppy.parallelized_cgi)
mode = cgi_phasec_poppy.parallelized_cgi.ParallelizedCGI(actors=actors, dm1_ref=dm1_flat, dm2_ref=dm2_flat)

mode.use_noise = True
mode.exp_time = 2*u.s
mode.dark_current_rate = 0.05*u.electron/u.pix/u.hour
# mode.dark_current_rate = 0.0*u.electron/u.pix/u.hour
mode.read_noise = 120*u.electron/u.pix
# mode.read_noise = 0*u.electron/u.pix
mode.gain = 1

mode.set_actor_attr('use_fpm',False)
ref_unocc_im = mode.snap()
imshow1(ref_unocc_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

max_ref = xp.max(ref_unocc_im)
print(max_ref)

mode.set_actor_attr('use_fpm',True)
mode.Imax_ref = max_ref

ref_im = mode.snap()
imshow1(ref_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)


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
probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.25, shift=(-10,8), nprobes=2, plot=True)

# choose calibration modes 
calib_amp = 5e-9
calib_modes = utils.create_all_poke_modes(mode.dm_mask)

exp_times = np.array([2, 10, 20, 30, 40, 50])*u.s

Ncalibs = 6
Nitr = 5
for i in range(Ncalibs):
    print('Calibrations {:d}/{:d}'.format(i+1, Ncalibs))
    
    # adjust exposure time to get better SNR within dark hole after every few iterations
    mode.exp_time = exp_times[i]
    mode.Imax_ref = None

    mode.set_actor_attr('use_fpm',False)
    ref_unocc_im = mode.snap()
    imshow1(ref_unocc_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

    max_ref = xp.max(ref_unocc_im)
    print(max_ref)

    mode.set_actor_attr('use_fpm',True)
    mode.Imax_ref = max_ref

    new_ref_im = mode.snap()
    imshow1(new_ref_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)
    
    
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
    

