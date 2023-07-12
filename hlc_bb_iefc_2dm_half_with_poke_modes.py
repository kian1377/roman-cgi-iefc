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

from importlib import reload
import time

import logging, sys
poppy_log = logging.getLogger('poppy')
poppy_log.setLevel('DEBUG')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

poppy_log.disabled = True

import warnings
warnings.filterwarnings("ignore")

import ray
if not ray.is_initialized():
    ray.init(log_to_driver=False)

import wfsc_tests as wfsc
wfsc.math_module.update_np(cp)
wfsc.math_module.update_scipy(cupyx.scipy)

from wfsc_tests.math_module import xp, _scipy, ensure_np_array

import cgi_phasec_poppy

import misc_funs as misc

from datetime import datetime
date = int(datetime.today().strftime('%Y%m%d'))

# iefc_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')
iefc_dir = Path('/home/kianmilani/Projects/roman-cgi-iefc-data')

dm1_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm1.fits')
dm2_flat = 2*fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'hlc_flattened_dm2.fits')


# Initialize source fluxes
wavelength_c = 575e-9*u.m

nlam = 5
bandwidth = 0.10
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


# create ray actors
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
    actors.append(rayCGI.options(num_gpus=1/nlam).remote(**kwargs))
    actors[i].setattr.remote('wavelength', wavelengths[i])
    actors[i].setattr.remote('source_flux', source_fluxes[i])
    
# Initialize the mode with the actors
mode = cgi_phasec_poppy.parallelized_cgi.ParallelizedCGI(actors=actors)

mode.set_dm1(dm1_flat)
mode.set_dm2(dm2_flat)

mode.set_actor_attr('use_fpm',False)
ref_unocc_im = mode.snap()
wfsc.imshow1(ref_unocc_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

max_ref = ref_unocc_im.get().max()
display(max_ref)

mode.set_actor_attr('use_fpm',True)
mode.Iref = max_ref
ref_im = mode.snap()
wfsc.imshow1(ref_im, pxscl=mode.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)


# create mask
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
roi1 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 2.8
owa = 9.7
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi2 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 3.2
owa = 6
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 2,
        'rotation':0,
        'full':True,
    }
roi3 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

relative_weight_1 = 0.9
relative_weight_2 = 0.2
weight_map = roi3 + relative_weight_1*(roi1*~roi3) + relative_weight_2*(roi2*~roi1*~roi3)
control_mask = weight_map>0
wfsc.imshow1(weight_map)

misc.save_fits(iefc_dir/'response-data'/f'hlcbb_iefc_2dm_weight_map_{date}.fits', ensure_np_array(weight_map))

# Make probe and calibration modes
probe_amp = 2.5e-8

fourier_modes, fs = wfsc.utils.select_fourier_modes(mode, control_mask*(fpx>0), fourier_sampling=0.5) 
probe_modes = wfsc.utils.create_fourier_probes(fourier_modes, shift_cos=(10,10), shift_sin=(-10,-10), plot=True)

calib_amp = 2.5e-9
Nacts = int(mode.dm_mask.sum())
calib_modes = xp.zeros((Nacts, mode.Nact, mode.Nact))
count=0
for i in range(mode.Nact):
    for j in range(mode.Nact):
        if mode.dm_mask[i,j]:
            calib_modes[count, i,j] = 1
            count+=1
calib_modes = calib_modes[:,:].reshape(Nacts, mode.Nact**2)

Ncalibs = 5
Nitr = 20
for i in range(Ncalibs):
    
    # calculate response matrix
    response_matrix, response_cube = wfsc.iefc_2dm.calibrate(mode, 
                                                         control_mask.ravel(),
                                                         probe_amp, probe_modes, 
                                                         calib_amp, ensure_np_array(calib_modes), 
                                                         return_all=True)

    misc.save_fits(iefc_dir/'response-data'/f'hlcbb_iefc_2dm_poke_response_matrix_{date}_calib{i+1}.fits', ensure_np_array(response_matrix))
    misc.save_fits(iefc_dir/'response-data'/f'hlcbb_iefc_2dm_poke_response_cube_{date}_calib{i+1}.fits', ensure_np_array(response_cube))
    
    reg_cond = 1e-2
    cm_wls = wfsc.utils.WeightedLeastSquares(response_matrix, weight_map, nprobes=len(probe_modes), rcond=reg_cond)

    images, dm1_commands, dm2_commands = wfsc.iefc_2dm.run(mode, 
                                              cm_wls,
                                              probe_modes, 
                                              probe_amp, 
                                              ensure_np_array(calib_modes),
                                              control_mask, 
                                              num_iterations=Nitr, 
                                              loop_gain=0.5, 
                                              leakage=0.0,
                                              plot_all=True,
                                             )
    
    
    misc.save_fits(iefc_dir/'images'/f'hlcbb_iefc_2dm_poke_images_{date}_calib{i+1}.fits', ensure_np_array(images))
    misc.save_fits(iefc_dir/'dm-commands'/f'hlcbb_iefc_2dm_poke_dm1_commands_{date}_calib{i+1}.fits', ensure_np_array(dm1_commands))
    misc.save_fits(iefc_dir/'dm-commands'/f'hlcbb_iefc_2dm_poke_dm2_commands_{date}_calib{i+1}.fits', ensure_np_array(dm2_commands))
    
    response_matrix = 0
    response_cube = 0

