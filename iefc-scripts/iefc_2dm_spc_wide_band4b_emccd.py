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

from emccd_detect import emccd_detect
meta_path = Path('/home/kianmilani/Projects/emccd_detect/emccd_detect/emccd_detect/util/metadata.yaml')

dm1_flat = fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'spc_wide_band4_flattened_dm1.fits')
dm2_flat = fits.getdata(cgi_phasec_poppy.data_dir/'dm-acts'/'flatmaps'/'spc_wide_band4_flattened_dm2.fits')

data_dir = iefc_2dm.iefc_data_dir
response_dir = data_dir/'response-data'

reload(cgi_phasec_poppy.source_flux)

wavelength_c = 825e-9*u.m

nwaves = 3
bandwidth = 2.9/100
minwave = wavelength_c * (1 - bandwidth/2)
maxwave = wavelength_c * (1 + bandwidth/2)
wavelengths = np.linspace( minwave, maxwave, nwaves )

minlam = 400*u.nm
maxlam = 1000*u.nm
nlam = int((maxlam-minlam).value*20) + 1
lambdas = np.linspace(minlam, maxlam, nlam)

from astropy.constants import h, c, k_B, R_sun

zpup = cgi_phasec_poppy.source_flux.SOURCE(wavelengths=wavelengths,
                                            temp=40000*u.K,
                                            distance=300*u.parsec,
                                            diameter=2*14*R_sun,
                                            name='$\zeta$ Puppis', 
                                            lambdas=lambdas,
                                           )

zpup.plot_spectrum_ph()
source_fluxes = zpup.calc_fluxes()
total_flux = np.sum(source_fluxes)

for i,flux in enumerate(source_fluxes):
    print(f'\tFlux for wavelength {wavelengths[i]:.3e}: {flux:.3e}')
print(f'Total flux: {total_flux:.3e}')


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
    actors[i].setattr.remote('source_flux', source_fluxes[i])


em_gain = 200
full_well_image=60000.  # e-
full_well_serial=100000.  # e-
#status=1,
dark_current=0.0028  # e-/pix/s
dark_current=1.5/3600  # e-/pix/s
cic=0.02  # e-/pix/frame
read_noise=120.  # e-/pix/frame
bias=250.  # e-
qe=0.5
cr_rate=0.  # hits/cm^2/s
pixel_pitch=13e-6  # m
eperdn=1 #7.,
nbits=32
numel_gain_register=604

emccd = emccd_detect.EMCCDDetect(em_gain=em_gain,
                                    full_well_image=full_well_image,  # e-
                                    full_well_serial=full_well_serial,  # e-
                                    #status=status,
                                    dark_current=dark_current,  # e-/pix/s
                                    cic=cic,  # e-/pix/frame
                                    read_noise=read_noise,  # e-/pix/frame
                                    bias=bias,  # e-
                                    qe=qe,
                                    cr_rate=cr_rate,  # hits/cm^2/s
                                    pixel_pitch=pixel_pitch,  # m
                                    eperdn=eperdn,
                                    nbits=nbits,
                                    numel_gain_register=numel_gain_register,
                                    meta_path=meta_path
                                    )

reload(cgi_phasec_poppy.parallelized_cgi)
mode = cgi_phasec_poppy.parallelized_cgi.ParallelizedCGI(actors=actors, dm1_ref=dm1_flat, dm2_ref=dm2_flat)

mode.use_photon_noise = True
mode.EMCCD = emccd

unocc_exp_time = 0.00001
unocc_em_gain = 200

mode.set_actor_attr('use_fpm',False)
mode.Nframes = 5
mode.exp_time = unocc_exp_time
mode.EMCCD.em_gain = unocc_em_gain

ref_unocc_im = mode.snap()
imshow1(ref_unocc_im, lognorm=True)

mode.set_actor_attr('use_fpm',True)
mode.Nframes = 4
mode.exp_time = 0.25
mode.EMCCD.em_gain = 750

occ_im = mode.snap(quiet=False)
imshow1(occ_im, lognorm=True)

mode.normalize = True
mode.Imax_ref = xp.max(ref_unocc_im)
mode.exp_time_ref = unocc_exp_time
mode.em_gain_ref = unocc_em_gain

ref_im = mode.snap(quiet=False)
imshow1(ref_im, 'Reference/Initial State', lognorm=True, save_fig='test_ref_im.png')

control_mask = utils.create_annular_focal_plane_mask(mode, inner_radius=5.4, outer_radius=20.6, edge=None, plot=True)
mean_ni = xp.mean(ref_im[control_mask])
print(mean_ni)


probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.2,
                                          shift=[(-12,7), (12,7),(0,-14), (0,0)], nprobes=3,
                                           use_weighting=True,)

calib_modes = utils.create_hadamard_modes(mode.dm_mask, ndms=2)
Nmodes = calib_modes.shape[0]
print(calib_modes.shape)


mode.EMCCD.em_gain = 400
mode.exp_time = 0.05
mode.Nframes = 10

calib_amp = 2.5e-9
probe_amp = 25e-9

mode.normalize = False
i = 2
mode.add_dm1(calib_amp*calib_modes[i,:mode.Nact**2].reshape(mode.Nact,mode.Nact))
mode.add_dm1(probe_amp*probe_modes[0])
im = mode.snap()
mode.add_dm1(-probe_amp*probe_modes[0])
mode.add_dm1(-calib_amp*calib_modes[i,:mode.Nact**2].reshape(mode.Nact,mode.Nact))
imshow1(im, lognorm=True, save_fig='test_saturation.png')

mode.normalize = True
mode.add_dm1(calib_amp*calib_modes[i,:mode.Nact**2].reshape(mode.Nact,mode.Nact))
differential_images, single_images = iefc_2dm.take_measurement(mode, probe_modes, probe_amp, return_all=True, plot=True)
ims = 2*calib_amp*differential_images.reshape(probe_modes.shape[0], mode.npsf, mode.npsf)
mode.add_dm1(-calib_amp*calib_modes[i,:mode.Nact**2].reshape(mode.Nact,mode.Nact))
imshow3(ims[0], ims[1], ims[2], save_fig='test_diff_ims.png')

mode.normalize = True
response_matrix, response_cube = iefc_2dm.calibrate(mode, 
                                                    control_mask,
                                                    probe_amp, probe_modes, 
                                                     calib_amp, calib_modes, 
                                                     return_all=True, 
#                                                     plot_responses=False,
                                                   )
utils.save_fits(response_dir/f'spc_wide_band4b_emccd_response_matrix_{today}.fits', 
                response_matrix,
                header={'em_gain':mode.EMCCD.em_gain,
                        'Nframes':mode.Nframes,
                        'exp_time':mode.exp_time})

# iefc_2dm_spc_wide_band4b_emccd.py

