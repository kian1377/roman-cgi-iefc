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

''' COMPUTE SOURCE FLUX FROM BLACKBODY '''

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

''' INITIALIZE RAY ACTORS FOR MONOCHROMATIC PROPAGATION '''

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

''' INITIALIZE EMCCD PARAMETERS AND CREATE EMCCD OBJECT '''

em_gain = 200
full_well_image = 60000.  # e-
full_well_serial = 100000.  # e-
#status=1,
dark_current = 8e-4  # e-/pix/s
cic = 0.01  # e-/pix/frame
read_noise = 120.  # e-/pix/frame
bias = 500.  # e-
qe = 0.5
cr_rate = 0.  # hits/cm^2/s
pixel_pitch = 13e-6  # m
eperdn = 1 #7.,
nbits = 16
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

''' GENERATE PARALLELIZED CGI MODEL '''

reload(cgi_phasec_poppy.parallelized_cgi)
mode = cgi_phasec_poppy.parallelized_cgi.ParallelizedCGI(actors=actors, dm1_ref=dm1_flat, dm2_ref=dm2_flat)

mode.EMCCD = emccd

unocc_em_gain = 500

mode.set_actor_attr('use_fpm',False)
mode.Nframes = 10
mode.EMCCD.em_gain = unocc_em_gain

mode.exp_times_list = [0.00001, 0.00005, 0.0001]

raw_im = mode.snap_many()

mode.normalize = True
mode.Imax_ref = xp.max(raw_im.max())
mode.em_gain_ref = unocc_em_gain

ref_unocc_im = mode.snap_many()
imshow3(raw_im, ref_unocc_im/mode.norm_factor, ref_unocc_im, 
        '', '', f'{xp.max(ref_unocc_im):.2f}', lognorm=True,
        save_fig='test_normalization_image.png')

mode.EMCCD.em_gain = 500

mode.set_actor_attr('use_fpm',True)
mode.exp_times_list = [0.25,2.5,5]
mode.Nframes_list = [4, 2, 1]
ref_im = mode.snap_many(quiet=False,)

control_mask = utils.create_annular_focal_plane_mask(mode, inner_radius=5.4, outer_radius=20.6, edge=None)
mean_ni = xp.mean(ref_im[control_mask])
imshow3(ref_im/mode.norm_factor, ref_im, ref_im*control_mask, 
        f'Reference/Initial State: {xp.max(ref_im/mode.norm_factor):.0f}', 
        'Normalized Reference Image',
        f'Mean NI: {mean_ni:.2e}',
        lognorm=True,
        save_fig='test_reference_image.png')


''' CREATE THE PROBE AND CALIBRATION MODES '''

probe_modes = utils.create_fourier_probes(mode, control_mask, fourier_sampling=0.2,
                                          shift=[(-12,7), (12,7),(0,-14), (0,0)], nprobes=3,
                                           use_weighting=True)

fourier_mask = utils.create_annular_focal_plane_mask(mode, inner_radius=0.5, outer_radius=23, edge=0, plot=True)
calib_modes, fs = utils.create_fourier_modes(mode, fourier_mask, fourier_sampling=1, ndms=2, return_fs=True)
Nfourier = calib_modes.shape[0]//2
print(calib_modes.shape)

''' CREATE THE SCALING FACTOR FOR THE CALIBRATION MODES '''

scale_factors = []
for i in range(fs.shape[0]):
    mode_extent = xp.sqrt(fs[i][0]**2 + fs[i][1]**2)
    if mode_extent<2:
        factor = 20
    elif mode_extent>2 and mode_extent<3:
        factor = 5
    elif mode_extent>3 and mode_extent<4:
        factor = 3.75
    elif mode_extent>4 and mode_extent<6:
        factor = 2
    elif mode_extent>6 and mode_extent<9:
        factor = 1
    elif mode_extent>9 and mode_extent<10:
        factor = 1
    elif mode_extent>10 and mode_extent<15:
        factor = 1
    elif mode_extent>15 and mode_extent<18:
        factor = 1
    elif mode_extent>18 and mode_extent<20:
        factor = 1.666
    elif mode_extent>20 and mode_extent<22:
        factor = 2.5
    elif mode_extent>22 and mode_extent<23:
        factor = 5
    elif mode_extent>23:
        factor = 10
    scale_factors.append(factor)
scale_factors = np.array(scale_factors)

scale_factors = np.concatenate([scale_factors,scale_factors, scale_factors, scale_factors])
print(scale_factors.shape)

''' GENERATE RESPONSE MATRIX '''

reload(iefc_2dm)

mode.reset_dms()
mode.EMCCD.em_gain = 250
mode.Nframes = 1
mode.exp_times_list = np.array([0.01, 0.05, 0.25, 0.5])/2
mode.Nframes_list = np.array([10, 3, 1, 1])

total_exp_time = np.sum(mode.exp_times_list*mode.Nframes_list)
print(f'Total exposure time: {total_exp_time:.2f}s')

calib_amp = 2.5e-9
probe_amp = 20e-9

response_matrix, response_cube, calib_amps = iefc_2dm.calibrate(mode, 
                                                    control_mask,
                                                    probe_amp, probe_modes, 
                                                     calib_amp, calib_modes, 
                                                     scale_factors=scale_factors, 
                                                     return_all=True, 
#                                                     plot_responses=False,
                                                   )

utils.save_fits(response_dir/f'spc_wide_band4b_emccd_response_matrix_{today}.fits', 
                response_matrix,
                header={'em_gain':mode.EMCCD.em_gain,})

# iefc_2dm_spc_wide_band4b_emccd_fourier.py

