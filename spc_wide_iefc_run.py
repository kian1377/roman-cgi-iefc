import numpy as np
import scipy

import cupy as cp
import cupyx.scipy

import astropy.units as u
from astropy.io import fits
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from IPython.display import display, clear_output
from importlib import reload
import time

import logging, sys
poppy_log = logging.getLogger('poppy')
poppy_log.setLevel('INFO')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

poppy_log.disabled = True

import warnings
warnings.filterwarnings("ignore")

import wfsc_tests as wfsc
from wfsc_tests.math_module import xp, _scipy
wfsc.math_module.update_np(cp)
wfsc.math_module.update_scipy(cupyx.scipy)

import cgi_phasec_poppy as cgi

import misc_funs as misc

iefc_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')

source_flux = 2.0208517e8 * u.ph/u.s/u.m**2 # flux of 47 UMa at 575nm with 10% bandpass

c = cgi.CGI(cgi_mode='spc-wide', npsf=150, 
              use_pupil_defocus=True, 
              use_opds=True,
              source_flux=source_flux,
            exp_time=5)

npsf = c.npsf
Nact = c.Nact

c.use_fpm = False
ref_unocc_im = c.snap()

max_ref = ref_unocc_im.get().max()
display(max_ref)

c.use_fpm = True
c.source_flux = source_flux/max_ref # divide the source flux to get nominal contrast images
ref_im = c.snap()

reload(wfsc)
reload(wfsc.utils)
reload(wfsc.imshows)
xfp = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*c.psf_pixelscale_lamD
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
roi1 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=False)

iwa = 6
owa = 20
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
#         'edge' : 0,
        'rotation':0,
        'full':True,
    }
roi2 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=False)

# wfsc.imshow2(roi1, roi2)
relative_weight = 0.2
weight_map = roi1 + relative_weight*(roi2 * ~roi1)
control_mask = weight_map>0

reload(wfsc.utils)
calib_amp = 5e-9
fourier_modes, fs = wfsc.utils.select_fourier_modes(c, control_mask*(fpx>0), fourier_sampling=0.8) 
nmodes = fourier_modes.shape[0]
print(fourier_modes.shape, fs.shape)

probe_modes = wfsc.utils.create_probe_poke_modes(Nact, 
                                                 poke_indices=[(Nact//3, Nact//3), 
                                                               (2*Nact//3, Nact//3), 
                                                               (Nact//2, 2*Nact//3)], 
                                                 plot=False)

probe_amp = 5e-8
calib_amp = 5e-9

differential_images, single_images = wfsc.iefc_2dm.take_measurement(c, 
                                                                   probe_modes, probe_amp, 
                                                                   return_all=True)
Ncalibs = 5
Nitr = 20
for i in range(Ncalibs):
    
    response_matrix, response_cube = wfsc.iefc_2dm.calibrate(c, 
                                                             control_mask.ravel(),
                                                             probe_amp, probe_modes, 
                                                             calib_amp, fourier_modes, 
                                                             return_all=True)

    misc.save_fits(iefc_dir/'response-data'/'spcwide_iefc_2dm_response_matrix_20230530_{:d}.fits'.format(i), 
                   wfsc.utils.ensure_np_array(response_matrix))
    misc.save_fits(iefc_dir/'response-data'/'spcwide_iefc_2dm_response_cube_20230530_{:d}.fits'.format(i), 
                   wfsc.utils.ensure_np_array(response_cube))
    
    # Run iEFC
    reg_cond = 1e-2

    Wmatrix = np.diag(np.concatenate((weight_map[control_mask], weight_map[control_mask], weight_map[control_mask])))
    cm_wls = wfsc.utils.WeightedLeastSquares(response_matrix, Wmatrix, rcond=reg_conds[i][0])
    
    images, dm1_commands, dm2_commands = wfsc.iefc_2dm.run(c, 
                                              cm_wls,
                                              probe_modes, 
                                              probe_amp, 
                                              fourier_modes,
                                              control_mask, 
                                              num_iterations=Nitr, 
                                              loop_gain=0.25, 
                                              leakage=0.0,
                                              plot_all=True,
                                             )











