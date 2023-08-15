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
from datetime import datetime
date = int(datetime.today().strftime('%Y%m%d'))

import logging, sys
poppy_log = logging.getLogger('poppy')
poppy_log.setLevel('DEBUG')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

poppy_log.disabled = True

import warnings
warnings.filterwarnings("ignore")

import misc_funs as misc

import wfsc_tests as wfsc
from wfsc_tests.imshows import *
from wfsc_tests.math_module import xp, _scipy, ensure_np_array
wfsc.math_module.update_np(cp)
wfsc.math_module.update_scipy(cupyx.scipy)

import cgi_phasec_poppy as cgi

# iefc_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')
iefc_dir = Path('/home/kianmilani/Projects/roman-cgi-iefc-data')

source_flux = 2.0208517e8 * u.ph/u.s/u.m**2 # flux of 47 UMa at 575nm with 10% bandpass

c = cgi.CGI(cgi_mode='spc-spec', 
              use_pupil_defocus=True, 
              use_opds=True,
              source_flux=source_flux,
            exp_time=5)

npsf = c.npsf
Nact = c.Nact

c.use_fpm = False
ref_unocc_im = c.snap()
wfsc.imshow1(ref_unocc_im, pxscl=c.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

max_ref = ref_unocc_im.get().max()
display(max_ref)

c.use_fpm = True
c.source_flux = source_flux/max_ref # divide the source flux to get nominal contrast images
ref_im = c.snap()
wfsc.imshow1(ref_im, pxscl=c.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

xfp = (xp.arange(-npsf//2, npsf//2) +1/2) * c.psf_pixelscale_lamD
xf,yf = xp.meshgrid(xfp,xfp)

edge = 2
iwa = 3
owa = 9
rot = 0

# Create the mask that is used to select which region to make dark.
roi1_params = {
    'inner_radius' : iwa,
    'outer_radius' : owa,
    'angle':65,
#     'rotation':45,
}
roi1= wfsc.utils.create_bowtie_focal_plane_mask(xf, yf, roi1_params, plot=True)

# Create the mask that is used to select which region to make dark.
roi2_params = {
    'inner_radius' : iwa-1,
    'outer_radius' : owa+0.5,
    'angle':70,
}
roi2 = wfsc.utils.create_bowtie_focal_plane_mask(xf, yf, roi2_params, plot=True)

relative_weight = 0.2
weight_map = roi1 + relative_weight*(roi2*~roi1)
control_mask = weight_map>0
imshow2(weight_map, control_mask*ref_im, lognorm2=True)

misc.save_fits(iefc_dir/'response-data'/f'spc_spec_iefc_2dm_weight_map_{date}.fits', ensure_np_array(weight_map))

probe_amp = 2.5e-8
fourier_modes, fs = wfsc.utils.select_fourier_modes(c, control_mask*(xf>0), fourier_sampling=1) 
probe_modes = wfsc.utils.create_fourier_probes(fourier_modes, plot=True)

Nacts = int(c.dm_mask.sum())

calib_amp = 2.5e-9

calib_modes = xp.zeros((Nacts, c.Nact, c.Nact))
count=0
for i in range(c.Nact):
    for j in range(c.Nact):
        if c.dm_mask[i,j]:
            calib_modes[count, i,j] = 1
            count+=1
            
calib_modes = calib_modes[:,:].reshape(Nacts, c.Nact**2)

response_matrix, response_cube = wfsc.iefc_2dm.calibrate(c, 
                                                         control_mask.ravel(),
                                                         probe_amp, probe_modes, 
                                                         calib_amp, ensure_np_array(calib_modes), 
                                                         return_all=True)

misc.save_fits(iefc_dir/'response-data'/f'spc_spec_iefc_2dm_poke_response_matrix_{date}.fits', ensure_np_array(response_matrix))
misc.save_fits(iefc_dir/'response-data'/f'spc_spec_iefc_2dm_poke_response_cube_{date}.fits', ensure_np_array(response_cube))



