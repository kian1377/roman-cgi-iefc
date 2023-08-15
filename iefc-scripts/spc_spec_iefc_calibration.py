import numpy as np
import cupy as cp
import astropy.units as u
from astropy.io import fits
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from IPython.display import clear_output
from importlib import reload

import logging, sys
poppy_log = logging.getLogger('poppy')
poppy_log.setLevel('INFO')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

poppy_log.disabled = True

import warnings
warnings.filterwarnings("ignore")

import cgi_phasec_poppy as cgi
from cgi_phasec_poppy import misc

from wfsc import iefc_2dm as iefc
from wfsc import utils


reload(cgi)
sys = cgi.CGI(cgi_mode='spc-spec', npsf=64,
              use_fpm=True,
              use_pupil_defocus=True, 
              polaxis=0,
              use_opds=True,
#               dm1_ref=dm1_flat, dm2_ref=dm2_flat,
             )
npsf = sys.npsf
Nact = sys.Nact

# Setup the masks
reload(utils)
xfp = np.linspace(-0.5, 0.5, npsf) * npsf * sys.psf_pixelscale_lamD
xf,yf = np.meshgrid(xfp,xfp)

edge = 2
iwa = 3
owa = 9
rot = 0

# Create the mask that is used to select which region to make dark.
dh_params = {
    'inner_radius' : iwa,
    'outer_radius' : owa,
    'side':'b'
}
dh_mask = utils.create_bowtie_mask(xf, yf, dh_params)

control_params = {
    'inner_radius' : iwa-0.3,
    'outer_radius' : owa+0.7,
    'side':'b'
}
control_mask = utils.create_bowtie_mask(xf, yf, control_params)

nmask = control_mask.sum()
print(nmask)

relative_weight = 0.95
weights = dh_mask * relative_weight + (1 - relative_weight) * control_mask

misc.myimshow3(dh_mask,
               control_mask, 
               weights,
               lognorm3=True,
               pxscl1=sys.psf_pixelscale_lamD, pxscl2=sys.psf_pixelscale_lamD, pxscl3=sys.psf_pixelscale_lamD)

# Create probe and calibration modes
reload(iefc)
probe_modes = iefc.create_probe_poke_modes(Nact, 
                                           xinds=[Nact//4, Nact//4+1],
                                           yinds=[Nact//4, Nact//4], 
                                           display=False)

calibration_modes, fx, fy = iefc.create_fourier_modes(xfp, 
                                                      control_mask.reshape((npsf,npsf)), 
                                                      Nact, 
                                                      circular_mask=False)
calibration_modes[:] *= sys.dm_mask.flatten()

nmodes = calibration_modes.shape[0]
print('Calibration modes required: {:d}'.format(nmodes))

calibration_amplitude = 0.006 * sys.wavelength_c.to(u.m).value
probe_amplitude = 0.05 * sys.wavelength_c.to(u.m).value

misc.myimshow3(dh_mask.reshape(npsf,npsf),
               control_mask.reshape(npsf,npsf), 
               weights.reshape(npsf,npsf),
               lognorm3=True,
               pxscl1=sys.psf_pixelscale_lamD, pxscl2=sys.psf_pixelscale_lamD, pxscl3=sys.psf_pixelscale_lamD)

# Create the calibration cube
reload(iefc)
response_cube, calibration_cube = iefc.calibrate(sys, probe_amplitude, probe_modes, 
                                                 calibration_amplitude, calibration_modes, start_mode=0)

fname = 'spc_spec_2dm_both_3to9'
iefc_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')

misc.save_pickle(iefc_dir/'response-data'/fname, response_cube)
misc.save_pickle(iefc_dir/'calibration-data'/fname, calibration_cube)



