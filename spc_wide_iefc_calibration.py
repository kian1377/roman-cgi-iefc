import numpy as np
import cupy as cp
import astropy.units as u
from astropy.io import fits
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from IPython.display import clear_output
from importlib import reload
import time

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

dm_dir = cgi.data_dir/'dm-acts'

sysi = cgi.CGI(cgi_mode='spc-wide', npsf=150,
              use_fpm=True,
              use_pupil_defocus=False, 
              polaxis=0,
              use_opds=True,
             )
sysi.show_dms()

npsf = sysi.npsf
Nact = sysi.Nact

xfp = (np.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*sysi.psf_pixelscale_lamD
fpx,fpy = np.meshgrid(xfp,xfp)
fpr = np.sqrt(fpx**2 + fpy**2)
    
iwa = 5.5
owa = 20.5
regions = [iwa, 6, 10, 20, owa]
weights = [0.1, 0.8, 1, 0.1]
weight_map = np.zeros((sysi.npsf,sysi.npsf), dtype=np.float64)
for i in range(len(weights)):
    roi_params = {
        'inner_radius' : regions[i],
        'outer_radius' : regions[i+1],
        'edge_position' : 0,
        'rotation':0,
        'full':True,
    }
    roi = utils.create_annular_focal_plane_mask(fpx, fpy, roi_params)
    weight_map += roi*weights[i]

control_mask = weight_map>0
misc.myimshow2(weight_map, control_mask)

probe_amp = 3e-8
calib_amp = 5e-9

fourier_modes, fs = utils.select_fourier_modes(sysi, control_mask*(fpx>0), fourier_sampling=0.85) 
nf = fourier_modes.shape[0]
print(fourier_modes.shape)

cos_modes = fourier_modes[:nf//2]
sin_modes = fourier_modes[nf//2:]

# had_modes = utils.get_hadamard_modes(sysi.dm_mask)[:1024]
# nh = had_modes.shape[0]
# print(had_modes.shape)
    
probe_modes = utils.create_probe_poke_modes(Nact, xinds=[Nact//4, Nact//4+1], yinds=[Nact//4, Nact//4], display=True)
    
sysi.reset_dms()
response_cube, calibration_cube = iefc.calibrate(sysi, 
                                                 probe_amp, probe_modes, 
                                                 calib_amp, 
#                                                  fourier_modes, had_modes,
                                                 cos_modes, sin_modes)

fname = 'spc-wide_2dm_annular_cos_sin.pkl'
iefc_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')

misc.save_pickle(iefc_dir/'response-data'/fname, response_cube)
misc.save_pickle(iefc_dir/'calibration-data'/fname, calibration_cube)


