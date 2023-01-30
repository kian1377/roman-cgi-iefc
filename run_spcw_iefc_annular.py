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

sysi = cgi.CGI(cgi_mode='spc-wide', npsf=150,
              use_fpm=True,
              use_pupil_defocus=True, 
              polaxis=0,
              use_opds=True,
             )
sysi.show_dms()

npsf = sysi.npsf
Nact = sysi.Nact

ref_psf = sysi.snap()

xfp = np.linspace(-0.5, 0.5, npsf) * npsf * sysi.psf_pixelscale_lamD
fpx,fpy = np.meshgrid(xfp,xfp)
fpr = np.sqrt(fpx**2 + fpy**2)

# Create the mask that is used to select which region to make dark.
iwa = 5.8
owa = 20.2
regions = [iwa, 6, 12, 20, owa]
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

calib_modes, fs = utils.select_fourier_modes(sysi, control_mask*(fpx>0), fourier_sampling=0.75) 
nmodes = calib_modes.shape[0]

probe_modes = utils.create_probe_poke_modes(Nact, xinds=[Nact//4, Nact//4+1], yinds=[Nact//4, Nact//4], display=True)

probe_amp = 3e-8
calib_amp = 5e-9

M = 3
N = 20

iefc_dir = Path('C:/Users/Kian/Documents/data-files/roman-cgi-iefc-data')

for i in range(M):
    response_cube, calibration_cube = iefc.calibrate(sysi, 
                                                     probe_amp, probe_modes, 
                                                     calib_amp, calib_modes, start_mode=0)
    fname = 'spc_wide_2dm_annular_5.8-20.2_{:d}.pkl'.format(i)
    misc.save_pickle(iefc_dir/'response-data'/fname, response_cube)
    
    reg_fun = iefc.construct_control_matrix
    reg_conds = [[0],
                 [(1e-2, 1e-2)]]
    
    images, dm1_commands, dm2_commands = iefc.run(sysi, 
                                                  reg_fun,reg_conds,
                                                  response_cube, 
                                                  probe_modes, 
                                                  probe_amp, 
                                                  calib_modes, 
                                                  weight_map, 
                                                  num_iterations=50, 
                                                  loop_gain=0.1, leakage=0.0,
                                                  display_all=True,
                                                 )
    
    misc.save_pickle(iefc_dir/'response-data'/'images_{:d}.pkl'.format(i), images)
    misc.save_pickle(iefc_dir/'response-data'/'dm1_commands_{:d}.pkl'.format(i), dm1_commands)
    misc.save_pickle(iefc_dir/'response-data'/'dm2_commands_{:d}.pkl'.format(i), dm2_commands)



    
    