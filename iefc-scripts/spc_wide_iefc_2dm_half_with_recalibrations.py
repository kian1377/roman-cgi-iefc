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

import wfsc_tests as wfsc
wfsc.math_module.update_np(cp)
wfsc.math_module.update_scipy(cupyx.scipy)

from wfsc_tests.math_module import xp, _scipy

import cgi_phasec_poppy as cgi
reload(cgi)

import misc_funs as misc

from matplotlib.colors import ListedColormap
#Blue to Red Color scale for S1 and S2
colmap = np.zeros((255,3));
# Red
colmap[126:183,0]= np.linspace(0,1,57);
colmap[183:255,0]= 1;
# Green
colmap[0:96,1] = np.linspace(1,0,96);
colmap[158:255,1]= np.linspace(0,1,97);
# Blue
colmap[0:71,2] = 1;
colmap[71:128,2]= np.linspace(1,0,57);
colmap2 = colmap[128:,:]
colmap = ListedColormap(colmap)

# iefc_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')
iefc_dir = Path('/home/kianmilani/Projects/roman-cgi-iefc-data')
from datetime import datetime
date = int(datetime.today().strftime('%Y%m%d'))

# Initialize the system and normalize
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
wfsc.imshow1(ref_unocc_im, pxscl=c.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

max_ref = ref_unocc_im.get().max()
display(max_ref)

c.use_fpm = True
c.source_flux = source_flux/max_ref # divide the source flux to get nominal contrast images
ref_im = c.snap()
wfsc.imshow1(ref_im, pxscl=c.psf_pixelscale_lamD, xlabel='$\lambda/D$', lognorm=True)

# Create the focal plane pixel mask and weight map
xfp = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*c.psf_pixelscale_lamD
fpx,fpy = xp.meshgrid(xfp,xfp)
  
iwa = 6
owa = 20
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 3,
        'rotation':0,
        'full':True,
    }
roi1 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 5.4
owa = 20.6
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 3,
        'rotation':0,
        'full':True,
    }
roi2 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

iwa = 6
owa = 11
roi_params = {
        'inner_radius' : iwa,
        'outer_radius' : owa,
        'edge' : 3,
        'rotation':0,
        'full':True,
    }
roi3 = wfsc.utils.create_annular_focal_plane_mask(fpx, fpy, roi_params, plot=True)

relative_weight_1 = 0.9
relative_weight_2 = 0.2
weight_map = roi3 + relative_weight_1*(roi1*~roi3) + relative_weight_2*(roi2*~roi1*~roi3)
control_mask = weight_map>0
wfsc.imshow1(weight_map)

misc.save_fits(iefc_dir/'response-data'/'spc_wide_iefc_2dm_weight_map_{:d}.fits'.format(date), 
               wfsc.utils.ensure_np_array(weight_map))

# Create the fourier modes and the probe modes
calib_amp = 5e-9
fourier_modes, fs = wfsc.utils.select_fourier_modes(c, control_mask*(fpx>0), fourier_sampling=1) 
nmodes = fourier_modes.shape[0]
nf = nmodes//2
print(fourier_modes.shape, fs.shape)

patches = []
for f in fs:
    center = (f[0], f[1])
    radius = 0.25
    patches.append(Circle(center, radius, fill=True, color='g'))
    
wfsc.imshow1(ref_im, lognorm=True, pxscl=c.psf_pixelscale_lamD, patches=patches)

probe_modes = wfsc.utils.create_probe_poke_modes(Nact, 
                                                 poke_indices=[(Nact//5, Nact//2), (Nact//5+1, Nact//2)], 
                                                 plot=True)

probe_amp = 5e-8
calib_amp = 5e-9

# Run iEFC with recalibrations after a given number of iterations

reg_cond = 1e-3

Wmatrix = np.diag(np.concatenate((weight_map[control_mask], weight_map[control_mask])))

Ncalibs = 5
Nitr = 20
for i in range(Ncalibs):
    print('Calibrations {:d}/{:d}'.format(i+1, Ncalibs))
    response_matrix, response_cube = wfsc.iefc_2dm.calibrate(c, 
                                                             control_mask.ravel(),
                                                             probe_amp, probe_modes, 
                                                             calib_amp, fourier_modes, 
                                                             return_all=True)
    
    response_sum = xp.sum(abs(response_cube), axis=(0,1))
    wfsc.imshow1(response_sum.reshape(npsf, npsf), lognorm=True)
    
    misc.save_fits(iefc_dir/'response-data'/'spc_wide_iefc_2dm_response_matrix_{:d}_{:d}_{:d}.fits'.format(i+1, Ncalibs, date), 
               wfsc.utils.ensure_np_array(response_matrix))
    misc.save_fits(iefc_dir/'response-data'/'spc_wide_iefc_2dm_response_cube_{:d}_{:d}_{:d}.fits'.format(i+1, Ncalibs, date), 
                   wfsc.utils.ensure_np_array(response_cube))
    
    cm_wls = wfsc.utils.WeightedLeastSquares(response_matrix, weight_map, rcond=reg_cond)
    
    images, dm1_commands, dm2_commands = wfsc.iefc_2dm.run(c, 
                                              cm_wls,
                                              probe_modes, 
                                              probe_amp, 
                                              fourier_modes,
                                              control_mask, 
                                              num_iterations=Nitr, 
                                              loop_gain=0.5, 
                                              leakage=0.0,
                                              plot_all=True,
                                             )
    misc.save_fits(iefc_dir/'images'/'spc_wide_iefc_2dm_images_{:d}_{:d}_{:d}.fits'.format(i+1, Ncalibs, date), 
                   wfsc.utils.ensure_np_array(images))
    misc.save_fits(iefc_dir/'dm-commands'/'spc_wide_iefc_2dm_dm1_{:d}_{:d}_{:d}.fits'.format(i+1, Ncalibs, date), 
                   wfsc.utils.ensure_np_array(dm1_commands))
    misc.save_fits(iefc_dir/'dm-commands'/'spc_wide_iefc_2dm_dm2_{:d}_{:d}_{:d}.fits'.format(i+1, Ncalibs, date), 
                   wfsc.utils.ensure_np_array(dm2_commands))


















