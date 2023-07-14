import numpy as np
import scipy

import cupy as cp
import cupyx.scipy

import astropy.units as u
from astropy.io import fits
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from IPython.display import clear_output
from importlib import reload
import time
import copy

import warnings
warnings.filterwarnings("ignore")

import poppy
import hcipy as hci

class PC(): # perfect coronagraph

    def __init__(self, 
                 wavelength=None, 
                 dm1_dm2=200e-3*u.m,
                 aberration_distance=100e-3*u.m,
                 influence_functions=None,
                 norm=None,
                ):
        
        self.wavelength_c = 575e-9*u.m
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        self.f = 500*u.mm
        
        self.dm_diam = 10.2*u.mm
        self.dm1_dm2 = dm1_dm2
        
        self.FN = (self.dm_diam**2/self.dm1_dm2/self.wavelength).decompose()
        
        # define the grids
        self.npix = 256
        self.oversample = 2
        self.spatial_resolution = self.f.to_value(u.m) * self.wavelength.to_value(u.m) / self.dm_diam.to_value(u.m)
        self.npsf = 128
        self.psf_pixelscale_lamD = 1/4
        
        self.pupil_grid = hci.make_pupil_grid(self.npix*self.oversample, self.dm_diam.to_value(u.m) * self.oversample)
        self.focal_grid = hci.make_focal_grid(4, 16, spatial_resolution=self.spatial_resolution)
        self.prop = hci.FraunhoferPropagator(self.pupil_grid, self.focal_grid, self.f.to_value(u.m))

        self.aperture = hci.Field(np.exp(-(self.pupil_grid.as_('polar').r / (0.5 * self.dm_diam.to_value(u.m)))**30), self.pupil_grid)
        
        # define the DMs
        self.Nact = 34
        self.actuator_spacing = 1.00 / self.Nact * self.dm_diam.to_value(u.m)
        
        self.norm = norm
        
        # make the aberrations of the system (make sure to remove tip-tilt)
        aberration_ptv = 0.02 * self.wavelength.to_value(u.m)
        tip_tilt = hci.make_zernike_basis(3, self.dm_diam.to_value(u.m), self.pupil_grid, starting_mode=2)
        wfe = hci.SurfaceAberration(self.pupil_grid, aberration_ptv, self.dm_diam.to_value(u.m), remove_modes=tip_tilt, exponent=-3)
        self.aberration_distance_m = aberration_distance.to_value(u.m)
        self.wfe_at_distance = hci.SurfaceAberrationAtDistance(wfe, self.aberration_distance_m)
        
        # make the coronagraph model
        self.coronagraph = hci.PerfectCoronagraph(self.aperture, order=6)
        
        if influence_functions is not None: 
            self.init_dms(influence_functions)
            
    def init_dms(self, influence_functions):
        self.DM1 = hci.DeformableMirror(influence_functions)
        self.DM2 = hci.DeformableMirror(influence_functions)

        self.prop_between_dms = hci.FresnelPropagator(self.pupil_grid, self.dm1_dm2.to_value(u.m))
        
        self.dm_mask = np.ones((self.Nact,self.Nact), dtype=bool)
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.actuator_spacing
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>0.0105] = 0 # had to set the threshold to 10.5 instead of 10.2 to include edge actuators
        
#         self.dm_zernikes = poppy.zernike.arbitrary_basis(self.dm_mask, nterms=15, outside=0)
        self.dm_zernikes = poppy.zernike.arbitrary_basis(cp.array(self.dm_mask), nterms=15, outside=0).get()
        
        
    def set_dm1(self, command):
        self.DM1.actuators = command.ravel()
        
    def set_dm2(self, command):
        self.DM2.actuators = command.ravel()
        
    def add_dm1(self, command):
        self.DM1.actuators += command.ravel()
        
    def add_dm2(self, command):
        self.DM2.actuators += command.ravel()
        
    def get_dm1(self):
        return self.DM1.actuators.reshape(self.Nact, self.Nact)
        
    def get_dm2(self):
        return self.DM2.actuators.reshape(self.Nact, self.Nact)
    
    def reset_dms(self):
        self.DM1.actuators = np.zeros((self.Nact, self.Nact)).ravel()
        self.DM2.actuators = np.zeros((self.Nact, self.Nact)).ravel()
    
    def show_dms(self):
        misc.imshow2(self.DM1.actuators.reshape(c.Nact, c.Nact),
                     self.DM2.actuators.reshape(c.Nact, c.Nact))
        
    def calc_psf(self):
        wf = hci.Wavefront(self.aperture, self.wavelength.to_value(u.m))
        wf = self.wfe_at_distance(wf) # apply aberrations to wavefront
        wf = self.prop_between_dms.backward(self.DM2(self.prop_between_dms.forward(self.DM1(wf)))) # apply DMs
        wf = self.prop(self.coronagraph(wf))
        psf_wf = wf.real.shaped + 1j*wf.imag.shaped

        return np.array(psf_wf)

    def snap(self):
        wf = hci.Wavefront(self.aperture, self.wavelength.to_value(u.m))
        wf = self.wfe_at_distance(wf) # apply aberrations to wavefront
        wf = self.prop_between_dms.backward(self.DM2(self.prop_between_dms.forward(self.DM1(wf)))) # apply DMs
        im = self.prop(self.coronagraph(wf))
        im = np.array(im.intensity.shaped)
        if self.norm is not None: 
            im /= self.norm
            
        return im

    
class SVC(): # scalar vortex coronagraph

    def __init__(self, 
                 npix = 256,
                 oversample = 4,
                 psf_pixelscale_lamD=1/4,
                 fgrid_extent = 16,
                 wavelength=None, 
#                  pupil_diam=6.5*u.m,
                 pupil_diam=10*u.mm,
                 fnum=20,
                 dm1_dm2=200e-3*u.m,
                 aberration_distance=100e-3*u.m,
                 influence_functions=None,
                 use_fpm=True,
                 charge=6,
                 norm=None,
                ):
        
        self.wavelength_c = 650e-9*u.m
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        self.pupil_diam = pupil_diam
        self.fnum = fnum
        
        self.npix = npix
        self.oversample = oversample
        self.dm1_dm2 = dm1_dm2
        
        self.FN = self.pupil_diam.to_value(u.m)**2/self.dm1_dm2.to_value(u.m)/self.wavelength.to_value(u.m)
        
        # define the grids and the optics
        self.npix = npix
        self.oversample = oversample
        
        self.psf_pixelscale_lamD = psf_pixelscale_lamD
        self.fgrid_sampling = 1/psf_pixelscale_lamD # number of pixels per resolution element (lambda*fnum)
        self.fgrid_extent = fgrid_extent # number of resolution elements across grid (lambda*fnum)
        
        self.pupil_grid = hci.make_pupil_grid(int(npix*oversample), pupil_diam.to_value(u.m) * oversample)
        self.focal_grid = hci.make_focal_grid(self.fgrid_sampling, self.fgrid_extent, 
                                         pupil_diameter=self.pupil_diam.to_value(u.m),
                                         focal_length=self.fnum * self.pupil_diam.to_value(u.m),
                                         reference_wavelength=self.wavelength_c.to_value(u.m))
        self.npsf = self.focal_grid.dims[0]
        self.norm = norm
        
        self.prop = hci.FraunhoferPropagator(self.pupil_grid, self.focal_grid,
                                             focal_length=self.fnum*self.pupil_diam.to_value(u.m))
        
        self.aperture = hci.evaluate_supersampled(hci.circular_aperture(self.pupil_diam.to_value(u.m)),
                                                  self.pupil_grid, 
                                                  self.oversample)
        self.lyot_mask = hci.evaluate_supersampled(hci.circular_aperture(0.90*self.pupil_diam.to_value(u.m)), 
                                                   self.pupil_grid, 
                                                   self.oversample)

        # define the DMs
        self.Nact = 34
        self.actuator_spacing = 1.00 * self.pupil_diam.to_value(u.m)/self.Nact
        
        
        # make the aberrations of the system (make sure to remove tip-tilt)
        aberration_ptv = 0.02 * self.wavelength_c.to_value(u.m)
        tip_tilt = hci.make_zernike_basis(3, self.pupil_diam.to_value(u.m), self.pupil_grid, starting_mode=2)
        wfe = hci.SurfaceAberration(self.pupil_grid, aberration_ptv, 
                                    self.pupil_diam.to_value(u.m), remove_modes=tip_tilt, exponent=-3)

        self.aberration_distance = aberration_distance
        self.wfe_at_distance = hci.SurfaceAberrationAtDistance(wfe, self.aberration_distance.to_value(u.m))
        
        # make the coronagraph model
        self.charge = charge
        self.coro = hci.VortexCoronagraph(self.pupil_grid, self.charge)
        self.lyot_stop = hci.Apodizer(self.lyot_mask)
        
        self.use_fpm = use_fpm
        
        if influence_functions is not None: 
            self.init_dms(influence_functions)
            
    def init_dms(self, influence_functions):
        self.DM1 = hci.DeformableMirror(influence_functions)
        self.DM2 = hci.DeformableMirror(influence_functions)
        
        self.prop_between_dms = hci.FresnelPropagator(self.pupil_grid, self.dm1_dm2.to_value(u.m))
        
        self.dm_mask = np.ones((self.Nact,self.Nact), dtype=bool)
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.actuator_spacing
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>(self.pupil_diam.to_value(u.m)+ self.actuator_spacing)/2] = 0
        
        self.Nacts = int(self.dm_mask.sum())
        
#         self.dm_zernikes = poppy.zernike.arbitrary_basis(self.dm_mask, nterms=15, outside=0)
#         self.dm_zernikes = poppy.zernike.arbitrary_basis(cp.array(self.dm_mask), nterms=15, outside=0).get()
        
        
    def set_dm1(self, command):
        self.DM1.actuators = command.ravel()
        
    def set_dm2(self, command):
        self.DM2.actuators = command.ravel()
        
    def add_dm1(self, command):
        self.DM1.actuators += command.ravel()
        
    def add_dm2(self, command):
        self.DM2.actuators += command.ravel()
        
    def get_dm1(self):
        return self.DM1.actuators.reshape(self.Nact, self.Nact)
        
    def get_dm2(self):
        return self.DM2.actuators.reshape(self.Nact, self.Nact)
    
    def reset_dms(self):
        self.DM1.actuators = np.zeros((self.Nact, self.Nact)).ravel()
        self.DM2.actuators = np.zeros((self.Nact, self.Nact)).ravel()
    
    def show_dms(self):
        misc.imshow2(self.DM1.actuators.reshape(c.Nact, c.Nact),
                     self.DM2.actuators.reshape(c.Nact, c.Nact))
    
    def calc_wfs(self):
        wfs = []
        
        wf = hci.Wavefront(self.aperture, self.wavelength.to_value(u.m))
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        wf = self.wfe_at_distance(wf)
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        wf = self.DM1(wf)
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        wf = self.DM2(self.prop_between_dms.forward(wf))
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        wf = self.prop_between_dms.backward(wf)
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        wf = self.coro(wf)
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        wf = self.lyot_stop(wf)
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        wf = self.prop(wf)
        wfs.append(copy.copy(wf.real.shaped + 1j*wf.imag.shaped))
        
        return wfs
    
    def calc_psf(self):
        wf = hci.Wavefront(self.aperture, self.wavelength.to_value(u.m))
        wf = self.wfe_at_distance(wf)
        wf = self.prop_between_dms.backward(self.DM2(self.prop_between_dms.forward(self.DM1(wf))))
        if self.use_fpm:
            wf = self.coro(wf)
            wf = self.lyot_stop(wf)
        wf = self.prop(wf)
        
        efield = wf.real.shaped + 1j*wf.imag.shaped
        
        if self.norm is not None:
            efield /= np.sqrt(self.norm)
            
        return np.array(efield)

    def snap(self):
        
        wf = hci.Wavefront(self.aperture, self.wavelength.to_value(u.m))
        wf = self.wfe_at_distance(wf)
        wf = self.prop_between_dms.backward(self.DM2(self.prop_between_dms.forward(self.DM1(wf))))
        if self.use_fpm:
            wf = self.coro(wf)
            wf = self.lyot_stop(wf)
        im = self.prop(wf).intensity.shaped
        
        if self.norm is not None:
            im /= self.norm
        
        return im
    
    
    
    
    
    