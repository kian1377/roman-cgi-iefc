import numpy as np
from matplotlib import pyplot as plt
from datetime import date
from astropy.io import fits
import astropy.units as u
import time
from datetime import date
from pathlib import Path

import poppy
from poppy_roman_cgi_phasec import hlc, spc, run

import proper
import roman_phasec_proper as phasec

import misc

class CGI_POPPY():

    def __init__(self, cgi_mode='HLC575', wavelength=None, npsf=64, psf_pixelscale=13e-6*u.m/u.pix,
                 use_pupil_defocus=True, use_fieldstop=False, use_opds=False, polaxis=0, quiet=True):
        
        self.cgi_mode = cgi_mode
        if cgi_mode=='HLC575': 
            self.wavelength_c = 575e-9*u.m
            self.npix = 310
            self.oversample = 1024/310
        elif cgi_mode=='SPC730':
            self.wavelength_c = 730e-9*u.m
            self.npix = 1000
            self.oversample = 2.048
        elif cgi_mode=='SPC825':
            self.wavelength_c = 730e-9*u.m
            self.npix = 1000
            self.oversample = 2.048
            
        if wavelength is None: self.wavelength = self.wavelength_c
        self.npsf = npsf
        self.psf_pixelscale = psf_pixelscale
        self.use_pupil_defocus = use_pupil_defocus
        self.use_fieldstop = use_fieldstop
        self.use_opds = use_opds
        self.polaxis = polaxis
        self.quiet = quiet
        
        self.cgi_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
        self.dm_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data/dm-acts')
        
        self.Nact = 48
        self.dm_diam = 46.3*u.mm
        self.act_spacing = 0.9906*u.mm
        
        self.DM1 = poppy.ContinuousDeformableMirror(name='DM1', dm_shape=(self.Nact,self.Nact), actuator_spacing=self.act_spacing, 
                                               radius=self.dm_diam/2, influence_func=str(self.dm_dir/'proper_inf_func.fits'))
        self.DM2 = poppy.ContinuousDeformableMirror(name='DM2', dm_shape=(self.Nact,self.Nact), actuator_spacing=self.act_spacing, 
                                               radius=self.dm_diam/2, influence_func=str(self.dm_dir/'proper_inf_func.fits'))
        
        self.texp = 1
    
    def reset_dms(self):
        self.DM1.set_surface( np.zeros((self.Nact, self.Nact)) )
        self.DM2.set_surface( np.zeros((self.Nact, self.Nact)) )  
        
    def check_dm_command_shape(self, dm_command):
        if dm_command.shape[0]==self.Nact**2 or dm_command.shape[1]==self.Nact**2:
            dm_command = dm_command.reshape((self.Nact, self.Nact))
        return dm_command
            
    def set_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1.set_surface(dm_command)
    
    def set_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2.set_surface(dm_command)
        
    def add_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1.set_surface(self.DM1.surface + dm_command)
        
    def add_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2.set_surface(self.DM2.surface + dm_command)
        
    def calc_psf(self):
        
        if self.cgi_mode=='HLC575':
            psf, wf = hlc.run(mode=self.cgi_mode, wavelength=self.wavelength, npix=self.npix, oversample=self.oversample, 
                                 npsf=self.npsf, psf_pixelscale=self.psf_pixelscale,
                                 dm1=self.DM1, dm2=self.DM2,
                                 use_opds=self.use_opds, polaxis=self.polaxis,
                                 use_pupil_defocus=self.use_pupil_defocus, use_fieldstop=self.use_fieldstop,
                                 cgi_dir=self.cgi_dir, quiet=self.quiet)
        else:
            psf, wf = spc.run(mode=self.cgi_mode, wavelength=self.wavelength, npix=self.npix, oversample=self.oversample,
                                 npsf=self.npsf, psf_pixelscale=self.psf_pixelscale,
                                 dm1=self.DM1, dm2=self.DM2,
                                 use_opds=self.use_opds, polaxis=self.polaxis,
                                 use_pupil_defocus=self.use_pupil_defocus,
                                 cgi_dir=self.cgi_dir, quiet=self.quiet)
            
        return wf[-1].intensity
    
    def calc_psfs(self, dm1_commands, dm2_commands):
        ''' dm1_commands and dm2_commands must be lists of np.ndarrays '''
        
        for i in range(len(dm1_commands)): 
            dm1_commands[i] += self.DM1.surface
            dm2_commands[i] += self.DM2.surface
#             misc.myimshow(dm1_commands[i])
            
        psfs, wfs = run.run_multi(ncpus=16, mode=self.cgi_mode, wavelength=self.wavelength, 
                                  npix=self.npix, oversample=self.oversample, 
                                  npsf=self.npsf, psf_pixelscale=self.psf_pixelscale,
                                 dm1=dm1_commands, dm2=dm2_commands,
                                 use_opds=self.use_opds, polaxis=self.polaxis,
                                 use_pupil_defocus=self.use_pupil_defocus, use_fieldstop=self.use_fieldstop,
                                 cgi_dir=self.cgi_dir, quiet=self.quiet)
        return wfs
    
class CGI_PROPER():

    def __init__(self, cgi_mode='hlc', wavelength=None, npsf=64, psf_pixelscale=13e-6*u.m/u.pix,
                 use_pupil_defocus=True, use_fieldstop=False, use_opds=False, polaxis=0, quiet=True):
        
        self.cgi_mode = cgi_mode
        if cgi_mode.find('hlc') != -1: 
            self.wavelength_c = 575e-9*u.m
            self.npix = 309
        elif cgi_mode=='spc-spec':
            self.wavelength_c = 730e-9*u.m
        elif cgi_mode=='spc-wide':
            self.wavelength_c = 825e-9*u.m
            
        if wavelength is None: self.wavelength = self.wavelength_c
        self.npsf = npsf
        self.psf_pixelscale = psf_pixelscale
        self.use_pupil_defocus = use_pupil_defocus
        self.use_fieldstop = use_fieldstop
        self.use_opds = use_opds
        self.polaxis = polaxis
        self.quiet = quiet
        
        self.cgi_dir = phasec.data_dir
        
        self.Nact = 48
        self.dm_diam = 46.3*u.mm
        self.act_spacing = 0.9906*u.mm
        
        self.DM1 = np.zeros((self.Nact,self.Nact))
        self.DM2 = np.zeros((self.Nact,self.Nact))
        
        self.texp = 1
    
    def reset_dms(self):
        self.DM1 = np.zeros((self.Nact,self.Nact))
        self.DM2 = np.zeros((self.Nact,self.Nact))
        
    def check_dm_command_shape(self, dm_command):
        if dm_command.shape[0]==self.Nact**2 or dm_command.shape[1]==self.Nact**2:
            dm_command = dm_command.reshape((self.Nact, self.Nact))
        return dm_command
            
    def set_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1 = dm_command
    
    def set_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2 = dm_command
        
    def add_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1 += dm_command
        
    def add_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2 += dm_command
        
    def calc_psf(self):
        
        options = {'cor_type':self.cgi_mode,
                   'final_sampling_m':self.psf_pixelscale.to(u.m/u.pix).value, 
                   'use_errors':self.use_opds,
                   'polaxis':self.polaxis,
                   'use_field_stop':self.use_fieldstop,
                   'use_pupil_defocus':self.use_pupil_defocus,
                   'use_dm1':1, 'dm1_m':self.DM1, 
                   'use_dm2':1, 'dm2_m':self.DM2, }
        
        (wfs, sampling_m) = proper.prop_run_multi('roman_phasec', 
                                                  self.wavelength.to(u.micron).value, self.npsf, QUIET=self.quiet, 
                                                  PASSVALUE=options)
        psf = np.sum(np.abs(wfs)**2, axis=0)
        return psf
    
    def calc_psfs(self, dm1_commands, dm2_commands):
        ''' dm1_commands and dm2_commands must be lists of np.ndarrays '''
        
        options = []
        for i in range(len(dm1_commands)): 
            dm1_commands[i] += self.DM1
            dm2_commands[i] += self.DM2
            options.append({'cor_type':self.cgi_mode,
                            'final_sampling_m':self.psf_pixelscale.to(u.m/u.pix).value, 
                            'use_errors':self.use_opds,
                            'polaxis':self.polaxis,
                            'use_field_stop':self.use_fieldstop,
                            'use_pupil_defocus':self.use_pupil_defocus,
                            'use_dm1':1, 'dm1_m':dm1_commands[i], 
                            'use_dm2':1, 'dm2_m':dm2_commands[i], 
                           })
            
        (wfs, sampling_m) = proper.prop_run_multi('roman_phasec', 
                                                  self.wavelength.to(u.micron).value, self.npsf, QUIET=self.quiet, 
                                                  PASSVALUE=options)
        
        psfs = np.abs(wfs)**2
        return psfs
        
        
        
        
        
        
        