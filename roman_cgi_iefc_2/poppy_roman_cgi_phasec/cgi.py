import numpy as np
from astropy.io import fits
import astropy.units as u
import time
from pathlib import Path
from scipy.interpolate import interp1d

import poppy
from poppy.poppy_core import PlaneType
if poppy.accel_math._USE_CUPY:
    import cupy as cp

import proper
import roman_phasec_proper as phasec    

from . import hlc, spc, polmap

import ray

cgi_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
dm_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data/dm-acts')

class CGI_POPPY():

    def __init__(self, cgi_mode='HLC575', wavelength=None, npsf=64, psf_pixelscale=13e-6*u.m/u.pix, psf_pixelscale_lamD=None,
                 offset=(0,0), use_pupil_defocus=True, use_fieldstop=False, use_opds=False, use_fpm=True, polaxis=0, 
                 return_intermediates=False, 
                 quiet=True, ngpus=0.5):
        
        self.pupil_diam = 2.363114*u.m
        
        self.cgi_mode = cgi_mode
        if cgi_mode=='HLC575': 
            self.wavelength_c = 575e-9*u.m
            self.npix = 310
            self.oversample = 1024/310
            self.D = self.pupil_diam*self.npix/309
        elif cgi_mode=='SPC730':
            self.wavelength_c = 730e-9*u.m
            self.npix = 1000
            self.oversample = 2.048
            self.D = self.pupil_diam
        elif cgi_mode=='SPC825':
            self.wavelength_c = 825e-9*u.m
            self.npix = 1000
            self.oversample = 2.048
            self.D = self.pupil_diam
            
        if wavelength is None: self.wavelength = self.wavelength_c
        
        self.offset = offset
        self.offset = offset
        self.use_fpm = use_fpm
        self.use_pupil_defocus = use_pupil_defocus
        self.use_fieldstop = use_fieldstop
        self.use_opds = use_opds
        self.polaxis = polaxis
        
        self.npsf = npsf
        if psf_pixelscale_lamD is not None: # overrides psf_pixelscale this way
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 13e-6*u.m/u.pix / (0.5e-6/self.wavelength_c.value) * self.psf_pixelscale_lamD/0.5
        else:
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = 1/2 * 0.5e-6/self.wavelength_c.value * self.psf_pixelscale.to(u.m/u.pix).value/13e-6
        
        self.texp = 1
        
        self.init_mode_optics()
        self.init_dms()
        self.init_inwave()
        
        self.return_intermediates = return_intermediates
        self.quiet = quiet
        
        self.ngpus = ngpus
        
        
    
    def init_mode_optics(self):
        self.FPM_plane = poppy.ScalarTransmission('FPM Plane (No Optic)', planetype=PlaneType.intermediate) # placeholder
        
        if self.cgi_mode=='HLC575':
            self.optics_dir = cgi_dir/'hlc'
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil.fits'),
                                                  planetype=PlaneType.pupil)
            self.LS = poppy.FITSOpticalElement('Lyot Stop', 
                                               transmission=str(self.optics_dir/'lyot_rotated.fits'), 
                                               planetype=PlaneType.pupil)
            self.SPM = poppy.ScalarTransmission('SPM Plane (No Optic)', planetype=PlaneType.intermediate)
            if self.use_fpm:
                # Find nearest available FPM wavelength that matches specified wavelength and initialize the FPM data
                lam_um = self.wavelength.value * 1e6
                f = open( str(self.optics_dir/'fpm_files.txt') )
                fpm_nlam = int(f.readline())
                fpm_lams = np.zeros((fpm_nlam),dtype=float)
                for j in range(0,fpm_nlam): 
                    fpm_lams[j] = float(f.readline())*1e-6
                fpm_root_fnames = [j.strip() for j in f.readlines()] 
                f.close()

                diff = np.abs(fpm_lams - self.wavelength.value)
                w = np.argmin( diff )
                if diff[w] > 0.1e-9: 
                    raise Exception('Only wavelengths within 0.1nm of avalable FPM wavelengths can be used.'
                                    'Closest available to requested wavelength is {}.'.format(fpm_lams[w]))
                fpm_rootname = self.optics_dir/fpm_root_fnames[w]

                fpm_r_fname = str(fpm_rootname)+'real.fits'
                fpm_i_fname = str(fpm_rootname)+'imag.fits'

                fpm_r = fits.getdata(fpm_r_fname)
                fpm_i = fits.getdata(fpm_i_fname)
                if poppy.accel_math._USE_CUPY:
                    fpm_r = cp.array(fpm_r)
                    fpm_i = cp.array(fpm_i)
                self.fpm_phasor = fpm_r + 1j*fpm_i
                self.fpm_mask = (fpm_r != fpm_r[0,0]).astype(int)
                fpm_ref_wavelength = fits.getheader(fpm_r_fname)['WAVELENC']
                fpm_pxscl_lamD = fits.getheader(fpm_r_fname)['PIXSCLLD']

                self.FPM = poppy.FixedSamplingImagePlaneElement('COMPLEX OCCULTER', fpm_r_fname)
                self.FPM.amplitude = np.abs(self.fpm_phasor)
                self.FPM.opd = np.angle(self.fpm_phasor)*fpm_ref_wavelength/(2*np.pi)
            else:
                self.FPM = None
            
            if self.use_fieldstop: 
                radius = 9.7/(309/(self.npix*self.oversample)) * (self.wavelength_c/self.wavelength) * 7.229503001768824e-06*u.m
                self.fieldstop = poppy.CircularAperture(radius=radius, name='HLC Field Stop')
            else: 
                self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
        elif self.cgi_mode=='SPC730': 
            self.optics_dir = cgi_dir/'spc-spec'            
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil_SPC-20200617_1000.fits'),
                                                  planetype=PlaneType.pupil)
            self.SPM = poppy.FITSOpticalElement('SPM', 
                                                transmission=str(self.optics_dir/'SPM_SPC-20200617_1000_rounded9_rotated.fits'),
                                                planetype=PlaneType.intermediate)
            self.LS = poppy.FITSOpticalElement('Lyot Stop',
                                               transmission=str(self.optics_dir/'LS_SPC-20200617_1000.fits'), 
                                               planetype=PlaneType.pupil)
            if self.use_fpm: 
                self.FPM = poppy.FixedSamplingImagePlaneElement('FPM', 
                                                                transmission=str(self.optics_dir/'fpm_0.05lamD.fits'))
            else: 
                self.FPM = poppy.ScalarTransmission(name='FPM Plane (No Optic)', planetype=PlaneType.intermediate) 
            self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
        elif self.cgi_mode=='SPC825':
            self.optics_dir = cgi_dir/'spc-wide'            
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil_SPC-20200610_1000.fits'),
                                                  planetype=PlaneType.pupil)
            self.SPM = poppy.FITSOpticalElement('SPM', 
                                                str(self.optics_dir/'SPM_SPC-20200610_1000_rounded9_gray_rotated.fits'),
                                                planetype=PlaneType.intermediate)
            self.LS = poppy.FITSOpticalElement('Lyot Stop',
                                               transmission=str(self.optics_dir/'LS_SPC-20200610_1000.fits'), 
                                               planetype=PlaneType.pupil)
            if self.use_fpm: 
                self.FPM = poppy.FixedSamplingImagePlaneElement('FPM', 
                                                                str(self.optics_dir/'FPM_SPC-20200610_0.1_lamc_div_D.fits'))
            else: 
                self.FPM = poppy.ScalarTransmission(name='FPM Plane (No Optic)', planetype=PlaneType.intermediate) 
            self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
        
    # DM methods
    def init_dms(self):
        self.Nact = 48
        self.dm_diam = 46.3*u.mm
        self.act_spacing = 0.9906*u.mm
        
        self.DM1 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM1', 
                                                    actuator_spacing=self.act_spacing, radius=self.dm_diam/2,
                                                    influence_func=str(dm_dir/'proper_inf_func.fits'))
        self.DM2 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM1', 
                                                    actuator_spacing=self.act_spacing, radius=self.dm_diam/2,
                                                    influence_func=str(dm_dir/'proper_inf_func.fits'))
    
    def reset_dms(self):
        self.DM1.set_surface( np.zeros((self.Nact, self.Nact)) )
        self.DM2.set_surface( np.zeros((self.Nact, self.Nact)) )
            
    def set_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1.set_surface(dm_command)
    
    def set_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2.set_surface(dm_command)
        
    def add_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1.set_surface(self.DM1.surface.get() + dm_command) # I should make the DM.surface attribute be Numpy no matter what
        
    def add_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2.set_surface(self.DM2.surface.get() + dm_command)
    
    def check_dm_command_shape(self, dm_command):
        if dm_command.shape[0]==self.Nact**2 or dm_command.shape[1]==self.Nact**2: # passes if shape does not have 2 values
            dm_command = dm_command.reshape((self.Nact, self.Nact))
        return dm_command
    
    # utility functions
    def glass_index(self, glass):
        a = np.loadtxt( str( cgi_dir/'glass'/(glass+'_index.txt') ) )  # lambda_um index pairs
        f = interp1d( a[:,0], a[:,1], kind='cubic' )
        return f( self.wavelength.value*1e6 )

    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.D/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        if self.polaxis!=0: 
            polfile = cgi_dir/'pol'/'phasec_pol'
            polmap.polmap( inwave, str(polfile), self.npix, self.polaxis )

        xoffset = self.offset[0]
        yoffset = self.offset[1]
        xoffset_lam = xoffset * (self.wavelength_c / self.wavelength).value # maybe use negative sign
        yoffset_lam = yoffset * (self.wavelength_c / self.wavelength).value 
        n = int(round(self.npix*self.oversample))
        if poppy.accel_math._USE_CUPY:
            x = cp.tile( (cp.arange(n)-n//2)/(self.npix/2.0), (n,1) )
            y = cp.transpose(x)
        else:
            x = np.tile( (np.arange(n)-n//2)/(self.npix/2.0), (n,1) )
            y = np.transpose(x)

        inwave.wavefront *= np.exp(complex(0,1) * np.pi * (xoffset_lam * x + yoffset_lam * y))

        self.inwave = inwave
    
    # Methods to calculate PSFs in parallel
    def calc_psf(self):
        start = time.time()
        if not self.quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
            
        if self.cgi_mode=='HLC575':
            psf, wfs = hlc.run(self)
        else: 
            psf, wfs = spc.run(self)
        
        if not self.quiet: print('PSF calculated in {:.2f}s'.format(time.time()-start))
        return wfs
    _calc_psf = ray.remote(calc_psf)
    
    def calc_psfs(self, wavelengths=None, dm_commands=None, offsets=None):
        start = time.time()
        
        if wavelengths is None:
            wavelengths = [self.wavelength_c]
        if dm_commands is None:
            dm_commands = [np.zeros((2,self.Nact**2))]
        if offsets is None:
            offsets = [(0,0)]
        
        pending_results = []
        for i in range(len(wavelengths)):
            self.wavelength = wavelengths[i]
            for j in range(len(dm_commands)):
                self.add_dm1(dm_commands[j][0])
                self.add_dm2(dm_commands[j][1])
                for k in range(len(offsets)):
                    self.offset = offsets[k]
                    self.init_inwave()
                    
                    if poppy.accel_math._USE_CUPY:
                        ref = self._calc_psf.options(num_gpus=self.ngpus).remote(self)
                    else: 
                        ref = self._calc_psf.remote(self)
                    pending_results.append(ref)
        wfs = ray.get(pending_results)
        if not self.quiet: print('All PSFs calculated in {:.3f}s'.format(time.time()-start))
        
        return wfs
    
    
    
class CGI_PROPER():

    def __init__(self, cgi_mode='hlc', wavelength=None, npsf=64, psf_pixelscale=13e-6*u.m/u.pix, psf_pixelscale_lamD=None,
                 use_pupil_defocus=True, use_fieldstop=False, use_opds=False, use_fpm=True, polaxis=0, quiet=True):
        
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
        
        if psf_pixelscale_lamD is not None: # overrides psf_pixelscale this way
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 13e-6*u.m/u.pix / (0.5e-6/self.wavelength_c.value) * self.psf_pixelscale_lamD/0.5
        else:
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = 1/2 * 0.5e-6/self.wavelength_c.value * self.psf_pixelscale.to(u.m/u.pix).value/13e-6
        
        self.use_fpm = use_fpm
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
                   'use_fpm':self.use_fpm,
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
                            'use_fpm':self.use_fpm,
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
        
        
        
        
        
        