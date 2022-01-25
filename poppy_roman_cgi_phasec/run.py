import multiprocessing as mp
import numpy as np
import astropy.units as u

from .import hlc

def run_hlc(params):
    
    mode, wavelength, npix, oversample, npsf, psf_pixelscale, offsets, dm1, dm2, use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, cgi_dir, return_intermediates, quiet = params
#     print('run_hlc')
    psf, wfs = hlc.run(mode=mode,
                           wavelength=wavelength,
                           npix=npix,
                           oversample=oversample,
                           npsf=npsf, 
                           psf_pixelscale=psf_pixelscale,
                           offsets=offsets,
                           dm1=dm1, 
                           dm2=dm2,
                           use_fpm=use_fpm,
                          use_fieldstop=use_fieldstop,
                           use_opds=use_opds,
                           use_pupil_defocus=use_pupil_defocus,
                           polaxis=polaxis,
                           cgi_dir=cgi_dir,
                           display_mode=False,
                           display_intermediates=False,
                           return_intermediates=return_intermediates,
                         quiet=quiet)
    return psf, wfs
    
def run_multi(ncpus=None,
              mode='HLC575',
              wavelength=None,
              npix=310,
              oversample=1024/310,
              npsf=64,
              psf_pixelscale=13e-6*u.m/u.pixel,
              offsets=(0,0),
              dm1=None,
              dm2=None,
              use_fpm=True,
              use_fieldstop=False,
              use_opds=False,
              use_pupil_defocus=False,
              polaxis=0,
              cgi_dir=None,
              return_intermediates=False, 
              quiet=True):
    
    multi_param = None
    params = []
    
    if isinstance(wavelength, np.ndarray) and wavelength.ndim==1 or isinstance(wavelength, list): 
        if not quiet: print('Running mode ' + mode + ' for multiple wavelengths.')
        multi_param = wavelength
        for i in range(len(wavelength)):
            params.append((mode, wavelength[i], npix, oversample, npsf, psf_pixelscale,
                           offsets, dm1, dm2, use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, 
                           cgi_dir, return_intermediates, quiet))
    elif isinstance(dm1, list) and isinstance(dm2, list):
        if not quiet: print('Running mode ' + mode + ' for multiple DM settings.')
        multi_param = dm1
        if len(dm1)==len(dm2):
            for i in range(len(dm1)):
                params.append((mode, wavelength, npix, oversample, npsf, psf_pixelscale,
                               offsets, dm1[i], dm2[i], use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, 
                               cgi_dir, return_intermediates, quiet))
        else: print('The length of the dm1 list must match the length of the dm2 list.')
    else: 
        params.append((mode, wavelength, npix, oversample, npsf, psf_pixelscale,
                       offsets, dm1, dm2, 
                       use_fpm, use_fieldstop, use_opds, use_pupil_defocus, polaxis, 
                       cgi_dir, return_intermediates, quiet))
    
    if ncpus is None: ncpus = mp.cpu_count()
    with mp.get_context("spawn").Pool(ncpus) as pool: results = pool.map(run_hlc, params)
    pool.close()
    pool.join()
    
    psfs = []
    wfs = []
    if multi_param is not None:
        for i in range(len(multi_param)): 
            psfs.append(results[i][0][0])
            if return_intermediates: wfs.append(results[i][1])
            else: wfs.append(results[i][1][-1])
    else:
        psfs.append(results[0][0][0])
        if return_intermediates: wfs.append(results[0][1])
        else: wfs.append(results[0][1][-1])
    
    return psfs, wfs

