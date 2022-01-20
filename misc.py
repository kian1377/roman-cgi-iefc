import numpy as np
import proper 
import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from matplotlib.colors import LogNorm, Normalize
from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy
import astropy.io.fits as fits
import astropy.units as u

import matplotlib
import copy

def myimshow(arr, title=None, 
             npix=None,
             lognorm=False, vmin=None, vmax=None,
             cmap='magma',
             pxscl=None,
             patches=None,
             figsize=(4,4), dpi=125, display_fig=True, return_fig=False):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    
    if npix is not None:
        arr = pad_or_crop(arr, npix)
    
    if pxscl is not None:
        if isinstance(pxscl, u.Quantity):
            if pxscl.unit==(u.meter/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('meters')
            elif pxscl.unit==(u.mm/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('millimeters')
            elif pxscl.unit==(u.arcsec/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('arcsec')
            elif pxscl.unit==(u.mas/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('mas')
        else:
            vext = pxscl * arr.shape[0]/2
            hext = pxscl * arr.shape[1]/2
            extent = [-vext,vext,-hext,hext]
            ax.set_xlabel('lambda/D')
    else:
        extent=None
    
    if lognorm:
        norm = LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = Normalize(vmin=vmin,vmax=vmax)
    im = ax.imshow(arr, cmap=cmap, norm=norm, extent=extent)
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9, rotation=30)
    if patches: 
        for patch in patches:
            ax.add_patch(patch)
            
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax
    
def myimshow2(arr1, arr2, 
              title1=None, title2=None,
              npix=None, npix1=None, npix2=None,
              pxscl=None, pxscl1=None, pxscl2=None,
              cmap1='magma', cmap2='magma',
              lognorm1=False, lognorm2=False,
              vmin1=None, vmax1=None, vmin2=None, vmax2=None, 
              patches1=None, patches2=None,
              display_fig=True, 
              return_fig=False, 
              figsize=(10,4), dpi=125, wspace=0.2):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    if npix is not None:
        arr1 = pad_or_crop(arr1, npix)
        arr2 = pad_or_crop(arr2, npix)
    if npix1 is not None:
        arr1 = pad_or_crop(arr1, npix1)
    if npix2 is not None:
        arr2 = pad_or_crop(arr2, npix2)
    
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
            if pxscl1.unit==(u.meter/u.pix): ax[0].set_xlabel('meters')
            elif pxscl1.unit==(u.millimeter/u.pix): ax[0].set_xlabel('millimeters')
            elif pxscl1.unit==(u.arcsec/u.pix): ax[0].set_xlabel('arcsec')
            elif pxscl1.unit==(u.mas/u.pix): ax[0].set_xlabel('mas')
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
            ax[0].set_xlabel('lambda/D')
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
            if pxscl2.unit==(u.meter/u.pix): ax[1].set_xlabel('meters')
            elif pxscl2.unit==(u.millimeter/u.pix): ax[1].set_xlabel('millimeters')
            elif pxscl2.unit==(u.arcsec/u.pix): ax[1].set_xlabel('arcsec')
            elif pxscl2.unit==(u.mas/u.pix): ax[1].set_xlabel('mas')
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
            ax[1].set_xlabel('lambda/D')
    else:
        extent2=None
    
    if lognorm1: norm1 = LogNorm(vmin=vmin1,vmax=vmax1)
    else: norm1 = Normalize(vmin=vmin1,vmax=vmax1)   
    if lognorm2: norm2 = LogNorm(vmin=vmin2,vmax=vmax2)
    else: norm2 = Normalize(vmin=vmin2,vmax=vmax2)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.subplots_adjust(wspace=wspace)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = np.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def maskgrid(npix, pixelscale):
    vector = (np.linspace(0,npix-1,npix)-npix/2+1/2)*pixelscale
    return np.meshgrid(vector,vector)

def mask_psf(psf, pixelscale, pixelscale_lamD, iwa=None, owa=None):
    x,y = maskgrid(psf.shape[0],pixelscale)
    psf_masked = np.ma.masked_array(psf)
    if iwa is not None:
        iwa = pixelscale/pixelscale_lamD * iwa
        iwa_mask = (x**2+y**2)<iwa**2
        psf_masked = np.ma.masked_array(psf_masked, iwa_mask)
    if owa is not None:
        owa = pixelscale/pixelscale_lamD * owa
        owa_mask = (x**2+y**2)>owa**2
        psf_masked = np.ma.masked_array(psf_masked, owa_mask)
    return psf_masked
        
def display_dm(dm, vmax=2e-7):
    fig, ax = plt.subplots(1,1,figsize=(5,5),dpi=125)
    dm.display(opd_vmax=vmax)
    plt.close()
    display(fig)

def create_annular_roi_mask(npix, pixelscale, iwa, owa, side=None, offset=0, dtype=np.float64):
    if isinstance(pixelscale, u.Quantity): pixelscale=pixelscale.value
    vector = (np.linspace(0,npix-1,npix)-npix/2+1/2)*pixelscale
    x,y = np.meshgrid(vector,vector)
    
    dark_zone = np.ma.masked_array(np.ones((npix,npix), dtype=dtype))
    
    if isinstance(iwa, u.Quantity): iwa = iwa.value
    iwa_mask = (x**2+y**2)<iwa**2
    dark_zone = np.ma.masked_array(dark_zone, iwa_mask)
    
    if isinstance(owa, u.Quantity): owa=owa.value
    owa_mask = (x**2+y**2)>owa**2
    dark_zone = np.ma.masked_array(dark_zone, owa_mask)
    
    if isinstance(offset, u.Quantity): offset=offset.value
    if side=='right': dark_zone = np.ma.masked_array(dark_zone, x<offset)
    elif side=='left': dark_zone = np.ma.masked_array(dark_zone, x>offset)
        
    return dark_zone

# for animations
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\ffmpeg-20191023\\bin\\ffmpeg.exe'
plt.rcParams['animation.convert_path'] = r'C:\\Program Files\\ImageMagick-7.0.10-Q16-HDRI\\magick.exe'
plt.rcParams['animation.html'] = 'html5'
plt.rcParams.update({'image.origin': 'lower',
                     'image.interpolation':'nearest'})

def create_anim(arrs1,arrs2,lognorms=False):
    numframes = len(times); print(numframes)

    # finding the maximum of all the PSFs to keep colorbar constant
    max_arrs1 = 0
    min_arrs1 = 1
    for arr in arrs1:
        if np.max(arr) > max_arrs1: max_arrs1 = np.max(arr)
        if np.min(arr) < min_arrs1: min_arrs1 = np.min(arr)
    print(max_arrs1,min_arrs1)

    # finding the maximum of all the PSFs to keep colorbar constant
    max_arrs2 = 0
    min_arrs2 = 1
    for arr in arrs2:
        if np.max(arr) > max_arrs2: max_arrs2 = np.max(arr)
        if np.min(arr) < min_arrs2: min_arrs2 = np.min(arr)
    print(max_arrs2,min_arrs2)

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4),dpi=150)
    
    if lognorms:
        im1 = ax[0].imshow(arrs1[0,:,:], norm=LogNorm())
        im2 = ax[1].imshow(arrs2[0,:,:], norm=LogNorm())
    else:
        im1 = ax[0].imshow(arrs1[0,:,:])
        im2 = ax[1].imshow(arrs2[0,:,:])
    cbar1 = fig.colorbar(im1,ax=ax[0])
    im1_title = ax[0].set_title('Time Instance {:.5f}s'.format(times[0]), fontsize = 18)

    cbar2 = fig.colorbar(im2,ax=ax[1])
    im2_title = ax[1].set_title('Time Instance {:.5f}s'.format(times[0]), fontsize = 18)

    def animate(i):
        im1.set_data(arrs1[i,:,:])
        im1.set_clim(min_arrs1, max_arrs1)
        im1_title.set_text('Time Instance {:.5f}s'.format(times[i]))

        im2.set_data(arrs2[i,:,:])
        im2.set_clim(min_arrs2, max_arrs2)
        im2_title.set_text('Time Instance {:.5f}s'.format(times[i]))

    anim = animation.FuncAnimation(fig, animate, frames=numframes)
    return anim
        