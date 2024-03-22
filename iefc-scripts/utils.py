from math_module import xp, _scipy, ensure_np_array
import imshows

import numpy as np
import scipy
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from IPython.display import display
from astropy.io import fits

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        # if isinstance(arr_in, np.ndarray):
        #     arr_out = np.zeros((npix,npix), dtype=arr_in.dtype)
        # elif isinstance(arr_in, cp.ndarray):
        #     arr_out = cp.zeros((npix,npix), dtype=arr_in.dtype)
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def centroid(arr, rounded=False):
    weighted_sum_x = 0
    total_sum_x = 0
    for i in range(arr.shape[1]):
        weighted_sum_x += np.sum(arr[:,i])*i
        total_sum_x += np.sum(arr[:,i])
    xc = round(weighted_sum_x/total_sum_x) if rounded else weighted_sum_x/total_sum_x
    
    weighted_sum_y = 0
    total_sum_y = 0
    for i in range(arr.shape[0]):
        weighted_sum_y += np.sum(arr[i,:])*i
        total_sum_y += np.sum(arr[i,:])
        
    yc = round(weighted_sum_y/total_sum_y) if rounded else weighted_sum_y/total_sum_y
    return (yc, xc)

def map_acts_to_dm(actuators, dm_mask, Nact=48):
    inds = xp.where(xp.array(dm_mask).flatten().astype(int))[0]
    
    command = xp.zeros((Nact, Nact))
    command.ravel()[inds] = actuators
    
    return command

# Create control matrix
def WeightedLeastSquares(A, weight_map, algorithm='iefc', nprobes=2, rcond=1e-1):
    control_mask = weight_map > 0
    w = weight_map[control_mask]
    if algorithm=='iefc':
        for i in range(nprobes-1):
            w = xp.concatenate((w, weight_map[control_mask]))
    elif algorithm=='efc':
        temp = xp.zeros(2*w.shape[0])
        temp[::2] = w
        temp[1::2] = w
        w = temp
    W = xp.diag(w)
    # print(A.T.shape, W.shape, A.shape)
    cov = A.T.dot(W.dot(A))
    return xp.linalg.inv(cov + rcond * xp.diag(cov).max() * xp.eye(A.shape[1])).dot( A.T.dot(W) )

def TikhonovInverse(A, rcond=1e-1):
    U, s, Vt = xp.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(S, beta=-1):
    # S is the sensitivity matrix also known as the Jacobian
    sts = xp.matmul(S.T, S)
    rho = xp.diag(sts)
    alpha2 = rho.max()

    gain_matrix = xp.matmul( xp.linalg.inv( sts + alpha2*10.0**(beta)*xp.eye(sts.shape[0]) ), S.T)
    return gain_matrix

def create_circ_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w//2), int(h//2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
        
    Y, X = xp.ogrid[:h, :w]
    dist_from_center = xp.sqrt((X - center[0] + 1/2)**2 + (Y - center[1] + 1/2)**2)

    mask = dist_from_center <= radius
    return mask

# Creating focal plane masks
def create_annular_focal_plane_mask(sysi, 
                                    inner_radius, outer_radius, 
                                    edge=None,
                                    shift=(0,0), 
                                    rotation=0,
                                    plot=False):
    x = (xp.linspace(-sysi.npsf/2, sysi.npsf/2-1, sysi.npsf) + 1/2)*sysi.psf_pixelscale_lamD
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    mask = (r < outer_radius) * (r > inner_radius)
    if edge is not None: mask *= (x > edge)
    
    mask = _scipy.ndimage.rotate(mask, rotation, reshape=False, order=0)
    mask = _scipy.ndimage.shift(mask, (shift[1], shift[0]), order=0)
    
    if plot:
        imshows.imshow1(mask)
        
    
    return mask

def create_box_focal_plane_mask(sysi, x0, y0, width, height):
    x = (xp.linspace(-sysi.npsf/2, sysi.npsf/2-1, sysi.npsf) + 1/2)*sysi.psf_pixelscale_lamD
    x,y = xp.meshgrid(x,x)
    x0, y0, width, height = (params['x0'], params['y0'], params['w'], params['h'])
    mask = ( abs(x - x0) < width/2 ) * ( abs(y - y0) < height/2 )
    return mask > 0

def create_bowtie_focal_plane_mask(x, y, params, plot=False):
    r = xp.hypot(x, y)
    th = xp.arctan2(y,x) * 180/np.pi
    if 'rotation' in params: th += -params['rotation']
    mask = (r < params['outer_radius']) * (r > params['inner_radius'])
    
    angle = params['angle']
    
    bowtie_mask = (th>-angle/2) * (th<angle/2)
    bowtie_mask += ((th)>(180-angle/2))
    bowtie_mask += ((th)<(-180+angle/2))
    mask *= bowtie_mask
    
    if plot:
        imshows.imshow1(mask)
    
    return mask

def masked_rms(image,mask=None):
    return np.sqrt(np.mean(image[mask.astype(bool)]**2))

def create_random_probes(rms, alpha, dm_mask, fmin=1, fmax=17, nprobes=2, 
                         plot=False,
                         calc_responses=False):
    # randomized probes generated by PSD
    shape = dm_mask.shape
    ndm = shape[0]

    probes = []
    for n in range(nprobes):
        fx = np.fft.rfftfreq(ndm, d=1.0/ndm)
        fy = np.fft.fftfreq(ndm, d=1.0/ndm)
        fxx, fyy = np.meshgrid(fx, fy)
        fr = np.sqrt(fxx**2 + fyy**2)
        spectrum = ( fr**(alpha/2.0) ).astype(complex)
        spectrum[fr <= fmin] = 0
        spectrum[fr >= fmax] = 0
        cvals = np.random.standard_normal(spectrum.shape) + 1j * np.random.standard_normal(spectrum.shape)
        spectrum *= cvals
        probe = np.fft.irfft2(spectrum)
        probe *= dm_mask
        probe = probe.real
        probe /= np.abs(probe).max()
        probes.append(probe)

    # probes = np.asarray(probes)/rms
    
    if plot:
        for i in range(nprobes):
            if calc_responses:
                response = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift( pad_or_crop(probes[i], 4*ndm) ))))
                imshows.imshow2(probes[i], response, pxscl2=1/4)
            else:
                imshows.imshow1(probes[i])
                
    return np.array(probes)

def create_2dm_mode_matrix(dm_modes_1, dm_modes_2=None):
    
    dm_modes_2 = dm_modes_1 if dm_modes_2 is None else dm_modes_2 # assumes same modes on each DM
    
    calib_modes_dm1 = np.concatenate([dm_modes_1, np.zeros_like(dm_modes_2)])
    calib_modes_dm2 = np.concatenate([np.zeros_like(dm_modes_1), dm_modes_2])
#     print(calib_modes_dm1.shape, calib_modes_dm2.shape)
    Mcalib = np.concatenate([calib_modes_dm1, calib_modes_dm2], axis=1)
    
    return Mcalib

def sort_modes_by_rms(calib_modes, response_matrix):
    
    rms_mode_responses = xp.sqrt(xp.mean((response_matrix)**2, axis=0))
    sorted_inds = ensure_np_array(xp.argsort(rms_mode_responses))
    
    sorted_modes = np.empty_like(calib_modes)
    sorted_modes[sorted_inds] = calib_modes
    
    sorted_response_matrix = xp.empty_like(response_matrix)
    sorted_response_matrix = response_matrix[:, sorted_inds]
    
    return sorted_modes, sorted_response_matrix


def fourier_filter_command(command, iwa, owa):
    
    oversamp = 4
    N = int(oversamp*command.shape[0])
    
    command_padded = pad_or_crop(command, N)
    
    command_ft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift( command_padded )))
    
    x = (np.linspace(-N//2, N//2-1, N) + 1/2) * 1/oversamp
    x,y = np.meshgrid(x,x)
    r = np.sqrt(x**2 + y**2)
    mask = (r>iwa) * (r<owa)
    imshows.imshow1(mask)
    command_ft *= mask
    
    filtered_command = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift( command_ft )))
    
    return pad_or_crop(filtered_command, command.shape[0])

def create_hadamard_modes(dm_mask, ndms=1): 
    Nacts = dm_mask.sum().astype(int)
    np2 = 2**int(np.ceil(np.log2(Nacts)))
    hmodes = hadamard(np2)
    
    had_modes = []

    inds = np.where(dm_mask.flatten().astype(int))
    for hmode in hmodes:
        hmode = hmode[:Nacts]
        mode = np.zeros((dm_mask.shape[0]**2))
        mode[inds] = hmode
        had_modes.append(mode)
    had_modes = np.array(had_modes)
    
    if ndms==2:
        had_modes = create_2dm_mode_matrix(had_modes)
    
    return had_modes

def create_fourier_modes(sysi, control_mask, fourier_sampling=0.75, use='both', return_fs=False, ndms=1):
    xfp = (np.linspace(-sysi.npsf/2, sysi.npsf/2-1, sysi.npsf) + 1/2) * sysi.psf_pixelscale_lamD
    fpx, fpy = np.meshgrid(xfp,xfp)
    
    intp = scipy.interpolate.interp2d(xfp, xfp, ensure_np_array(control_mask)) # setup the interpolation function
    
    xpp = np.linspace(-sysi.Nact/2, sysi.Nact/2-1, sysi.Nact) + 1/2
    ppx, ppy = np.meshgrid(xpp,xpp)
    
    fourier_lim = fourier_sampling * int(np.round(xfp.max()/fourier_sampling))
    xfourier = np.arange(-fourier_lim-fourier_sampling/2, fourier_lim+fourier_sampling, fourier_sampling)
    fourier_x, fourier_y = np.meshgrid(xfourier, xfourier) 
    
    # Select the x,y frequencies for the Fourier modes to calibrate the dark hole region
    fourier_grid_mask = ( (intp(xfourier, xfourier) * (((fourier_x!=0) + (fourier_y!=0)) > 0)) > 0 )
    
    fxs = fourier_x.ravel()[fourier_grid_mask.ravel()]
    fys = fourier_y.ravel()[fourier_grid_mask.ravel()]
    sampled_fs = np.vstack((fxs, fys)).T
    
    cos_modes = []
    sin_modes = []
    for f in sampled_fs:
        fx = f[0]/sysi.Nact
        fy = f[1]/sysi.Nact
        cos_modes.append( ( np.cos(2 * np.pi * (fx * ppx + fy * ppy)) * sysi.dm_mask ).flatten() ) 
        sin_modes.append( ( np.sin(2 * np.pi * (fx * ppx + fy * ppy)) * sysi.dm_mask ).flatten() )
        
    if use=='both' or use=='b':
        fourier_modes = cos_modes + sin_modes
    elif use=='cos' or use=='c':
        modes = cos_modes
    elif use=='sin' or use=='s':
        modes = sin_modes
    
    fourier_modes = np.array(fourier_modes)
    
    if ndms==2:
        fourier_modes = create_2dm_mode_matrix(fourier_modes)
    
    if return_fs:
        return fourier_modes, sampled_fs
    else:
        return fourier_modes

def create_fourier_probes(sysi, control_mask, fourier_sampling=0.25, shift=(0,0), nprobes=2,
                          use_weighting=False, plot=False): 
    Nact = sysi.Nact
    xfp = (xp.linspace(-sysi.npsf/2, sysi.npsf/2-1, sysi.npsf) + 1/2) * sysi.psf_pixelscale_lamD
    fpx, fpy = xp.meshgrid(xfp,xfp)
    
    fourier_modes, fs = create_fourier_modes(sysi, control_mask*(fpx>0), 
                                             fourier_sampling=fourier_sampling, 
                                             use='both', ndms=1,
                                             return_fs=True)
    nfs = fourier_modes.shape[0]//2

    probes = np.zeros((nprobes, sysi.Nact, sysi.Nact))
    if use_weighting:
        fmax = np.max(np.sqrt(fs[:,0]**2 + fs[:,1]**2))
        # print(fmax)
        sum_cos = 0
        sum_sin = 0
        for i in range(nfs):
            f = np.sqrt(fs[i][0]**2 + fs[i][1]**2)
            weight = f/fmax
            # print(f,weight)
            sum_cos += weight*fourier_modes[i]
            sum_sin += weight*fourier_modes[i+nfs]
        sum_cos = sum_cos.reshape(Nact, Nact)
        sum_sin = sum_sin.reshape(Nact, Nact)
    else:
        sum_cos = fourier_modes[:nfs].sum(axis=0).reshape(Nact,Nact)
        sum_sin = fourier_modes[nfs:].sum(axis=0).reshape(Nact,Nact)
    
    # nprobes=2 will give one probe that is purely the sum of cos and another that is the sum of sin
    cos_weights = np.linspace(1,0,nprobes)
    sin_weights = np.linspace(0,1,nprobes)
    
    if not isinstance(shift, list):
        shifts = [shift]*nprobes
    else:
        shifts = shift
    for i in range(nprobes):
        probe = cos_weights[i]*sum_cos + sin_weights[i]*sum_sin
        probe = scipy.ndimage.shift(probe, (shifts[i][1], shifts[i][0]))
        probes[i] = probe/np.max(probe)

        if plot: 
            imshows.imshow1(probes[i])

    return probes

def fourier_mode(lambdaD_yx, rms=1, acts_per_D_yx=(48,48), Nact=48, phase=0):
    '''
    Allow linear combinations of sin/cos to rotate through the complex space
    * phase = 0 -> pure cos
    * phase = np.pi/4 -> sqrt(2) [cos + sin]
    * phase = np.pi/2 -> pure sin
    etc.
    '''
    idy, idx = np.indices((Nact, Nact)) - (34-1)/2.
    
    #cfactor = np.cos(phase)
    #sfactor = np.sin(phase)
    prefactor = rms * np.sqrt(2)
    arg = 2*np.pi*(lambdaD_yx[0]/acts_per_D_yx[0]*idy + lambdaD_yx[1]/acts_per_D_yx[1]*idx)
    
    return prefactor * np.cos(arg + phase)

def create_poke_probes(poke_indices, Nact=48, plot=False):
    Nprobes = len(poke_indices)
    probe_modes = np.zeros((Nprobes, Nact, Nact))
    for i in range(Nprobes):
        probe_modes[i, poke_indices[i][1], poke_indices[i][0]] = 1
    if plot:
        fig,ax = plt.subplots(nrows=1, ncols=Nprobes, dpi=125, figsize=(10,4))
        for i in range(Nprobes):
            im = ax[i].imshow(probe_modes[i], cmap='viridis')
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            fig.colorbar(im, cax=cax)
        plt.close()
        display(fig)
        
    return probe_modes

def create_all_poke_modes(dm_mask, ndms=1):
    Nact = dm_mask.shape[0]
    Nacts = int(np.sum(dm_mask))
    poke_modes = np.zeros((Nacts, Nact, Nact))
    count=0
    for i in range(Nact):
        for j in range(Nact):
            if dm_mask[i,j]:
                poke_modes[count, i,j] = 1
                count+=1

    poke_modes = poke_modes[:,:].reshape(Nacts, Nact**2)
    
    if ndms==2:
        poke_modes = create_2dm_mode_matrix(poke_modes)
    
    return poke_modes

def create_sinc_probe(Nacts, amp, probe_radius, probe_phase=0, offset=(0,0), bad_axis='x'):
    print('Generating probe with amplitude={:.3e}, radius={:.1f}, phase={:.3f}, offset=({:.1f},{:.1f}), with discontinuity along '.format(amp, probe_radius, probe_phase, offset[0], offset[1]) + bad_axis + ' axis.')
    
    xacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[0])/Nacts
    yacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[1])/Nacts
    Xacts,Yacts = np.meshgrid(xacts,yacts)
    if bad_axis=='x': 
        fX = 2*probe_radius
        fY = probe_radius
        omegaY = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaY*Yacts + probe_phase)
    elif bad_axis=='y': 
        fX = probe_radius
        fY = 2*probe_radius
        omegaX = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaX*Xacts + probe_phase) 
    if probe_phase == 0:
        f = 2*probe_radius
        probe_commands = amp * np.sinc(f*Xacts)*np.sinc(f*Yacts)
    return probe_commands

def create_sinc_probes(Npairs, Nacts, dm_mask, probe_amplitude, probe_radius=10, probe_offset=(0,0), display=False):
    
    probe_phases = np.linspace(0, np.pi*(Npairs-1)/Npairs, Npairs)
    
    probes = []
    for i in range(Npairs):
        if i%2==0:
            axis = 'x'
        else:
            axis = 'y'
            
        probe = create_sinc_probe(Nacts, probe_amplitude, probe_radius, probe_phases[i], offset=probe_offset, bad_axis=axis)
            
        probes.append(probe*dm_mask)
    
    return np.array(probes)
    
def get_radial_dist(shape, scaleyx=(1.0, 1.0), cenyx=None):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    if cenyx is None:
        cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def get_radial_contrast(im, mask, nbins=50, cenyx=None):
    im = ensure_np_array(im)
    mask = ensure_np_array(mask)
    radial = get_radial_dist(im.shape, cenyx=cenyx)
    bins = np.linspace(0, radial.max(), num=nbins, endpoint=True)
    digrad = np.digitize(radial, bins)
    profile = np.asarray([np.mean(im[ (digrad == i) & mask]) for i in np.unique(digrad)])
    return bins, profile
    
def plot_radial_contrast(im, mask, pixelscale, nbins=30, cenyx=None, 
                         xlims=None, ylims=None, title='Contrast vs Radial Position'):
    bins, contrast = get_radial_contrast(im, mask, nbins=nbins, cenyx=cenyx)
    r = bins * pixelscale

    fig,ax = plt.subplots(nrows=1, ncols=1, dpi=125, figsize=(6,4))
    ax.semilogy(r,contrast)
    ax.set_xlabel('Radial Position [$\lambda/D$]')
    ax.set_ylabel('Contrast')
    ax.set_title(title)
    ax.grid()
    if xlims is not None: ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None: ax.set_ylim(ylims[0], ylims[1])
    plt.close()
    display(fig)

def save_fits(fpath, data, header=None, ow=True, quiet=False):
    data = ensure_np_array(data)
    if header is not None:
        keys = list(header.keys())
        hdr = fits.Header()
        for i in range(len(header)):
            hdr[keys[i]] = header[keys[i]]
    else: 
        hdr = None
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(fpath), overwrite=ow) 
    if not quiet: print('Saved data to: ', str(fpath))

# functions for saving python objects
import pickle
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data    