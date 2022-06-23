import numpy as np
import cupy as cp
import poppy
if poppy.accel_math._USE_CUPY:
    from cupyx.scipy.sparse import linalg as sLA
else:
    from scipy.sparse import linalg as sLA

from scipy import interpolate
import time
from astropy.io import fits
from matplotlib.patches import Circle, Rectangle
import pickle

import misc

# Create control matrix
def WeightedLeastSquares(A, W, rcond=1e-15):
    cov = A.T.dot(W.dot(A))
    if isinstance(A, cp.ndarray) and isinstance(W, cp.ndarray):
        return cp.linalg.inv(cov + rcond * cp.diag(cov).max() * cp.eye(A.shape[1])).dot( A.T.dot(W) )
    else:
        return np.linalg.inv(cov + rcond * np.diag(cov).max() * np.eye(A.shape[1])).dot( A.T.dot(W) )

def TikhonovInverse(A, rcond=1e-15):
    if isinstance(A, cp.ndarray):
        U, s, Vt = cp.linalg.svd(A, full_matrices=False)
    else:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def create_circ_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w//2), int(h//2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
        
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0] + 1/2)**2 + (Y - center[1] + 1/2)**2)

    mask = dist_from_center <= radius
    return mask

# Creating focal plane masks
def create_annular_focal_plane_mask(x, y, params):
    inner_radius, outer_radius, edge_position, direction = (params['inner_radius'], params['outer_radius'], 
                                                            params['edge_position'], params['direction'])
    r = np.hypot(x, y)
    mask = (r < outer_radius) * (r > inner_radius)
    if direction == '+x': mask *= (x > edge_position)
    elif direction == '-x': mask *= (x < -edge_position)
    elif direction == '+y': mask *= (y > edge_position)
    elif direction == '-y': mask *= (y < -edge_position)
        
    return mask

def create_box_focal_plane_mask(x, y, params):
    x0, y0, width, height = (params['x0'], params['y0'], params['w'], params['h'])
    mask = ( abs(x - x0) < width/2 ) * ( abs(y - y0) < height/2 )
    return mask > 0

# Creating patches for display purposes
def create_rect_patch(rect_params):
    rect_patch = Rectangle((rect_params['x0']-rect_params['w']/2, rect_params['y0']-rect_params['h']/2), 
                           rect_params['w'], rect_params['h'], color='c', fill=False)
    return rect_patch

def create_circ_patches(circ_params):
    circ_patches = [Circle( (0,0), circ_params['inner_radius'], color='c', fill=False), 
                  Circle( (0,0), circ_params['outer_radius'], color='c', fill=False)]
    return circ_patches

def create_fourier_modes(xfp, mask, Nact=48, use_both=True, circular_mask=True):
    print("Creating Fourier modes: ", mask.shape)
    intp = interpolate.interp2d(xfp, xfp, mask)
    
    # This creates the grid and frequencies
    xs = np.linspace(-0.5, 0.5, Nact) * (Nact-1)
    x, y = np.meshgrid(xs, xs)
    x = x.ravel()
    y = y.ravel()
    
    # Create the fourier frequencies. An odd number of modes is preferred for symmetry reasons.
    if Nact % 2 == 0: 
        fxs = np.fft.fftshift( np.fft.fftfreq(Nact+1) )
    else:
        fxs = np.fft.fftshift( np.fft.fftfreq(Nact) )
        
    fx, fy = np.meshgrid(fxs, fxs)
    
    # Select all Fourier modes of interest based on the dark hole mask and remove the piston mode
    mask2 = intp(fxs * Nact, fxs * Nact) * (((fx!=0) + (fy!=0)) > 0) > 0
    
    fx = fx.ravel()[mask2.ravel()]
    fy = fy.ravel()[mask2.ravel()]
    
    # The modes can rewritten to a single (np.outer(x, fx) + np.outer(y, fy))
    if use_both:
        M1 = [np.cos(2 * np.pi * (fi[0] * x + fi[1] * y)) for fi in zip(fx, fy)]
        M2 = [np.sin(2 * np.pi * (fi[0] * x + fi[1] * y)) for fi in zip(fx, fy)]
        
        # Normalize the modes
        M = np.array(M1+M2)
    else:
        M = np.array([np.sin(2 * np.pi * (fi[0] * x + fi[1] * y)) for fi in zip(fx, fy)])
        
#     M /= np.std(M, axis=1, keepdims=True)
    
    if circular_mask: 
        circ = np.ones((Nact,Nact))
        r = np.sqrt(x.reshape((Nact,Nact))**2 + y.reshape((Nact,Nact))**2)
        circ[r>(Nact)/2] = 0
        M[:] *= circ.flatten()
        
    M /= np.std(M, axis=1, keepdims=True)
        
    return M, fx, fy

def create_probe_poke_modes(Nact, indx0, indy0, indx1, indy1):
    probe_modes = np.zeros((2, Nact, Nact))
    probe_modes[0, indy0, indx0] = 1
    probe_modes[1, indy1, indx1] = 1
    probe_modes = probe_modes.reshape((2, -1))
    return probe_modes


