import numpy as np
import cupy as cp
import poppy
if poppy.accel_math._USE_CUPY:
    from cupyx.scipy.sparse import linalg as sLA
else:
    from scipy.sparse import linalg as sLA

from scipy import interpolate, ndimage
import time
from astropy.io import fits
from matplotlib.patches import Circle, Rectangle
import pickle

import matplotlib.pyplot as plt

from cgi_phasec_poppy import misc

# Create control matrix
def WeightedLeastSquares(A, W, rcond=1e-15):
    cov = A.T.dot(W.dot(A))
    return np.linalg.inv(cov + rcond * np.diag(cov).max() * np.eye(A.shape[1])).dot( A.T.dot(W) )

def TikhonovInverse(A, rcond=1e-15):
    if isinstance(A, cp.ndarray):
        U, s, Vt = cp.linalg.svd(A, full_matrices=False)
    else:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(S, beta=-1):
    # S is the sensitivity matrix, also known as the Jacobian
    sts = cp.matmul(S.conj().T, S)
    rho = cp.diag(sts)
    alpha2 = rho.max()
    
    gain_matrix = cp.matmul( cp.linalg.inv( sts + alpha2*10.0**(beta)*cp.eye(sts.shape[0]) ), S.T)
    return gain_matrix


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
    inner_radius, outer_radius, edge_position, rot = (params['inner_radius'], params['outer_radius'], 
                                                      params['edge_position'], params['rotation'])
    
    r = np.hypot(x, y)
    mask = (r < outer_radius) * (r > inner_radius)
    if params['full']==False:
        mask *= (x > edge_position)
    else:
        mask *= (abs(x) > edge_position)
    
    mask = ndimage.rotate(mask, rot, reshape=False, order=0)
    
    return mask

def create_box_focal_plane_mask(x, y, params):
    x0, y0, width, height = (params['x0'], params['y0'], params['w'], params['h'])
    mask = ( abs(x - x0) < width/2 ) * ( abs(y - y0) < height/2 )
    return mask > 0


def create_bowtie_mask(x, y, params):
    inner_radius, outer_radius, side = (params['inner_radius'], params['outer_radius'], params['side'])
    
    r = np.hypot(x, y)
    th = np.arctan2(x,y)*180/np.pi + 180
    
    mask = (r < outer_radius) * (r > inner_radius)
    
    if side=='left' or side=='l':
        mask *= (th>57.5) * (th<57.5+65)
    elif side=='right' or side=='r':
        mask *= (th<(360-57.5)) * (th>(360-57.5-65))
    if side=='both' or side=='b':
        mask *= (th>57.5) * (th<57.5+65) + (th<(360-57.5)) * (th>(360-57.5-65))
    
    return mask
    
    
    
def sms(U, s, alpha2, electric_field, N_DH, Imax_unocc, itr): 
    # jac: system jacobian
    # electric_field: the electric field acquired by estimation or from the model
    
    E_ri = U.conj().T.dot(electric_field)
    SMS = cp.abs(E_ri)**2/(N_DH/2*Imax_unocc)
#     print(SMS.shape)
    
    Nbox = 31
    box = cp.ones(Nbox)/Nbox
    SMS_smooth = cp.convolve(SMS, box, mode='same')
    
    x = (s**2/alpha2).get()
    y = SMS_smooth.get()
    
#     print(I_ri_smooth)
#     contrast = np.trapz(y, x)
#     print(contrast)
    
    xmax = np.max(x)
    xmin = 1e-10 
    ymax = 1
    ymin = 1e-14
    
    fig = plt.figure(dpi=125)
    plt.loglog(x, y)
    plt.title('Singular Mode Spectrum: Iteration {:d}'.format(itr))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r'$(s_{i}/\alpha)^2$: Square of Normalized Singular Values')
    plt.ylabel('SMS')
    plt.grid()
    plt.close()
    display(fig)
    
    return fig
    
    
    
    
    
    
    
