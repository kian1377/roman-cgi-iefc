from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows

import numpy as np
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output
import time
import copy

def run_pwp_bp(sysi, 
               control_mask, 
               probes,
               use='J', jacobian=None, model=None, 
               plot=False,
               plot_est=False):
    """ 
    This method of PWP will use the supplied probe commands to estimate the electric field
    within the pixels specified by the boolean control mask. 

    Parameters
    ----------
    sysi : object
        the system model or testbed interface to use for image capture
    control_mask : xp.ndarray
        boolean array of focal plane pixels to be estimated
    probes : np.ndarray
        3D array of probes to be used for estimation
    use : str, optional
        whether to use a jacobian or the direct model to perform estimation, by default 'J'
    jacobian : xp.ndarray, optional
        the Jacobian to use if use='J', by default None
    model : object, optional
        the model to use, by default None
    plot : bool, optional
        plot all stages of the estimation algorithm, by default False
    plot_est : bool, optional
        plot the estimated field and ignore the other plots, by default False

    Returns
    -------
    xp.ndarray
        2D array containing the electric field estimate within the control mask
    """
    Nmask = int(control_mask.sum())

    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm(amp*probe)
            psf = sysi.snap()
                
            if amp==-1: 
                In.append(psf)
            else: 
                Ip.append(psf)
                
            sysi.add_dm(-amp*probe) # remove probe from DM
            
        if plot:
            imshows.imshow3(Ip[i], In[i], Ip[i]-In[i], lognorm1=True, lognorm2=True, pxscl=sysi.psf_pixelscale_lamD)
            
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(len(probes)):
        if (use=='jacobian' or use.lower()=='j') and jacobian is not None:
            probe = xp.array(probes[i])
            E_probe = jacobian.dot(xp.array(probe[sysi.dm_mask.astype(bool)]))
            E_probe = E_probe[::2] + 1j*E_probe[1::2]
        elif (use=='model' or use=='m') and model is not None:
            if i==0: 
                E_full = model.calc_psf()[control_mask]
                
            model.add_dm(probes[i])
            E_full_probe = model.calc_psf()[control_mask]
            model.add_dm(-probes[i])
            
            E_probe = E_full_probe - E_full
            # print(type(E_probe))
            
        if plot:
            E_probe_2d = xp.zeros((sysi.npsf,sysi.npsf), dtype=xp.complex128)
            # print(xp)
            # print(type(E_probe_2d), type(dark_mask))
            xp.place(E_probe_2d, mask=control_mask, vals=E_probe)
            imshows.imshow2(xp.abs(E_probe_2d), xp.angle(E_probe_2d),
                            f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$')
            
        E_probes[i, ::2] = E_probe.real
        E_probes[i, 1::2] = E_probe.imag

        I_diff[i:(i+1), :] = (Ip[i] - In[i])[control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        M = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Minv = xp.linalg.pinv(M.T@M, 1e-2)@M.T
    
        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((sysi.npsf,sysi.npsf), dtype=xp.complex128)
    xp.place(E_est_2d, mask=control_mask, vals=E_est)
    
    if plot or plot_est:
        imshows.imshow2(xp.abs(E_est_2d)**2, xp.angle(E_est_2d), 
                        'Estimated Intensity', 'Estimated Phase',
                        lognorm1=True, pxscl=sysi.psf_pixelscale_lamD)
    return E_est_2d


