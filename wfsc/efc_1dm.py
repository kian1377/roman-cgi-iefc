import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output
import time
import copy
from importlib import reload

from . import pwp_1dm as pwp
from . import utils
reload(pwp)
reload(utils)

from cgi_phasec_poppy import misc

def build_jacobian(sysi, epsilon, dark_mask, display=False):
    start = time.time()
    print('Building Jacobian.')
    
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    for i, mode in enumerate(modes):
        if dm_mask[i]==1:
            response = 0
            for amp in amps:
                mode = mode.reshape(sysi.Nact,sysi.Nact)
                
                sysi.add_dm1(amp*mode)
                
                psf = sysi.calc_psf()
                wavefront = psf.wavefront
                response += amp*wavefront/np.var(amps)
                
                sysi.add_dm1(-amp*mode)

            if display:
                misc.myimshow2(cp.abs(response), cp.angle(response))
            
            response = response.flatten().get()[dark_mask.flatten()]
            
        else:
            response = np.zeros((sysi.npsf, sysi.npsf), dtype=np.complex128).flatten()[dark_mask.flatten()]

        responses.append(np.concatenate((response.real, response.imag)))
        
        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1,num_modes,time.time()-start))
        
    jacobian = np.array(responses).T
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return jacobian

# def run_pwp(sysi, probes, jacobian, dark_mask, use_noise=False, display=False):
#     nmask = dark_mask.sum()
    
#     dm_ref = sysi.get_dm1()
#     amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
#     Ip = []
#     In = []
#     for i,probe in enumerate(probes):
#         for amp in amps:
#             sysi.add_dm1(amp*probe)
#             psf = sysi.snap()
                
#             if amp==-1: 
#                 In.append(psf)
#             else: 
#                 Ip.append(psf)
                
#             sysi.add_dm1(-amp*probe) # remove probe from DM
            
#         if display:
#             misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
#                            'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
#                            'Intensity Difference',
#                            lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
#                           )
        
#     E_probes = np.zeros((probes.shape[0], 2*nmask))
#     I_diff = np.zeros((probes.shape[0], nmask))
#     for i in range(len(probes)):
#         E_probe = jacobian.dot(np.array(probes[i].flatten())) # Use jacobian to model probe E-field at the focal plane
#         E_probe = E_probe[:nmask] + 1j*E_probe[nmask:]

#         E_probes[i, :nmask] = E_probe.real
#         E_probes[i, nmask:] = E_probe.imag

#         I_diff[i:(i+1), :] = (Ip[i] - In[i])[dark_mask]
    
#     # Use batch process to estimate each pixel individually
#     E_est = np.zeros((nmask,), dtype=cp.complex128)
#     for i in range(nmask):
#         delI = I_diff[:, i]
#         M = 2*np.array([E_probes[:,i], E_probes[:,i+nmask]]).T
#         Minv = np.linalg.inv(M)
        
#         est = Minv.dot(delI)

#         E_est[i] = est[0] + 1j*est[1]
        
#     E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
#     np.place(E_est_2d, mask=dark_mask, vals=E_est)
    
#     return E_est_2d

def run_pwp(sysi, probes, jacobian, dark_mask, use_noise=False, display=False):
    nmask = dark_mask.sum()
    nprobes = probes.shape[0]
    
    dm_ref = sysi.get_dm1()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm1(amp*probe)
            
            im = sysi.snap()
                
            if amp==-1: 
                In.append(im)
            else: 
                Ip.append(im)
                
            sysi.add_dm1(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
                          )

    E_probes = np.zeros((2*nmask*nprobes,))
    I_diff = np.zeros((nmask*nprobes,))
    for i in range(nprobes):
        I_diff[ i*nmask : (i+1)*nmask ] = (Ip[i] - In[i])[dark_mask]
        
        E_probe = jacobian.dot(np.array(probes[i].flatten()))
        E_probes[ i*2*nmask : (i+1)*2*nmask ] = E_probe
        
        E_probe_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
        np.place(E_probe_2d, mask=dark_mask, 
                 vals=E_probes[i*nmask:i*nmask + 2*nmask][:nmask] + 1j*E_probes[i*nmask:i*nmask + 2*nmask][nmask:])
        misc.myimshow2(np.abs(E_probe_2d), np.angle(E_probe_2d))
        
    B = np.diag(np.ones((nmask,2*nmask))[0], k=0)[:nmask,:2*nmask] + np.diag(np.ones((nmask,2*nmask))[0], k=nmask)[:nmask,:2*nmask]
    misc.myimshow(B)
    print('B.shape', B.shape)
    
    for i in range(nprobes):
        h = B @ np.diag( E_probes[ i*2*nmask : (i+1)*2*nmask ] )
        Hinv = h if i==0 else np.vstack((Hinv,h))
    
    print('Hinv.shape', Hinv.shape)
    
    H = np.linalg.inv(Hinv.T@Hinv)@Hinv.T
    print('H.shape', H.shape)
    
    E_est = H.dot(I_diff)
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    return E_est_2d



def run_efc_pwp(sysi, 
                probes,
                    jac, 
                    reg_fun,
                    reg_conds,
                    dark_mask, 
                    Imax_unocc,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    display_all=False, 
                    display_current=True,
                    plot_sms=True):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    jac_cp = cp.array(jac) if isinstance(jac, np.ndarray) else jac
    
    U, s, V = cp.linalg.svd(jac_cp, full_matrices=False)
    alpha2 = cp.max( cp.diag( cp.real( jac_cp.conj().T @ jac_cp ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    N_DH = dark_mask.sum()
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        if i==0 or i in reg_conds[0]:
            reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
            reg_cond = reg_conds[1, reg_cond_ind]
            print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
            efc_matrix = reg_fun(jac_cp, reg_cond).get()
        
        sysi.set_dm1(dm_ref + dm_command)
        E_est = pwp.run_pwp_bp(sysi, probes, dark_mask, use='j', jacobian=jac)
        I_exact = sysi.snap()
        
        commands.append(sysi.get_dm1())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))
        
        efield_ri = np.concatenate( (E_est[dark_mask].real, E_est[dark_mask].imag) )
        del_dm = efc_matrix.dot(efield_ri).reshape(sysi.Nact,sysi.Nact)
        
        dm_command -= efc_loop_gain * del_dm.reshape(sysi.Nact,sysi.Nact)
        
        if display_current or display_all:
            if not display_all: clear_output(wait=True)
                
            fig,ax = misc.myimshow3(commands[i], np.abs(E_est)**2, I_exact, 
                                        'DM', 'Estimated Intensity', 'Image: Iteration {:d}'.format(i),
                                        lognorm2=True, vmin2=1e-12, lognorm3=True, vmin3=1e-12,
                                        return_fig=True, display_fig=True)
            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, cp.array(efield_ri), N_DH, Imax_unocc, i)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images

def run_efc_perfect(sysi, 
                    jac, 
                    reg_fun,
                    reg_conds,
                    dark_mask, 
                    Imax_unocc,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    display_all=False, 
                    display_current=True,
                    plot_sms=True):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')    
    commands = []
    efields = []
    
    start = time.time()
    
    jac = cp.array(jac) if isinstance(jac, np.ndarray) else jac
    
    U, s, V = cp.linalg.svd(jac, full_matrices=False)
    alpha2 = cp.max( cp.diag( cp.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    N_DH = dark_mask.sum()
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    print()
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        if i==0 or i in reg_conds[0]:
            reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
            reg_cond = reg_conds[1, reg_cond_ind]
            print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
            efc_matrix = reg_fun(jac, reg_cond).get()
        
        sysi.set_dm1(dm_ref + dm_command) 
        
        psf = sysi.calc_psf()
        electric_field = psf.wavefront.get()
        
        commands.append(sysi.get_dm1())
        efields.append(copy.copy(electric_field))
        
        efield_ri = np.concatenate( (electric_field[dark_mask].real, electric_field[dark_mask].imag) )
        del_dm = efc_matrix.dot(efield_ri).reshape(sysi.Nact,sysi.Nact)
        
        dm_command -= efc_loop_gain * del_dm
        
        if display_current or display_all:
            if not display_all: clear_output(wait=True)
                
            fig,ax = misc.myimshow2(commands[i], np.abs(electric_field)**2, 
                                        'DM', 'Image: Iteration {:d}'.format(i),
                                        lognorm2=True, vmin2=1e-12,
                                        return_fig=True, display_fig=True)
            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, cp.array(efield_ri), N_DH, Imax_unocc, i)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields



