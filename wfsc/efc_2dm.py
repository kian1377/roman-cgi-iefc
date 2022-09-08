import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output, display
import time
import copy
from importlib import reload

from . import utils
reload(utils)

from cgi_phasec_poppy import misc

def build_jacobian(sysi, epsilon, dark_mask, display=False):
    start = time.time()
    print('Building Jacobian.')
    
    responses_1 = []
    responses_2 = []
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    for i, mode in enumerate(modes):
        if dm_mask[i]==1:
            response1 = 0
            response2 = 0
            for amp in amps:
                mode = mode.reshape(sysi.Nact,sysi.Nact)
                
                sysi.add_dm1(amp*mode)
                psf = sysi.calc_psf()
                wavefront = psf.wavefront
                response1 += amp*wavefront/np.var(amps)
                sysi.add_dm1(-amp*mode)
                
                sysi.add_dm2(amp*mode)
                psf = sysi.calc_psf()
                wavefront = psf.wavefront
                response2 += amp*wavefront/np.var(amps)
                sysi.add_dm2(-amp*mode)

            if display:
                misc.myimshow2(cp.abs(response), cp.angle(response))
            
            response1 = response1.flatten().get()[dark_mask]
            response2 = response2.flatten().get()[dark_mask]
        else:
            response1 = np.zeros((sysi.npsf, sysi.npsf), dtype=np.complex128).flatten()[dark_mask.flatten()]
            response2 = np.zeros((sysi.npsf, sysi.npsf), dtype=np.complex128).flatten()[dark_mask.flatten()]

        responses_1.append(np.concatenate((response1.real, response1.imag)))
        responses_2.append(np.concatenate((response2.real, response2.imag)))
        
        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1,num_modes,time.time()-start))
        
    responses_1 = np.array(responses_1)
    responses_2 = np.array(responses_2)
    print(responses_1.shape, responses_2.shape)
    
    responses = np.concatenate( ( responses_1, responses_2 ), axis=0 )
    print(responses.shape)
    jacobian = responses.T
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return jacobian

def run_pwp(sysi, probes, jacobian, dark_mask, reg_cond=1e-2, use_noise=False, display=False):
    nmask = dark_mask.sum()
    
    dm_ref = sysi.get_dm1()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip_1 = []
    In_1 = []
    Ip_2 = []
    In_2 = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm1(amp*probe)
            image = sysi.snap() 
            if amp==-1: 
                In_1.append(image)
            else: 
                Ip_1.append(image)
            sysi.add_dm1(-amp*probe) # remove probe from DM
            
            sysi.add_dm2(amp*probe)
            image = sysi.snap()
            if amp==-1: 
                In_2.append(image)
            else: 
                Ip_2.append(image)
            sysi.add_dm2(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip_1[i], In_1[i], Ip_1[i]-In_1[i],
                           'DM1: Probe {:d} Positive Image'.format(i+1), 'DM1: Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, 
                          )
            misc.myimshow3(Ip_2[i], In_2[i], Ip_2[i]-In_2[i],
                           'DM2: Probe {:d} Positive Image'.format(i+1), 'DM2: Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, 
                          )
        
    E_probes_1 = np.zeros((probes.shape[0], 2*nmask))
    I_diff_1 = np.zeros((probes.shape[0], nmask))
    E_probes_2 = np.zeros((probes.shape[0], 2*nmask))
    I_diff_2 = np.zeros((probes.shape[0], nmask))
    for i in range(len(probes)):
        E_probe_1 = jacobian[:,:sysi.Nact**2].dot(np.array(probes[i].flatten())) 
        E_probe_2 = jacobian[:,sysi.Nact**2:].dot(np.array(probes[i].flatten())) 
        
        E_probe_1 = E_probe_1[:nmask] + 1j*E_probe_1[nmask:]
        E_probe_2 = E_probe_2[:nmask] + 1j*E_probe_2[nmask:]

        E_probes_1[i, :nmask] = E_probe_1.real
        E_probes_1[i, nmask:] = E_probe_1.imag
        E_probes_2[i, :nmask] = E_probe_2.real
        E_probes_2[i, nmask:] = E_probe_2.imag

        I_diff_1[i:(i+1), :] = (Ip_1[i] - In_1[i])[dark_mask]
        I_diff_2[i:(i+1), :] = (Ip_2[i] - In_2[i])[dark_mask]
    print(I_diff_1.shape)
    I_diff = np.concatenate( (I_diff_1, I_diff_2), axis=0)
    E_probes = np.concatenate( (E_probes_1, E_probes_2), axis=0)
    
    # Use batch process to estimate each pixel individually
    E_est = np.zeros((nmask,), dtype=cp.complex128)
    for i in range(nmask):
        delI = I_diff[:, i]
        M = 2*np.array([E_probes[:,i], E_probes[:,i+nmask]]).T
        Minv = utils.TikhonovInverse(M, reg_cond)

        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    return E_est_2d

def run_efc_pwp(sysi, efc_matrix, jac, probes, dark_mask, efc_loop_gain=0.5, iterations=5, display=False):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    for i in range(iterations):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        sysi.set_dm1(dm_ref + dm_command)
        E_est = run_pwp(sysi, probes, jac, dark_mask)
        I_exact = sysi.snap()
        
        commands.append(sysi.get_dm1())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))
        
        if display:
            misc.myimshow2(dm1_commands[i], dm2_commands[i], 'DM1', 'DM2')
            misc.myimshow2(E_est[i], I_exact[i], 'DM1', 'DM2')
        
        x = np.concatenate( (E_est[dark_mask].imag, E_est[dark_mask].real) )
        y = efc_matrix.dot(x)
        
        dm_command -= efc_loop_gain * y.reshape(sysi.Nact,sysi.Nact)
        
        
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
    dm1_commands = []
    dm2_commands = []
    efields = []
    
    start = time.time()
    
    jac = cp.array(jac) if isinstance(jac, np.ndarray) else jac
    
    U, s, V = cp.linalg.svd(jac, full_matrices=False)
    alpha2 = cp.max( cp.diag( cp.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    N_DH = dark_mask.sum()
    
    dm1_ref = sysi.get_dm1()
    dm2_ref = sysi.get_dm2()
    
    dm1_command = 0.0
    dm2_command = 0.0
    print()
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
            
        if i==0 or i in reg_conds[0]:
            reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
            reg_cond = reg_conds[1, reg_cond_ind]
            print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
            efc_matrix = reg_fun(jac, reg_cond).get()
        
        sysi.set_dm1(dm1_ref + dm1_command) 
        sysi.set_dm2(dm2_ref + dm2_command) 
        psf = sysi.calc_psf()
        electric_field = psf.wavefront.get()
        
        # evaluate contrast
        intensity = np.abs(electric_field)**2 * dark_mask
        C = intensity.sum()
        
        dm1_commands.append(sysi.get_dm1())
        dm2_commands.append(sysi.get_dm2())
        efields.append(copy.copy(electric_field))
        
        if not display_all: 
            clear_output(wait=True)
            time.sleep(0.25)
            
        if display_current or display_all:
            misc.myimshow3(dm1_commands[i], dm2_commands[i], np.abs(electric_field)**2, 
                           'DM1', 'DM2', 'Image: Iteration {:d}'.format(i),
                           lognorm3=True, vmin3=1e-12,
                           return_fig=True)
        
        efield_ri = np.concatenate( (electric_field[dark_mask].real, electric_field[dark_mask].imag) )
        del_dms = efc_matrix.dot(efield_ri)
        
        dm1_command -= efc_loop_gain * del_dms[:sysi.Nact**2].reshape(sysi.Nact,sysi.Nact)
        dm2_command -= efc_loop_gain * del_dms[sysi.Nact**2:].reshape(sysi.Nact,sysi.Nact)
        
        if plot_sms:
            sms(U, s, alpha2, cp.array(efield_ri), N_DH, Imax_unocc, i)
            
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return dm1_commands, dm2_commands, efields

def sms(U, s, alpha2, electric_field, N_DH, Imax_unocc, itr): 
    # jac: system jacobian
    # electric_field: the electric field acquired by estimation or from the model
#     print(alpha2)
#     print(s.shape, U.shape, U.conj().T.shape)
#     print(electric_field.shape)
    
#     E_ri = U.conj().T @ electric_field
#     I_ri = cp.abs(E_ri)**2
#     print(I_ri.shape)

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
    display(fig)
    plt.close()

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
    
    if display:
        if Npairs==2:
            misc.myimshow2(probes[0], probes[1])
        elif Npairs==3:
            misc.myimshow3(probes[0], probes[1], probes[2])
    
    return np.array(probes)



