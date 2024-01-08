from math_module import xp, _scipy, ensure_np_array
import utils
import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

from pathlib import Path
# iefc_data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')
iefc_data_dir = Path('/home/kianmilani/Projects/roman-cgi-iefc-data')

def build_jacobian(sysi, 
                   calibration_amplitude, calibration_modes, 
                   control_mask, 
                   plot_responses=False,
                  ):
    start = time.time()
    
    Nmodes = calibration_modes.shape[0]
    Nmask = int(control_mask.sum())
    Nact = sysi.Nact

    response_matrix = xp.zeros((2*Nmask, Nmodes), dtype=xp.float64)
    print('Calculating Jacobian: ')
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]:
            # reshape calibration mode into the DM1 and DM2 components
            dm1_mode = calibration_mode[:sysi.Nact**2].reshape(sysi.Nact, sysi.Nact)
            dm2_mode = calibration_mode[sysi.Nact**2:].reshape(sysi.Nact, sysi.Nact)
            
            # Add the mode to the DMs
            sysi.add_dm1(s * calibration_amplitude * dm1_mode)
            sysi.add_dm2(s * calibration_amplitude * dm2_mode)
            
            # Compute reponse with difference images of probes
            efield = sysi.calc_psf()
            response += s * efield / (2 * calibration_amplitude)
            
            # Remove the mode form the DMs
            sysi.add_dm1(-s * calibration_amplitude * dm1_mode) # remove the mode
            sysi.add_dm2(-s * calibration_amplitude * dm2_mode) 

        response_matrix[::2,ci] = response[control_mask].real
        response_matrix[1::2,ci] = response[control_mask].imag

        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(ci+1, Nmodes, time.time()-start), end='')
        print("\r", end="")
    
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    if plot_responses:
        responses = response_matrix[::2] + 1j*response_matrix[1::2]
        dm_rms = xp.sqrt(xp.mean(xp.abs(responses.dot(xp.array(calibration_modes)))**2, axis=0))
        dm1_rms = dm_rms[:sysi.Nact**2].reshape(sysi.Nact, sysi.Nact)
        dm2_rms = dm_rms[sysi.Nact**2:].reshape(sysi.Nact, sysi.Nact)
        imshows.imshow2(dm1_rms, dm2_rms, 
                        'DM1 RMS Actuator Responses', 'DM2 RMS Actuator Responses')

    return response_matrix

def run_efc_perfect(sysi, 
                    # control_matrix,
                    response_matrix,
                    reg_fun, reg_conds,
                    calibration_modes,
                    control_mask, 
                    est_fun=None, 
                    est_params=None, 
                    nonlin_model=None,
                    Imax_unocc=1,
                    loop_gain=0.5, 
                    leakage=0.0,
                    iterations=5, 
                    plot_all=False, 
                    plot_current=True,
                    plot_sms=False,
                    plot_radial_contrast=False,
                    old_images=None,
                    old_dm1_commands=None,
                    old_dm2_commands=None,
                    old_regs =None, 
                    ):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')    
    # commands = np.zeros((iterations, sysi.Nact, sysi.Nact), dtype=np.float64)
    # efields = xp.zeros((iterations, sysi.npsf, sysi.npsf), dtype=xp.complex128)
    # images = xp.zeros((iterations, sysi.npsf, sysi.npsf), dtype=xp.float64)
    start = time.time()
    
    # U, s, V = xp.linalg.svd(jac, full_matrices=False)
    # alpha2 = xp.max( xp.diag( xp.real( jac.conj().T @ jac ) ) )
    # print('Max singular value squared:\t', s.max()**2)
    # print('alpha^2:\t\t\t', alpha2) 
    
    # calibration_modes = xp.array(calibration_modes)

    Nact = sysi.Nact
    Nmask = int(control_mask.sum())
    
    # The metric
    # efields = []
    metric_images = []
    dm1_commands = []
    dm2_commands = []
    regs = []

    dm1_ref = sysi.get_dm1()
    dm2_ref = sysi.get_dm2()
    command = 0.0
    dm1_command = 0.0
    dm2_command = 0.0

    if old_images is None:
        starting_iteration = 0
    else:
        starting_iteration = len(old_images) - 1

    for i in range(iterations):
        print(f'\tRunning iteration {i+1+starting_iteration}/{iterations+starting_iteration}.')
        
        if est_fun is None:
            electric_field = sysi.calc_psf() # no PWP, just use model
        else:
            electric_field = est_fun(sysi, **est_params)
        efield_ri = xp.zeros(2*Nmask)

        efield_ri[::2] = electric_field[control_mask].real
        efield_ri[1::2] = electric_field[control_mask].imag

        # modal_coefficients = -control_matrix.dot(efield_ri)
        # command = (1.0-leakage)*command + loop_gain*modal_coefficients
        
        # # Reconstruct the full phase from the Fourier modes
        # act_commands = calibration_modes.T.dot(utils.ensure_np_array(command))
        # dm1_command = act_commands[:sysi.Nact**2].reshape(sysi.Nact,sysi.Nact)
        # dm2_command = act_commands[sysi.Nact**2:].reshape(sysi.Nact,sysi.Nact)

        # # Set the current DM state
        # sysi.set_dm1(dm1_ref + dm1_command)
        # sysi.set_dm2(dm2_ref + dm2_command)
        
        # # Take an image to estimate the metrics
        # # electric_field = sysi.calc_psf()
        # # image = xp.abs(electric_field)**2
        # image = sysi.snap()

        # # efields.append([copy.copy(electric_field)])
        # metric_images.append(copy.copy(image))
        # dm1_commands.append(sysi.get_dm1())
        # dm2_commands.append(sysi.get_dm2())
        
        # mean_ni = xp.mean(image.ravel()[control_mask.ravel()])
        # print(f'\tMean NI of this iteration: {mean_ni:.3e}')

        if nonlin_model is None or np.isscalar(reg_conds):
            control_matrix = reg_fun(response_matrix, reg_conds)
            modal_coefficients = -control_matrix.dot( efield_ri)
            command = (1.0-leakage)*command + loop_gain*modal_coefficients

            act_commands = calibration_modes.T.dot(utils.ensure_np_array(command))
            dm1_command = act_commands[:sysi.Nact**2].reshape(sysi.Nact,sysi.Nact)
            dm2_command = act_commands[sysi.Nact**2:].reshape(sysi.Nact,sysi.Nact)

            best_dm1_command = dm1_ref + dm1_command
            best_dm2_command = dm2_ref + dm2_command
            sysi.set_dm1(best_dm1_command)
            sysi.set_dm2(best_dm2_command)
            best_image = sysi.snap()

            mean_ni = xp.mean(best_image.ravel()[control_mask.ravel()])

            regs.append(reg_conds)
        else:
            print('\t\tUsing nonlinear coronagraph model to predict best regularization parameter')
            print(f'\t\tRegularization values: ', reg_conds)

            mean_nis = xp.zeros(len(reg_conds))
            del_dm1_per_reg = np.zeros((len(reg_conds), sysi.Nact, sysi.Nact))
            del_dm2_per_reg= np.zeros((len(reg_conds), sysi.Nact, sysi.Nact))
            image_per_reg = xp.zeros((len(reg_conds), sysi.npsf, sysi.npsf))
            command_per_reg = xp.zeros((len(reg_conds), response_matrix.shape[1]))
            for j in range(len(reg_conds)): # compute the control matrix for a range of regularization parameters and use the best one
                # print(j, reg_conds[j])
                control_matrix = reg_fun(response_matrix, reg_conds[j])
                modal_coefficients = -control_matrix.dot( efield_ri )
                command_per_reg[j] = (1.0-leakage)*command + loop_gain*modal_coefficients
                
                act_commands = calibration_modes.T.dot(utils.ensure_np_array(command_per_reg[j]))
                del_dm1_per_reg[j] = act_commands[:sysi.Nact**2].reshape(sysi.Nact,sysi.Nact)
                del_dm2_per_reg[j] = act_commands[sysi.Nact**2:].reshape(sysi.Nact,sysi.Nact)
                
                # Set the current DM state
                # sysi.set_dm1(dm1_ref + dm1_command_per_reg[j])
                # sysi.set_dm2(dm2_ref + dm2_command_per_reg[j])
                nonlin_model.add_dm1(del_dm1_per_reg[j])
                nonlin_model.add_dm2(del_dm2_per_reg[j])
                
                # Take an image to estimate the metrics with the nonlinear model
                # image_per_reg[j] = copy.copy(sysi.snap())
                image_per_reg[j] = copy.copy(nonlin_model.snap())
                # print(xp.mean(image_per_reg[j][control_mask]))
                mean_nis[j] = xp.mean(image_per_reg[j][control_mask])

                nonlin_model.add_dm1(-del_dm1_per_reg[j])
                nonlin_model.add_dm2(-del_dm2_per_reg[j])

            # print(mean_nis)
            j_min = ensure_np_array(np.argmin(mean_nis))
            best_reg = reg_conds[j_min]
            print(f'\t\tBest regularization parameter is {best_reg:f}')
            # print('Mean NIs: ', mean_nis)
            mean_ni = mean_nis[j_min]
            command = command_per_reg[j_min]
            best_del_dm1 = del_dm1_per_reg[j_min]
            best_del_dm2 = del_dm2_per_reg[j_min]
            best_image = image_per_reg[j_min]

            best_dm1_command = dm1_ref + best_del_dm1
            best_dm2_command = dm2_ref + best_del_dm2
            sysi.set_dm1(best_dm1_command)
            sysi.set_dm2(best_dm2_command)
            nonlin_model.set_dm1(best_dm1_command)
            nonlin_model.set_dm2(best_dm2_command)
            best_image = sysi.snap()
            regs.append(best_reg)

        metric_images.append(copy.copy(best_image))
        dm1_commands.append(best_dm1_command)
        dm2_commands.append(best_dm2_command)

        if plot_current or plot_all:

            imshows.imshow3(dm1_commands[i], dm2_commands[i], best_image, 
                               'DM1', 'DM2', f'Image: Iteration {i+starting_iteration+1}\nMean NI: {mean_ni:.3e}',
                            cmap1='viridis', cmap2='viridis',
                            cbar1_label='m', cbar2_label='m', cbar3_label='NI',
                            xlabel1='Actuators', xlabel2='Actuators',
                               lognorm3=True, vmin3=1e-11, pxscl3=sysi.psf_pixelscale_lamD, xlabel3='$\lambda/D$')

            if plot_sms:
                sms_fig = sms(U, s, alpha2, efield_ri, Nmask, Imax_unocc, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(best_image, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)

    metric_images = xp.array(metric_images)
    dm1_commands = xp.array(dm1_commands)
    dm2_commands = xp.array(dm2_commands)
    
    if old_images is not None:
        metric_images = xp.concatenate([old_images, metric_images], axis=0)
    if old_dm1_commands is not None: 
        dm1_commands = xp.concatenate([old_dm1_commands, dm1_commands], axis=0)
    if old_dm2_commands is not None:
        dm2_commands = xp.concatenate([old_dm2_commands, dm2_commands], axis=0)
    if old_regs is not None:
        regs = xp.concatenate([xp.array(old_regs), xp.array(regs)], axis=0)

    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return metric_images, dm1_commands, dm2_commands, regs



