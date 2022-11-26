from functools import partial

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from enum import Enum
from extraction_functions import EDC_array_with_SE, final_EDC_array_with_SE, final_EDC_array
from general import lorentz_form_with_secondary_electrons, reduced_chi, lorentz_form


class FittingOrder(Enum):
    center_out = 0
    left_to_right = 1
    right_to_left = 2


def extract_ac(Z, k, w, temp, minWidth, maxWidth, energy_conv_sigma, fullFunc=True, hasBackground=True, plot_trajectory_fits=False,
               plot_EDC_fits=False, fittingOrder=FittingOrder.center_out, simulated=False):
    """
    Extracts initial a and c dispersion estimates by fitting lorentz curves to the trajectory. NOTE: also modifies k if
    there a k-offset is detected
    :param Z:
    :param k:
    :param w:
    :param temp:
    :param minWidth:
    :param maxWidth:
    :param energy_conv_sigma
    :param fullFunc: Whether to use full EDC function
    :param hasBackground: (Only when not fullFunc) Whether to use lorentz with secondary electrons
    :param plot_trajectory_fits:
    :param plot_EDC_fits:
    :param fittingOrder:
    :param simulated:
    :return: initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, new_k
    """
    inv_Z = np.array([list(i) for i in zip(*Z)])
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)
    super_state_trajectory_errors = np.zeros(z_width)
    if fullFunc:
        params = [1e+05, 10, 1, 1500, -7, 0.7, 300, -15]  # For real
        if simulated:
            params = [3e+07, 20, 1, -25]  # For simulated
    else:
        if hasBackground:
            # params = [4000000, -50, 10, 25000, -60, 0.1, 0]  # For simulated
            params = [40000, -25, 20, 1500, -10, 0.1, 200]  # For real
        else:
            params = [2000000, -50, 10, 0]

    if fittingOrder == FittingOrder.center_out:
        fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
    elif fittingOrder == FittingOrder.left_to_right:
        fitting_range = range(z_width)
    elif fittingOrder == FittingOrder.right_to_left:
        fitting_range = range(z_width - 1, 0 - 1, -1)

    avgRedchi = 0
    for i in fitting_range:  # width
        if fittingOrder == FittingOrder.center_out:
            if i == int(z_width / 2) - 1:
                save_params = params
            if i == int(z_width / 2) + 1:
                params = save_params
        try:
            if fullFunc:
                EDC = partial(final_EDC_array_with_SE, energy_conv_sigma=energy_conv_sigma, temp=temp)
                bounds = ([0, 0, 0, 0, -np.inf, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, 30, np.inf, np.inf, 30])  # Real
                if simulated:
                    EDC = partial(final_EDC_array, energy_conv_sigma=energy_conv_sigma, temp=temp)
                    bounds = ([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, 30])
            else:
                if hasBackground:
                    EDC = partial(lorentz_form_with_secondary_electrons, temp=temp)
                    bounds = ([0, -70, 0, 0, -70, 0, 0], [np.inf, 20, np.inf, np.inf, 0, 1, np.inf])
                else:
                    EDC = partial(lorentz_form, temp=temp)
                    bounds = ([0, -70, 0, 0], [np.inf, 0, np.inf, 500000])
            params, pcov = scipy.optimize.curve_fit(EDC, w, inv_Z[i], p0=params, bounds=bounds, ftol=1e+06*1.49012e-08)
            p_sigma = np.sqrt(np.diag(pcov))
            redchi = reduced_chi(inv_Z[i], EDC(w, *params), inv_Z[i], len(inv_Z[i]) - 7)
            avgRedchi += redchi
            if plot_EDC_fits and i % (z_width / 10) == 0:
                print(params)
                plt.title(str(k[i]))
                plt.plot(w, inv_Z[i])
                plt.plot(w,
                         EDC(
                             w, *params))
                plt.show()
        except RuntimeError as err:
            print('ERROR: Extract ac failed on index ' + str(i))
            print(err)
            plt.title("ERROR on " + str(k[i]))
            plt.plot(w, inv_Z[i])
            plt.show()
            quit()
        super_state_trajectory[i] = params[1] if not fullFunc else -np.sqrt((params[7 if not simulated else 3]) ** 2 + params[2] ** 2)
        super_state_trajectory_errors[i] = p_sigma[1] if not fullFunc else 1
    avgRedchi /= z_width
    print(f"Average lorentz+sigmoid EDC redchi (maxWidth): {avgRedchi}")
    print("Lorentz fits finished.\n")

    def trajectory_form(x, a, c, dk, k_error):
        return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5

    pars = lmfit.Parameters()
    pars.add('a', value=3000, min=0)
    pars.add('c', value=-25, max=0)
    pars.add('dk', value=1)
    pars.add('k_error', value=0, min=-0.03, max=0.03)

    def calculate_residual(p, k, sst, sst_error):
        residual = (trajectory_form(k, p['a'], p['c'], p['dk'], p['k_error']) - sst) / sst_error
        return residual

    dk_over_width = np.zeros(len(range(minWidth, maxWidth + 1, 2)))
    dk_error_over_width = np.zeros(len(range(minWidth, maxWidth + 1, 2)))
    redchi_over_width = np.zeros(len(range(minWidth, maxWidth + 1, 2)))
    kf_over_width = np.zeros(len(range(minWidth, maxWidth + 1, 2)))

    for i, width in enumerate(range(minWidth, maxWidth + 1, 2)):
        if fittingOrder == FittingOrder.center_out:
            start = round((maxWidth - width) / 2)
            end = start + width
        elif fittingOrder == FittingOrder.right_to_left:
            start = z_width - width
            end = start + width
        elif fittingOrder == FittingOrder.left_to_right:
            start = 0
            end = start + width

        fit_function = partial(calculate_residual, k=k[start:end], sst=super_state_trajectory[start:end],
                               sst_error=super_state_trajectory_errors[start:end])
        mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        dk_over_width[i] = result.params.get('dk').value
        dk_error_over_width[i] = result.params.get('dk').stderr
        redchi_over_width[i] = result.redchi
        kf_over_width[i] = (-result.params.get('c') / result.params.get('a')) ** 0.5

        if plot_trajectory_fits and i % 2 == 0:
            plt_Z = Z[:, start:end]
            im = plt.imshow(plt_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(k[start:end]), max(k[start:end]), min(w), max(w)])  # drawing the function
            plt.title(f"Width of {width}")
            plt.colorbar(im)
            plt.plot(k[start:end], super_state_trajectory[start:end], label='trajectory')
            plt.plot(k[start:end],
                     trajectory_form(k[start:end], result.params.get('a').value, result.params.get('c').value,
                                     result.params.get('dk').value, result.params.get('k_error').value),
                     label='trajectory fit')
            plt.ylim([min(w), max(w)])
            plt.show()

    dk_over_width = np.abs(dk_over_width)
    print("Gap size as a function of width:")
    print(repr(dk_over_width))
    print("Gap error as a function of width:")
    print(repr(dk_error_over_width))
    print("Reduced chi as a function of width:")
    print(repr(redchi_over_width))
    print("kf as a function of width:")
    print(repr(kf_over_width))

    return

    # fit_function = partial(calculate_residual, k=k)
    #
    # mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
    # result = mini.minimize(method='leastsq')
    # print(lmfit.fit_report(result))
    #
    # initial_a_estimate = result.params.get('a').value
    # initial_c_estimate = result.params.get('c').value
    # initial_dk_estimate = result.params.get('dk').value
    # k_error = result.params.get('k_error').value
    #
    # initial_kf_estimate = (-initial_c_estimate / initial_a_estimate) ** 0.5
    #
    # new_k = k - k_error
    # if plot_trajectory_fits:
    #     print("\nINITIAL KF ESTIMATE:")
    #     print(str(initial_kf_estimate) + "\n")
    #     plt.title("Initial AC extraction and k error calculation")
    #     im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto',
    #                     extent=[min(new_k), max(new_k), min(w), max(w)])  # drawing the function
    #     plt.colorbar(im)
    #     plt.plot(new_k, super_state_trajectory, label='trajectory')
    #     plt.plot(new_k, trajectory_form(new_k, initial_a_estimate, initial_c_estimate, initial_dk_estimate, 0),
    #              label='trajectory fit')
    #     plt.show()
    #
    # return initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, k_error
