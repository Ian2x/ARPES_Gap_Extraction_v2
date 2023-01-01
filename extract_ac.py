from enum import Enum
from functools import partial

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from fitter import Fitter
from general import reject_outliers


class FittingOrder(Enum):
    center_out = 0
    left_to_right = 1
    right_to_left = 2


def extract_ac(Z, k, w, minWidth, maxWidth, energy_conv_sigma, plot_trajectory_fits=False,
               plot_EDC_fits=False, fittingOrder=FittingOrder.center_out, simulated=False):
    """
    Extracts initial a and c dispersion estimates by fitting lorentz curves to the trajectory. NOTE: also modifies k if
    there a k-offset is detected
    :param Z:
    :param k:
    :param w:
    :param minWidth:
    :param maxWidth:
    :param energy_conv_sigma
    :param plot_trajectory_fits:
    :param plot_EDC_fits:
    :param fittingOrder:
    :param simulated:
    :return: initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, new_k
    """
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)
    super_state_trajectory_errors = np.zeros(z_width)

    if fittingOrder == FittingOrder.center_out:
        fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
    elif fittingOrder == FittingOrder.left_to_right:
        fitting_range = range(z_width)
    elif fittingOrder == FittingOrder.right_to_left:
        fitting_range = range(z_width - 1, 0 - 1, -1)

    avgRedchi = 0
    params = None
    for i in fitting_range:  # width
        if fittingOrder == FittingOrder.center_out and i == int(z_width / 2) + 1:
            params = None

        loc, loc_std, redchi, params = Fitter.NormanFit(Z, k, w, i, energy_conv_sigma, simulated, params=params, print_results=False, plot_results=plot_EDC_fits and i % int(z_width / 9) == 0)
        avgRedchi += redchi

        if plot_EDC_fits and i % int(z_width / 9) == 0:
            print(params)

        if loc_std is None or loc_std > energy_conv_sigma:
            loc_std = energy_conv_sigma
        super_state_trajectory[i] = -loc
        super_state_trajectory_errors[i] = loc_std

    avgRedchi /= z_width
    print(f"Average lorentz+sigmoid EDC redchi (maxWidth): {avgRedchi}")
    print("Lorentz fits finished.\n")

    def trajectory_form(x, a, c, dk, k_error):
        return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5

    pars = lmfit.Parameters()
    pars.add('a', value=1901.549259 if simulated else 2800, min=0)
    pars.add('c', value=-20.6267059166932 if simulated else -27, max=0)
    pars.add('dk', value=1)
    pars.add('k_error', value=0, min=-0.03, max=0.03, vary=not simulated)

    def calculate_residual(p, k, sst, sst_error):
        residual = (trajectory_form(k, p['a'], p['c'], p['dk'], p['k_error']) - sst) / sst_error
        return residual

    start = int(z_width / 2)
    start_peak = -np.inf
    end = int(z_width / 2)
    end_peak = -np.inf

    max_decrement = 2.5

    while start > 0:
        if super_state_trajectory[start-1] < start_peak - max_decrement:
            break
        else:
            start -= 1
            start_peak = max(super_state_trajectory[start], start_peak)
    while end < z_width - 1:
        if super_state_trajectory[end+1] < end_peak - max_decrement:
            break
        else:
            end += 1
            end_peak = max(super_state_trajectory[end], end_peak)

    fit_function = partial(calculate_residual, k=k[start:end], sst=super_state_trajectory[start:end],
                           sst_error=super_state_trajectory_errors[start:end])
    mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
    result = mini.minimize(method='least_squares')

    plt_Z = Z[:, start:end]
    im = plt.imshow(plt_Z, cmap=plt.cm.RdBu, aspect='auto',
                    extent=[min(k[start:end]), max(k[start:end]), min(w), max(w)])  # drawing the function
    plt.title(f"Width of {end - start}")
    plt.colorbar(im)
    plt.plot(k[start:end], super_state_trajectory[start:end], label='trajectory')
    plt.plot(k[start:end],
             trajectory_form(k[start:end], result.params.get('a').value, result.params.get('c').value,
                             result.params.get('dk').value, result.params.get('k_error').value),
             label='trajectory fit')
    plt.ylim([min(w), max(w)])
    plt.show()

    print("\nREP GAP, error, redchi   [width:", end-start)
    print(result.params.get('dk').value, ",", result.params.get('dk').stderr, ",", result.redchi)

    return result.params.get('a').value, result.params.get('c').value, (-result.params.get('c').value / result.params.get('a').value) ** 0.5, result.params.get('k_error').value






    dk_over_width = np.zeros(len(range(minWidth, maxWidth + 1)))
    dk_error_over_width = np.zeros(len(range(minWidth, maxWidth + 1)))
    redchi_over_width = np.zeros(len(range(minWidth, maxWidth + 1)))
    kf_over_width = np.zeros(len(range(minWidth, maxWidth + 1)))
    k_error_over_width = np.zeros(len(range(minWidth, maxWidth + 1)))
    k_over_width = np.zeros((len(range(minWidth, maxWidth + 1))))
    a_over_width = np.zeros((len(range(minWidth, maxWidth + 1))))
    c_over_width = np.zeros((len(range(minWidth, maxWidth + 1))))
    lr_over_width = np.zeros((len(range(minWidth, maxWidth + 1))))

    for i, width in enumerate(range(minWidth, maxWidth + 1)):
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
        result = mini.minimize(method='least_squares')

        dk_over_width[i] = np.abs(result.params.get('dk').value)
        dk_error_over_width[i] = result.params.get('dk').stderr
        redchi_over_width[i] = result.redchi
        k_error_over_width[i] = result.params.get('k_error').value
        kf_over_width[i] = (-result.params.get('c').value / result.params.get('a').value) ** 0.5
        a_over_width[i] = result.params.get('a').value
        c_over_width[i] = result.params.get('c').value

        if np.abs(k[start] - k_error_over_width[i]) > np.abs(k[end - 1] - k_error_over_width[i]):
            k_over_width[i] = np.abs(k[start] - k_error_over_width[i])
            lr_over_width[i] = 0
        else:
            k_over_width[i] = np.abs(k[end - 1] - k_error_over_width[i])
            lr_over_width[i] = 1

        if plot_trajectory_fits and i % 10 == 0:
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

    print(dk_over_width)
    print(dk_error_over_width)
    print(redchi_over_width)
    print(kf_over_width)
    rep_i = None
    for i, width in reversed(list(enumerate(range(minWidth, maxWidth + 1)))):
        if k_over_width[i] <= (1.0 if simulated else 0.9) * max(kf_over_width):
            rep_i = i
            print("\nREP GAP, error, redchi   [width:", width, " | L/R out on:", lr_over_width[i], "]")
            print(dk_over_width[i], ",", dk_error_over_width[i], ",", redchi_over_width[i])
            break
    if rep_i is None:
        print("FAILED TO FIND REP for ", kf_over_width[i])
        assert False

    return a_over_width[rep_i], c_over_width[rep_i], kf_over_width[rep_i], k_error_over_width[rep_i]