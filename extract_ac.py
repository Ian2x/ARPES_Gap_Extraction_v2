from enum import Enum
from functools import partial

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from data_reader import FileType
from fitter import Fitter


class FittingOrder(Enum):
    center_out = 0
    left_to_right = 1
    right_to_left = 2


def extract_ac(Z, k, w, energy_conv_sigma, fileType, plot_EDC_fits=False, fittingOrder=FittingOrder.center_out):
    """
    Extracts initial a and c dispersion estimates by fitting lorentz curves to the trajectory. NOTE: also modifies k if
    there a k-offset is detected
    :param Z:
    :param k:
    :param w:
    :param energy_conv_sigma
    :param fileType:
    :param plot_EDC_fits:
    :param fittingOrder:
    :return: initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, new_k
    """
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)
    super_state_trajectory_errors = np.zeros(z_width)
    simulated = fileType == FileType.SIMULATED

    if fittingOrder == FittingOrder.center_out:
        fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
    elif fittingOrder == FittingOrder.left_to_right:
        fitting_range = range(z_width)
    elif fittingOrder == FittingOrder.right_to_left:
        fitting_range = range(z_width - 1, 0 - 1, -1)

    avgRedchi = 0
    params = None

    stderr_estimate = energy_conv_sigma / np.sqrt(energy_conv_sigma * 2.35482004503 * 2 / (w[0] - w[1]))

    for i in fitting_range:  # width
        if fittingOrder == FittingOrder.center_out and i == int(z_width / 2) + 1:
            params = None

        loc, loc_std, redchi, params = Fitter.NormanFit(Z, k, w, i, energy_conv_sigma, fileType, params=params,
                                                        print_results=False,
                                                        plot_results=plot_EDC_fits and i % int(z_width / 9) == 0)
        avgRedchi += redchi

        # if plot_EDC_fits and i % int(z_width / 9) == 0:
        #     print(params)
        # print(params)

        if loc_std is None or loc_std > stderr_estimate:
            loc_std = stderr_estimate
        super_state_trajectory[i] = -loc
        super_state_trajectory_errors[i] = loc_std

    # Plot trajectory
    # im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])
    # plt.title(f"Super state trajectory")
    # plt.colorbar(im)
    # plt.plot(k, super_state_trajectory, label='trajectory')
    # plt.ylim([min(w), max(w)])
    # plt.show()

    # print(energy_conv_sigma)
    # print(super_state_trajectory)
    # print(super_state_trajectory_errors)

    avgRedchi /= z_width

    # print(f"Average lorentz+sigmoid EDC redchi (maxWidth): {avgRedchi}")
    # print("Lorentz fits finished.\n")

    def trajectory_form(x, a, c, dk, k_error):
        return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5

    pars = lmfit.Parameters()
    pars.add('a', value=1901.549259 if simulated else 2800, min=0)
    pars.add('c', value=-20.6267059166932 if simulated else -27, max=0)
    pars.add('dk', value=1)
    pars.add('k_error', value=0, min=-0.03, max=0.03, vary=not (simulated or fileType == FileType.NEAR_NODE))

    def calculate_residual(p, k, sst, sst_error):
        residual = (trajectory_form(k, p['a'], p['c'], p['dk'], p['k_error']) - sst) / sst_error
        return residual

    # Estimate kf
    fit_function = partial(calculate_residual, k=k, sst=super_state_trajectory,
                           sst_error=super_state_trajectory_errors)
    mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
    result = mini.minimize(method='least_squares')
    kf_estimate = (-result.params.get('c').value / result.params.get('a').value) ** 0.5
    k_error = result.params.get('k_error').value

    start = int(z_width / 2)
    start_peak = -np.inf
    end = int(z_width / 2)
    end_peak = -np.inf
    if fittingOrder == FittingOrder.right_to_left:
        end = z_width - 1
    elif fittingOrder == FittingOrder.left_to_right:
        start = 0

    max_decrement = energy_conv_sigma
    while start > 0:
        if fileType != FileType.ANTI_NODE:
            if super_state_trajectory[start - 1] < start_peak - max_decrement and int(z_width / 2) - start > 10 and k[
                start - 1] < (
                    -0.85 if fileType == FileType.FAR_OFF_NODE else -1.0) * kf_estimate:
                print("WARNING: START BROKE BEFORE KF POINT")
                break
            elif k[start - 1] < 1.0 * (-kf_estimate):
                break
        else:
            if super_state_trajectory[start - 1] < start_peak - max_decrement and k[start - 1] < -0.025:
                print("WARNING: START BROKE BEFORE KF POINT")
                break
            elif k[start - 1] < -0.07:
                break
        start -= 1
        if super_state_trajectory[start] < -energy_conv_sigma:
            start_peak = max(super_state_trajectory[start], start_peak)
    while end < z_width - 1:
        if fileType != FileType.ANTI_NODE:
            if super_state_trajectory[end + 1] < end_peak - max_decrement and end - int(z_width / 2) > 10 and k[
                end + 1] > (
                    0.85 if fileType == FileType.FAR_OFF_NODE else 1.0) * kf_estimate:
                print("WARNING: END BROKE BEFORE KF POINT")
                break
            elif k[end + 1] > 1.0 * kf_estimate:
                break
        else:
            if super_state_trajectory[end + 1] < end_peak - max_decrement and k[end + 1] > 0.02:
                print("WARNING: END BROKE BEFORE KF POINT")
                break
            elif k[end + 1] > 0.09:
                break
        end += 1
        if super_state_trajectory[end] < -energy_conv_sigma:
            end_peak = max(super_state_trajectory[end], end_peak)

    fit_function = partial(calculate_residual, k=k[start:end], sst=super_state_trajectory[start:end],
                           sst_error=super_state_trajectory_errors[start:end])
    mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
    result = mini.minimize(method='least_squares')

    # Plot chosen fit
    im = plt.imshow(Z[:, start:end], cmap=plt.cm.RdBu, aspect='auto',
                    extent=[min(k[start:end]), max(k[start:end]), min(w), max(w)])  # drawing the function
    plt.title(f"From " + str(start) + " to " + str(end))
    plt.colorbar(im)
    plt.plot(k[start:end], super_state_trajectory[start:end], label='trajectory')
    plt.plot(k[start:end],
             trajectory_form(k[start:end], result.params.get('a').value, result.params.get('c').value,
                             result.params.get('dk').value, result.params.get('k_error').value),
             label='trajectory fit')
    plt.ylim([min(w), max(w)])
    plt.show()

    # Check not bounded by window
    assert start != 0 or fittingOrder == FittingOrder.left_to_right
    assert end != z_width - 1 or fittingOrder == FittingOrder.right_to_left

    print("\nREP GAP, error, redchi   [start/end: ", start, "/", end, "] out of [0,", z_width - 1, "]")
    print(np.abs(result.params.get('dk').value), ",", result.params.get('dk').stderr, ",", result.redchi)

    return result.params.get('a').value, result.params.get('c').value, (
            -result.params.get('c').value / result.params.get('a').value) ** 0.5, result.params.get(
        'k_error').value, np.abs(result.params.get('dk').value), result.params.get('dk').stderr, result.redchi
