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


def extract_ac(Z, k, w, energy_conv_sigma, fileType, fittingOrder=FittingOrder.center_out, plot=False):
    """
    Extracts initial a and c dispersion estimates by fitting lorentz curves to the trajectory. NOTE: also modifies k if
    there a k-offset is detected
    :param Z:
    :param k:
    :param w:
    :param energy_conv_sigma
    :param fileType:
    :param plot:
    :param fittingOrder:
    :return: initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, new_k
    """
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)
    super_state_trajectory_errors = np.zeros(z_width)
    simulated = fileType == FileType.SIMULATED

    # im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])
    # plt.title(f"Plain plot")
    # plt.colorbar(im)
    # plt.ylim([min(w), max(w)])
    # plt.show()
    # return

    if fittingOrder == FittingOrder.center_out:
        fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
    elif fittingOrder == FittingOrder.left_to_right:
        fitting_range = range(z_width)
    elif fittingOrder == FittingOrder.right_to_left:
        fitting_range = range(z_width - 1, 0 - 1, -1)

    avgRedchi = 0
    params = None

    max_std_err = energy_conv_sigma / 2.35482004503

    for i in fitting_range:  # width
        if fittingOrder == FittingOrder.center_out and i == int(z_width / 2) + 1:
            params = None

        loc_1, loc_std_err_1, redchi_1, params_1 = Fitter.NormanFit(Z, k, w, i, energy_conv_sigma, fileType, params=params,
                                                            plot=plot and i % int(z_width / 9) == 0, force_b_zero=False)

        # loc_2, loc_std_err_2, redchi_2, params_2 = Fitter.NormanFit(Z, k, w, i, energy_conv_sigma, fileType, params=params,
        #                                                     plot=True and i % int(z_width / 9) == 0, force_b_zero=False)

        loc, loc_std_err, redchi, params = loc_1, loc_std_err_1, redchi_1, params_1

        # print(redchi_1, redchi_2)
        # if redchi_1 <= 100 * redchi_2:
        #     loc, loc_std_err, redchi, params = loc_1, loc_std_err_1, redchi_1, params_1
        # else:
        #     loc, loc_std_err, redchi, params = loc_2, loc_std_err_2, redchi_2, params_2

        avgRedchi += redchi

        if plot and i % int(z_width / 9) == 0:
            print(str(k[i])+": " + str(params))

        # if loc_std_err is None or (loc_std_err > max_std_err and np.abs(loc) < 0.5):
        #     loc_std_err = max_std_err

        # if np.abs(loc) >= 0.5:
        #     max_std_err = max(max_std_err, loc_std_err)
        #
        # if loc_std_err is None and not np.isclose(0, loc):
        #     loc_std_err = energy_conv_sigma / 2.35482004503

        # if not np.isclose(0, loc):
        #     new_max_std_err = max(new_max_std_err, loc_std_err)

        super_state_trajectory[i] = -loc
        super_state_trajectory_errors[i] = loc_std_err

    # Added 6/3/23 as test
    # for i in range(len(super_state_trajectory_errors)):
    #     if np.abs(super_state_trajectory[i] < 0.5):
    #         if super_state_trajectory[i] is None or super_state_trajectory_errors[i] > max_std_err:
    #             super_state_trajectory_errors[i] = max_std_err
    # print(max_std_err)
    # super_state_trajectory_errors = [min(x, new_max_std_err) for x in super_state_trajectory_errors]



    # Plot trajectory
    if plot:
        im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])
        plt.title(f"Super state trajectory")
        plt.colorbar(im)
        plt.plot(k, super_state_trajectory, label='trajectory')
        plt.ylim([min(w), max(w)])
        plt.show()

        # print(energy_conv_sigma)
        # print(super_state_trajectory)
        # print(super_state_trajectory_errors)

        avgRedchi /= z_width

        print(f"Average lorentz+sigmoid EDC redchi (maxWidth): {avgRedchi}")
        print("Lorentz fits finished.\n")

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

    t1, t2, t3 = np.array([]), np.array([]), np.array([])
    for i in range(len(super_state_trajectory_errors)):
        if super_state_trajectory_errors[i] is not None and (abs(k[i]) > 0.075 or super_state_trajectory_errors[i] < abs(super_state_trajectory[i])):
            np.append(t1, k[i])
            np.append(t2, super_state_trajectory[i])
            np.append(t3, super_state_trajectory_errors[i] / np.exp(abs(k[i])))

    # Estimate kf
    fit_function = partial(calculate_residual, k=t1, sst=t2,
                           sst_error=t3)
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

    EXTRA_KF_FACTOR = 1
    while start > 0:
        if fileType != FileType.ANTI_NODE:
            if super_state_trajectory[start - 1] < start_peak - max_decrement and k[start - 1] < -0.85 * kf_estimate + k_error:
                print("WARNING: START BROKE BEFORE KF POINT")
                break
            elif k[start - 1] < -EXTRA_KF_FACTOR * kf_estimate + k_error:  # -0.07: # 1.0 * (-kf_estimate):
                break
            start -= 1
            if super_state_trajectory[start] < -energy_conv_sigma:
                start_peak = max(super_state_trajectory[start], start_peak)
        else:
            start -= 1
            if k[start - 1] < -0.06:
                break
    while end < z_width - 1:
        if fileType != FileType.ANTI_NODE:
            if super_state_trajectory[end + 1] < end_peak - max_decrement and k[end + 1] > 0.85 * kf_estimate + k_error:
                print("WARNING: END BROKE BEFORE KF POINT")
                break
            elif k[end + 1] > EXTRA_KF_FACTOR * kf_estimate + k_error:  # 0.09:  # 1.0 * kf_estimate:
                break
            end += 1
            if super_state_trajectory[end] < -energy_conv_sigma:
                end_peak = max(super_state_trajectory[end], end_peak)
        else:
            end += 1
            if k[end + 1] > 0.08:
                break

    if simulated:
        print(super_state_trajectory_errors)
        # super_state_trajectory_errors = np.ones(len(super_state_trajectory_errors))

    fit_function = partial(calculate_residual, k=k[start:end], sst=super_state_trajectory[start:end],
                           sst_error=super_state_trajectory_errors[start:end])
    mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
    result = mini.minimize(method='least_squares')

    # Plot chosen fit
    if plot or True:
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

        print("\nREP GAP, error, redchi   [start/end: ", start, "/", end, "] out of [0,", z_width - 1, "]")
        print(np.abs(result.params.get('dk').value), ",", result.params.get('dk').stderr, ",", result.redchi)

    # Check not bounded by window
    # assert start != 0 or fittingOrder == FittingOrder.left_to_right
    # assert end != z_width - 1 or fittingOrder == FittingOrder.right_to_left

    return result.params.get('a').value, result.params.get('c').value, (
            -result.params.get('c').value / result.params.get('a').value) ** 0.5, result.params.get(
        'k_error').value, np.abs(result.params.get('dk').value), result.params.get('dk').stderr, result.redchi
