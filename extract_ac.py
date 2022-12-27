from enum import Enum
from functools import partial

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from extraction_functions import final_EDC_array_with_SE, final_EDC_array
from general import lorentz_form_with_secondary_electrons, lorentz_form, gaussian_form_with_secondary_electrons


class FittingOrder(Enum):
    center_out = 0
    left_to_right = 1
    right_to_left = 2


def extract_ac(Z, k, w, temp, minWidth, maxWidth, energy_conv_sigma, fullFunc=True, plot_trajectory_fits=False,
               plot_EDC_fits=False, fittingOrder=FittingOrder.center_out, noBackground=True, simulated=False):
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
    :param plot_trajectory_fits:
    :param plot_EDC_fits:
    :param fittingOrder:
    :param noBackground:
    :param simulated:
    :return: initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, new_k
    """
    inv_Z = np.array([list(i) for i in zip(*Z)])
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)
    super_state_trajectory_errors = np.zeros(z_width)

    if fittingOrder == FittingOrder.center_out:
        fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
    elif fittingOrder == FittingOrder.left_to_right:
        fitting_range = range(z_width)
    elif fittingOrder == FittingOrder.right_to_left:
        fitting_range = range(z_width - 1, 0 - 1, -1)

    individual_EDC_fits = np.zeros(len(fitting_range))

    avgRedchi = 0
    reset = True
    for i in fitting_range:  # width
        if fittingOrder == FittingOrder.center_out and i == int(z_width / 2) + 1:
            reset = True
        pars = lmfit.Parameters()
        if fullFunc:
            if noBackground:
                EDC = partial(final_EDC_array, energy_conv_sigma=energy_conv_sigma, temp=temp)
                def calculate_residual(p):
                    return final_EDC_array(w, p['scale'], p['T1'], p['T0'], p['dk'], p['loc'], energy_conv_sigma,
                                           temp) - inv_Z[i]
                pars.add('scale', value=3e+07 if reset else result.params.get('scale').value, min=0,
                         max=np.inf)
                pars.add('T1', value=20 if reset else result.params.get('T1').value, min=0, max=np.inf)
                pars.add('T0', value=0 if reset else result.params.get('T0').value, min=0, max=np.inf)
                pars.add('dk', value=1 if reset else result.params.get('dk').value, min=0, max=np.inf)
                pars.add('loc', value=-25 if reset else result.params.get('loc').value, min=-np.inf, max=30)
            else:
                EDC = partial(final_EDC_array_with_SE, energy_conv_sigma=energy_conv_sigma, temp=temp)

                def calculate_residual(p):
                    return final_EDC_array_with_SE(w, p['scale'], p['T1'], p['T0'], p['dk'], p['p'], p['q'], p['r'],
                                                   p['s'], p['loc'], energy_conv_sigma, temp) - inv_Z[i]

                pars.add('scale', value=1.6e+05 if reset else result.params.get('scale').value, min=0, max=np.inf)
                pars.add('T1', value=20 if reset else result.params.get('T1').value, min=0, max=np.inf)
                pars.add('T0', value=0 if reset else result.params.get('T0').value, min=0, max=np.inf)
                pars.add('dk', value=0 if reset else result.params.get('dk').value, min=0, max=np.inf, vary=False)
                pars.add('p', value=1400 if reset else result.params.get('p').value, min=0, max=np.inf)
                pars.add('q', value=-17 if reset else result.params.get('q').value, min=-np.inf, max=30)
                pars.add('r', value=0.07 if reset else result.params.get('r').value, min=-np.inf, max=np.inf)
                pars.add('s', value=250 if reset else result.params.get('s').value, min=0, max=np.inf)
                pars.add('loc', value=-25 if reset else result.params.get('loc').value, min=-np.inf, max=30)
        else:
            if noBackground:
                EDC = partial(lorentz_form, temp=temp)
                def calculate_residual(p):
                    return lorentz_form(w, p['a'], p['b'], p['c'], p['d'], temp) - inv_Z[i]

                pars.add('a', value=2000000 if reset else result.params.get('a').value, min=0, max=np.inf)
                pars.add('b', value=-50 if reset else result.params.get('b').value, min=-70, max=20)
                pars.add('c', value=10 if reset else result.params.get('c').value, min=0, max=np.inf)
                pars.add('d', value=0 if reset else result.params.get('d').value, min=0, max=np.inf)
            else:
                EDC = partial(lorentz_form_with_secondary_electrons, temp=temp)
                def calculate_residual(p):
                    return lorentz_form_with_secondary_electrons(w, p['a'], p['b'], p['c'], p['p'], p['q'],
                                                                 p['r'], p['s'], temp) - inv_Z[i]

                pars.add('a', value=(6e+06 if simulated else 40000) if reset else result.params.get('a').value, min=0, max=np.inf)
                pars.add('b', value=(-20 if simulated else -25) if reset else result.params.get('b').value, min=-70, max=20)
                pars.add('c', value=(8 if simulated else 20) if reset else result.params.get('c').value, min=0, max=np.inf)
                pars.add('p', value=(80000 if simulated else 1500) if reset else result.params.get('p').value, min=0, max=np.inf)
                pars.add('q', value=(-10 if simulated else -10) if reset else result.params.get('q').value, min=-70, max=0)
                pars.add('r', value=(5 if simulated else 0.1) if reset else result.params.get('r').value, min=0, max=np.inf)
                pars.add('s', value=(70000 if simulated else 200) if reset else result.params.get('s').value, min=0, max=np.inf)

        mini = lmfit.Minimizer(calculate_residual, pars)
        result = mini.minimize(method='least_squares', ftol=1e+03 * 1.49012e-08)
        avgRedchi += result.redchi
        reset = False

        if plot_EDC_fits and i % (z_width / 10) == 0:
            print(*result.params.values())
            plt.title(str(k[i]))
            plt.plot(w, inv_Z[i])
            plt.plot(w,
                     EDC(w, *result.params.values()))
            plt.show()
        if not fullFunc:
            super_state_trajectory[i] = -np.abs(result.params.get('b').value)
            super_state_trajectory_errors[i] = result.params.get('b').stderr
        else:
            loc = result.params.get('loc').value
            loc_std = result.params.get('loc').stderr
            gap = result.params.get('dk').value
            gap_std = result.params.get('dk').stderr
            if loc_std is None:
                loc_std = energy_conv_sigma
            if gap_std is None:
                gap_std = energy_conv_sigma

            individual_EDC_fits[i] = gap

            super_state_trajectory[i] = -np.sqrt(loc ** 2 + gap ** 2)
            super_state_trajectory_errors[i] = super_state_trajectory[i] * np.sqrt(
                (loc * loc_std) ** 2 + (gap * gap_std) ** 2) / (loc ** 2 + gap ** 2)
    avgRedchi /= z_width
    print(f"Average lorentz+sigmoid EDC redchi (maxWidth): {avgRedchi}")
    print("Lorentz fits finished.\n")

    def trajectory_form(x, a, c, dk, k_error):
        return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5

    pars = lmfit.Parameters()
    pars.add('a', value=3295.466693229615 if noBackground else 2000, min=0)
    pars.add('c', value=-27.303276388313904 if noBackground else -25, max=0)
    pars.add('dk', value=0.1)
    pars.add('k_error', value=0, min=-0.03, max=0.03)

    def calculate_residual(p, k, sst, sst_error):
        residual = (trajectory_form(k, p['a'], p['c'], p['dk'], p['k_error']) - sst) / sst_error
        return residual

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
        result = mini.minimize(method='leastsq')
        dk_over_width[i] = result.params.get('dk').value
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

        if plot_trajectory_fits and i % 4 == 0:
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
    # print("Gap size as a function of width:")
    # print(repr(dk_over_width))
    # print("Gap error as a function of width:")
    # print(repr(dk_error_over_width))
    # print("Reduced chi as a function of width:")
    # print(repr(redchi_over_width))
    # print("kf as a function of width:")
    # print(repr(kf_over_width))
    print(np.mean(dk_over_width))
    print(dk_over_width)
    print(dk_error_over_width)
    print(redchi_over_width)
    print("\nREP GAP, error, redchi:")
    rep_i = None
    for i, width in reversed(list(enumerate(range(minWidth, maxWidth + 1)))):
        if k_over_width[i] < (1.0 if noBackground else 0.95) * kf_over_width[i]:
            rep_i = i
            print("width:", width, " | L/R out on:", lr_over_width[i])
            print(dk_over_width[i], ",", dk_error_over_width[i], ",", redchi_over_width[i])
            break
    if rep_i is None:
        print("FAILED TO FIND REP for ", kf_over_width[i])
        assert False

    if fullFunc:
        # print("All EDC gaps:")
        # print(repr(individual_EDC_fits))
        # print("\nEDC gap size (adjusted by", np.median(k_error_over_width), "to k=", adjusted_kf, "):")
        # print(individual_EDC_fits[k_as_index(adjusted_kf, k)])
        # print(repr(individual_EDC_fits))
        # print(-np.median(kf_over_width) + np.median(k_error_over_width))
        return a_over_width[rep_i], c_over_width[rep_i], kf_over_width[rep_i], k_error_over_width[rep_i]

    return a_over_width[rep_i], c_over_width[rep_i], kf_over_width[rep_i], k_error_over_width[rep_i]

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
