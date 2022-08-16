from functools import partial

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from general import lorentz_form_with_secondary_electrons, reduced_chi


def extract_ac(Z, k, w, temp, minWidth, maxWidth, plot_trajectory_fits=False, plot_EDC_fits=False):
    """
    Extracts initial a and c dispersion estimates by fitting lorentz curves to the trajectory. NOTE: also modifies k if
    there a k-offset is detected
    :param Z:
    :param k:
    :param w:
    :param temp:
    :param minWidth:
    :param maxWidth:
    :param plot_trajectory_fits:
    :return: initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, new_k
    """
    inv_Z = np.array([list(i) for i in zip(*Z)])
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)
    super_state_trajectory_errors = np.zeros(z_width)
    params = [40000, -25, 20, 1500, -10, 0.1, 200]

    fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
    avgRedchi = 0
    for i in fitting_range:  # width
        if i == int(z_width / 2) - 1:
            save_params = params
        if i == int(z_width / 2) + 1:
            params = save_params
        try:
            simpleEDC = partial(lorentz_form_with_secondary_electrons, temp=temp)
            params, pcov = scipy.optimize.curve_fit(simpleEDC, w, inv_Z[i], p0=params,
                                                    bounds=([0, -70, 0, 0, -70, 0, 0],
                                                            [np.inf, 20, np.inf, np.inf, 0, 1, np.inf]))
            p_sigma = np.sqrt(np.diag(pcov))
            redchi = reduced_chi(inv_Z[i], simpleEDC(w, *params), inv_Z[i], len(inv_Z[i]) - 7)
            avgRedchi += redchi
            if plot_EDC_fits and i < 10:
                plt.title(str(i))
                plt.plot(w, inv_Z[i])
                plt.plot(w,
                         simpleEDC(
                             w, *params))
                plt.show()
        except RuntimeError:
            print('ERROR: Extract ac failed on index ' + str(i))
            quit()
        super_state_trajectory[i] = params[1]
        super_state_trajectory_errors[i] = p_sigma[1]

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
        start = round((maxWidth - width) / 2)
        end = start + width
        fit_function = partial(calculate_residual, k=k[start:end], sst=super_state_trajectory[start:end],
                               sst_error=super_state_trajectory_errors[start:end])
        mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        dk_over_width[i] = result.params.get('dk').value
        dk_error_over_width[i] = result.params.get('dk').stderr
        redchi_over_width[i] = result.redchi
        kf_over_width[i] = (-result.params.get('c') / result.params.get('a')) ** 0.5

        if plot_trajectory_fits:
            plt_Z = Z[:,start:end]
            im = plt.imshow(plt_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(k[start:end]), max(k[start:end]), min(w), max(w)])  # drawing the function
            plt.title(f"Width of {width}")
            plt.colorbar(im)
            plt.plot(k[start:end], super_state_trajectory[start:end], label='trajectory')
            plt.plot(k[start:end],
                     trajectory_form(k[start:end], result.params.get('a').value, result.params.get('c').value,
                                     result.params.get('dk').value, result.params.get('k_error').value),
                     label='trajectory fit')
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
