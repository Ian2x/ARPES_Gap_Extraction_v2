import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from general import lorentz_form_with_secondary_electrons


def extract_ac(Z, k, w, show_results=False, fix_k=True):
    """
    Extracts initial a and c dispersion estimates by fitting lorentz curves to the trajectory. NOTE: also modifies k if
    there a k-offset is detected
    :param Z:
    :param k:
    :param w:
    :param show_results:
    :param fix_k: Adjust k if a k-offset is detected
    :return: initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, new_k
    """
    inv_Z = np.array([list(i) for i in zip(*Z)])
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)

    # increase fitting speed by saving data
    params, pcov = scipy.optimize.curve_fit(
        lorentz_form_with_secondary_electrons, w, inv_Z[0], bounds=(
            [0, -70, 0, 500, -70, 0, 0],
            [np.inf, 0, np.inf, 700, 0, 0.5, 100]), maxfev=2000)

    last_a, last_b, last_c, last_p, last_q, last_r, last_s = params

    super_state_trajectory[0] = last_b

    for i in range(1, z_width):  # width
        params, pcov = scipy.optimize.curve_fit(lorentz_form_with_secondary_electrons, w, inv_Z[i], p0=(
              last_a, last_b, last_c, last_p, last_q, last_r, last_s), maxfev=2000)
        last_a, last_b, last_c, last_p, last_q, last_r, last_s = params
        super_state_trajectory[i] = last_b

    def trajectory_form(x, a, c, dk, k_error):
        return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5

    params, pcov = scipy.optimize.curve_fit(trajectory_form, k,
                                                                                    super_state_trajectory,
                                                                                    bounds=(
                                                                                        [0, -np.inf, 0, -0.02],
                                                                                        [np.inf, 0, np.inf, 0.02]))
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, k_error = params
    if fix_k:
        k = k - k_error
    initial_kf_estimate = (-initial_c_estimate / initial_a_estimate) ** 0.5

    if show_results:
        print("INITIAL AC PARAMS [a, c, dk, k shift]:")
        print(params)
        print("\nINITIAL KF ESTIMATE:")
        print(initial_kf_estimate)
        plt.title("Initial AC extraction and k error calculation")
        im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto',
                        extent=[min(k), max(k), min(w), max(w)])  # drawing the function
        plt.colorbar(im)
        plt.plot(k, super_state_trajectory, label='trajectory')
        plt.plot(k, trajectory_form(k, initial_a_estimate, initial_c_estimate, initial_dk_estimate, 0), label='trajectory fit')
        plt.show()

    return initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, k