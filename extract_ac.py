import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from general import lorentz_form_with_secondary_electrons


def extract_ac(Z, k, w, show_results=False):
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
    params = [40000, -25, 20, 1500, -10, 0.1, 200]

    super_state_trajectory[0] = params[1]

    fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
    for i in fitting_range:  # width
        if i == int(z_width / 2) - 1:
            save_params = params
        if i == int(z_width / 2) + 1:
            params = save_params
        try:
            params, pcov = scipy.optimize.curve_fit(lorentz_form_with_secondary_electrons, w, inv_Z[i], p0=params,
                                                    bounds=([0, -70, 0, 0, -70, 0, 0],
                                                            [np.inf, 0, np.inf, np.inf, 0, 1, np.inf]))
        except RuntimeError:
            print('ERROR: Extract ac failed on index ' + str(i))
            quit()
        super_state_trajectory[i] = params[1]

    def trajectory_form(x, a, c, dk, k_error):
        return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5

    params, pcov = scipy.optimize.curve_fit(trajectory_form, k,
                                            super_state_trajectory,
                                            bounds=(
                                                [0, -np.inf, 0, -0.03],
                                                [np.inf, 0, np.inf, 0.03]))
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, k_error = params

    initial_kf_estimate = (-initial_c_estimate / initial_a_estimate) ** 0.5

    new_k = k - k_error
    if show_results:
        print("INITIAL AC PARAMS [a, c, dk, k shift]:")
        print(params)
        print("\nINITIAL KF ESTIMATE:")
        print(str(initial_kf_estimate) + "\n")
        plt.title("Initial AC extraction and k error calculation")
        im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto',
                        extent=[min(new_k), max(new_k), min(w), max(w)])  # drawing the function
        plt.colorbar(im)
        plt.plot(new_k, super_state_trajectory, label='trajectory')
        plt.plot(new_k, trajectory_form(new_k, initial_a_estimate, initial_c_estimate, initial_dk_estimate, 0),
                 label='trajectory fit')
        plt.show()

    return initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, k_error
