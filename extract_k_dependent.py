import random
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

from extraction_functions import EDC_prep, EDC_array_with_SE
from general import ONE_BILLION, F_test, polynomial_functions


class KDependentExtractor:
    # Optionally save scale/T/secondary_electron_scale trajectories to save time
    scale_trajectory = None
    T_trajectory = None
    T1_trajectory = None
    secondary_electron_scale_trajectory = None
    dk_trajectory = None

    def __init__(self, Z, w, k, initial_a_estimate, initial_c_estimate, energy_conv_sigma, temp, min_fit_count=1):
        self.scale_polynomial_degree = None
        self.T0_polynomial_degree = None
        self.T1_polynomial_degree = None
        self.secondary_electron_scale_degree = None
        self.scale_polynomial_fit = None
        self.T_polynomial_fit = None
        self.T1_polynomial_fit = None
        self.secondary_electron_scale_fit = None
        self.Z = Z
        self.w = w
        self.k = k
        self.initial_a_estimate = initial_a_estimate
        self.initial_c_estimate = initial_c_estimate
        self.energy_conv_sigma = energy_conv_sigma
        self.temp = temp
        self.min_fit_count = min_fit_count

    def get_kdependent_trajectory(self, print_results=False, plot_fits=False):
        z_width = self.Z[0].size
        scale_trajectory = np.zeros(z_width)
        T_trajectory = np.zeros(z_width)
        SEC_trajectory = np.zeros(z_width)
        dk_trajectory = np.zeros(z_width)
        params = [1, 1, 1, 1500, -10, 0.1, 200]
        fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))
        for i in fitting_range:
            if i == int(z_width / 2) - 1:
                save_params = params
            if i == int(z_width / 2) + 1:
                params = save_params
            low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index = \
                EDC_prep(i, self.Z, self.w, self.min_fit_count, exclude_secondary=False)
            try:
                params, pcov = scipy.optimize.curve_fit(
                    partial(EDC_array_with_SE, a=self.initial_a_estimate, c=self.initial_c_estimate, fixed_k=self.k[i],
                            energy_conv_sigma=self.energy_conv_sigma, temp=self.temp), low_noise_w, low_noise_slice,
                    bounds=(
                        [0, 0, 0, 0, -np.inf, 0, -np.inf],
                        [ONE_BILLION, 75, 75, np.inf, 0, np.inf, np.inf]),
                    p0=params,
                    sigma=fitting_sigma,
                    maxfev=2000)
            except RuntimeError:
                print("Optimal parameters not found, skipping...")
            scale_trajectory[i] = params[0]
            T_trajectory[i] = params[1]
            dk_trajectory[i] = params[2]
            SEC_trajectory[i] = params[3]
            if plot_fits:
                plt.title(str(i))
                plt.plot(low_noise_w, low_noise_slice)
                plt.plot(low_noise_w,
                         EDC_array_with_SE(
                             low_noise_w, *params, self.initial_a_estimate, self.initial_c_estimate, self.k[i],
                             self.energy_conv_sigma, self.temp))
                plt.show()
            print(i)
        KDependentExtractor.scale_trajectory = scale_trajectory
        KDependentExtractor.T_trajectory = T_trajectory
        KDependentExtractor.dk_trajectory = dk_trajectory
        KDependentExtractor.secondary_electron_scale_trajectory = SEC_trajectory

        if print_results:
            print("Scale trajectory: ")
            scale_trajectory_string = ""
            for i in range(scale_trajectory.size):
                scale_trajectory_string += str(scale_trajectory[i]) + ", "
            print(scale_trajectory_string)
            print("T trajectory: ")
            T_trajectory_string = ""
            for i in range(T_trajectory.size):
                T_trajectory_string += str(T_trajectory[i]) + ", "
            print(T_trajectory_string)
            print("dk trajectory: ")
            dk_trajectory_string = ""
            for i in range(dk_trajectory.size):
                dk_trajectory_string += str(dk_trajectory[i]) + ", "
            print(dk_trajectory_string)
            print("SEC trajectory: ")
            SEC_trajectory_string = ""
            for i in range(SEC_trajectory.size):
                SEC_trajectory_string += str(SEC_trajectory[i]) + ", "
            print(SEC_trajectory_string)

    def get_secondary_electron_scale_trajectory(self, y_pos, plot=True):
        """
        :param plot:
        :param y_pos: index of MDC to fit secondary electron polynomial to
        :return:
        """
        if not (0 <= y_pos < self.w.size):
            raise ValueError("y_pos out of range of w")
        KDependentExtractor.secondary_electron_scale_trajectory = self.Z[y_pos]
        if plot:
            plt.title("Secondary electron scale extraction line")
            im = plt.imshow(self.Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(self.k), max(self.k), min(self.w), max(self.w)])
            plt.hlines(self.w[y_pos], min(self.k), max(self.k))
            plt.colorbar(im)
            plt.show()
            plt.title("Secondary electron scale trajectory")
            plt.plot(self.k, KDependentExtractor.secondary_electron_scale_trajectory)
            plt.show()

    def get_scale_polynomial_fit(self, plot=True):
        if KDependentExtractor.scale_trajectory is None:
            raise AttributeError("Uninitialized scale_trajectory.")

        inner_params, polynomial_function, self.scale_polynomial_degree, self.scale_polynomial_fit = self.get_polynomial_fit(
            self.scale_trajectory, curve_name="scale", plot=plot)
        return inner_params, polynomial_function

    def get_T_polynomial_fit(self, plot=True):
        if KDependentExtractor.T_trajectory is None:
            raise AttributeError("Uninitialized T_trajectory.")

        inner_params, polynomial_function, self.T0_polynomial_degree, self.T_polynomial_fit = self.get_polynomial_fit(
            self.T_trajectory, curve_name="T", plot=plot)
        return inner_params, polynomial_function

    def get_secondary_electron_scale_polynomial_fit(self, plot=True):
        if KDependentExtractor.secondary_electron_scale_trajectory is None:
            raise AttributeError("Uninitialized secondary_electron_scale_trajectory.")

        inner_params, polynomial_function, self.secondary_electron_scale_degree, self.secondary_electron_scale_fit = self.get_polynomial_fit(
            self.secondary_electron_scale_trajectory, curve_name="secondary electron scale", plot=plot)
        return inner_params, polynomial_function

    def get_polynomial_fit(self, curve, curve_name="", plot=True):
        for i in range(len(polynomial_functions) - 2):
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k, curve)
            outer_params, outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 1], self.k, curve)
            outer_outer_params, outer_outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 2], self.k, curve)
            inner_fit = polynomial_functions[i](self.k, *inner_params)
            outer_fit = polynomial_functions[i + 1](self.k, *outer_params)
            outer_outer_fit = polynomial_functions[i + 2](self.k, *outer_outer_params)
            if plot:
                plt.plot(self.k, curve)
                plt.plot(self.k, inner_fit)
                plt.show()
            sig_lvl = 0.05
            F_statistic = F_test(curve, inner_fit, i + 1, outer_fit, i + 2, np.ones(len(curve)), len(curve))
            critical_value = scipy.stats.f.ppf(q=1 - sig_lvl, dfn=1, dfd=len(curve) - (i + 2))
            F_statistic_2 = F_test(curve, inner_fit, i + 1, outer_outer_fit, i + 3, np.ones(len(curve)), len(curve))
            critical_value_2 = scipy.stats.f.ppf(q=1 - sig_lvl, dfn=2, dfd=len(curve) - (i + 3))
            if (F_statistic < critical_value and F_statistic_2 < critical_value_2) or (np.isclose(outer_params[0], 1) and np.isclose(outer_outer_params[0], 1)):
                print(F_statistic, critical_value)
                print("Optimal " + curve_name + " polynomial is of degree: " + str(i + 1))
                print("Fitted " + curve_name + " parameters are: " + str(inner_params))
                curve_degree = i + 1
                curve_fit = inner_fit
                return inner_params, polynomial_functions[i], curve_degree, curve_fit
        raise RuntimeError("Unable to find optimal " + curve_name + " polynomial fit")

    def plot(self):
        if self.scale_polynomial_fit is None or self.T_polynomial_fit is None or self.T1_polynomial_fit is None:
            raise NotImplementedError("Must get scale/T polynomial fits before plotting")
        plt.title("Scale")
        plt.plot(self.k, KDependentExtractor.scale_trajectory)
        plt.plot(self.k, self.scale_polynomial_fit)
        plt.show()
        plt.title("T")
        plt.plot(self.k, KDependentExtractor.T_trajectory)
        plt.plot(self.k, self.T_polynomial_fit)
        plt.show()
