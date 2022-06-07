from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

from extraction_functions import EDC_prep, EDC_array
from general import ONE_BILLION, d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial, \
    d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial, d10_polynomial, F_test


class KDependentExtractor:
    # Optionally save scale/T/secondary_electron_scale trajectories to save time
    scale_trajectory = None
    T_trajectory = None
    secondary_electron_scale_trajectory = None

    def __init__(self, Z, w, k, initial_a_estimate, initial_c_estimate, energy_conv_sigma, temp, min_fit_count=1):
        self.scale_polynomial_degree = None
        self.T_polynomial_degree = None
        self.secondary_electron_scale_degree = None
        self.scale_polynomial_fit = None
        self.T_polynomial_fit = None
        self.secondary_electron_scale_fit = None
        self.Z = Z
        self.w = w
        self.k = k
        self.initial_a_estimate = initial_a_estimate
        self.initial_c_estimate = initial_c_estimate
        self.energy_conv_sigma = energy_conv_sigma
        self.temp = temp
        self.min_fit_count = min_fit_count

    def get_scale_T_trajectory(self, print_results=True, plot_fits=False):
        z_width = self.Z[0].size
        scale_trajectory = np.zeros(z_width)
        T_trajectory = np.zeros(z_width)
        scale_0, T_0, dk_0 = 1, 1, 1
        for i in range(z_width):
            low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index = \
                EDC_prep(i, self.Z, self.w, self.min_fit_count, exclude_secondary=False)

            params, pcov = scipy.optimize.curve_fit(
                partial(EDC_array, a=self.initial_a_estimate, c=self.initial_c_estimate, fixed_k=self.k[i],
                        energy_conv_sigma=self.energy_conv_sigma, temp=self.temp), low_noise_w, low_noise_slice,
                bounds=(
                    [0, 0, 0],
                    [ONE_BILLION, 75, 75]),
                p0=[scale_0, T_0, dk_0],
                sigma=fitting_sigma,
                maxfev=2000)
            scale_trajectory[i] = params[0]
            T_trajectory[i] = params[1]
            scale_0, T_0, dk_0 = params
            if plot_fits:
                plt.plot(low_noise_w, low_noise_slice)
                plt.plot(low_noise_w,
                         EDC_array(
                             low_noise_w, *params, self.initial_a_estimate, self.initial_c_estimate, self.k[i],
                             self.energy_conv_sigma, self.temp))
                plt.show()
            print(i)
        KDependentExtractor.scale_trajectory = scale_trajectory
        KDependentExtractor.T_trajectory = T_trajectory
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

    def get_secondary_electron_scale_trajectory(self, y_pos, plot=True):
        """
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

        polynomial_functions = [d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial,
                                d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial,
                                d10_polynomial]

        for i in range(9):
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k, self.scale_trajectory)
            outer_params, outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 1], self.k,
                                                                self.scale_trajectory)
            inner_fit = polynomial_functions[i](self.k, *inner_params)
            outer_fit = polynomial_functions[i + 1](self.k, *outer_params)
            if plot:
                plt.plot(self.k, self.scale_trajectory)
                plt.plot(self.k, inner_fit)
                plt.show()
            F_statistic = F_test(self.scale_trajectory, inner_fit, i + 1, outer_fit, i + 2,
                                 np.ones(len(self.scale_trajectory)),
                                 len(self.scale_trajectory))
            critical_value = scipy.stats.f.ppf(q=1 - 0.5, dfn=1, dfd=len(self.scale_trajectory) - (i + 2))
            if F_statistic < critical_value:
                print("Optimal scale polynomial is of degree: " + str(i + 1))
                print("Fitted scale parameters are: " + str(inner_params))
                self.scale_polynomial_degree = i + 1
                self.scale_polynomial_fit = inner_fit
                return inner_params, polynomial_functions[i]
        raise RuntimeError("Unable to find optimal scale polynomial fit")

    def get_T_polynomial_fit(self, plot=True):
        if KDependentExtractor.T_trajectory is None:
            raise AttributeError("Uninitialized T_trajectory.")
        polynomial_functions = [d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial,
                                d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial,
                                d10_polynomial]
        for i in range(9):
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k, self.T_trajectory)
            outer_params, outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 1], self.k,
                                                                self.T_trajectory)
            inner_fit = polynomial_functions[i](self.k, *inner_params)
            outer_fit = polynomial_functions[i + 1](self.k, *outer_params)
            if plot:
                plt.plot(self.k, self.T_trajectory)
                plt.plot(self.k, inner_fit)
                plt.show()
            F_statistic = F_test(self.T_trajectory, inner_fit, i + 1, outer_fit, i + 2,
                                 np.ones(len(self.T_trajectory)),
                                 len(self.T_trajectory))
            critical_value = scipy.stats.f.ppf(q=1 - 0.5, dfn=1, dfd=len(self.T_trajectory) - (i + 2))
            if F_statistic < critical_value:
                print("Optimal T polynomial is of degree: " + str(i + 1))
                print("Fitted T parameters are: " + str(inner_params))
                self.T_polynomial_degree = i + 1
                self.T_polynomial_fit = inner_fit
                return inner_params, polynomial_functions[i]
        raise RuntimeError("Unable to find optimal T polynomial fit")

    def get_secondary_electron_scale_polynomial_fit(self, plot=True):
        if KDependentExtractor.secondary_electron_scale_trajectory is None:
            raise AttributeError("Uninitialized secondary_electron_scale_trajectory.")
        polynomial_functions = [d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial,
                                d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial,
                                d10_polynomial]
        for i in range(9):
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k,
                                                                self.secondary_electron_scale_trajectory)
            outer_params, outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 1], self.k,
                                                                self.secondary_electron_scale_trajectory)
            inner_fit = polynomial_functions[i](self.k, *inner_params)
            outer_fit = polynomial_functions[i + 1](self.k, *outer_params)
            if plot:
                plt.plot(self.k, self.secondary_electron_scale_trajectory)
                plt.plot(self.k, inner_fit)
                plt.show()
            F_statistic = F_test(self.secondary_electron_scale_trajectory, inner_fit, i + 1, outer_fit, i + 2,
                                 np.ones(len(self.secondary_electron_scale_trajectory)),
                                 len(self.secondary_electron_scale_trajectory))
            critical_value = scipy.stats.f.ppf(q=1 - 0.5, dfn=1,
                                               dfd=len(self.secondary_electron_scale_trajectory) - (i + 2))
            if F_statistic < critical_value:
                print("Optimal secondary electron scale polynomial is of degree: " + str(i + 1))
                print("Fitted secondary electron scale parameters are: " + str(inner_params))
                self.secondary_electron_scale_degree = i + 1
                self.secondary_electron_scale_fit = inner_fit
                return inner_params, polynomial_functions[i]
        raise RuntimeError("Unable to find optimal secondary electron scale polynomial fit")

    def plot(self):
        if self.scale_polynomial_fit is None or self.T_polynomial_fit is None:
            raise NotImplementedError("Must get scale and/or T polynomial fits before plotting")
        plt.title("Scale")
        plt.plot(self.k, KDependentExtractor.scale_trajectory)
        plt.plot(self.k, self.scale_polynomial_fit)
        plt.show()
        plt.title("T")
        plt.plot(self.k, KDependentExtractor.T_trajectory)
        plt.plot(self.k, self.T_polynomial_fit)
        plt.show()
