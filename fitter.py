from functools import partial
import lmfit
import math
import matplotlib.pyplot as plt
import numpy as np
import re

import scipy

from extraction_functions import EDC_prep, EDC_array_with_SE, symmetrize_EDC
from general import k_as_index, get_degree_polynomial


class Fitter:

    @staticmethod
    def NormanFit(Z, k, w, a_estimate, c_estimate, kf_index, energy_conv_sigma):
        # pars = lmfit.Parameters()
        # pars.add('scale', value=60000, min=0, max=1000000)
        # pars.add('T', value=20, min=0, max=30)
        # pars.add('dk', value=0, min=-25, max=25)
        # pars.add('s', value=1300, min=1000, max=2000, vary=True)
        # pars.add('a', value=a_estimate, min=a_estimate / 1.5, max=a_estimate * 1.5)
        # pars.add('c', value=c_estimate, min=c_estimate * 1.5, max=c_estimate / 1.5)
        # # low_noise_w, low_noise_slice, _, _, _, _ = EDC_prep(kf_index, Z, w, min_fit_count)
        low_noise_w = w
        low_noise_slice = [Z[i][kf_index] for i in range(len(w))]
        low_noise_w, low_noise_slice = symmetrize_EDC(low_noise_w, low_noise_slice)
        # EDC_func = partial(Norman_EDC_array, fixed_k=k[kf_index],
        #                    energy_conv_sigma=energy_conv_sigma)
        #
        # def calculate_residual(p):
        #     EDC_residual = EDC_func(
        #         np.asarray(low_noise_w), p['scale'], p['T'], p['dk'], p['s'], p['a'], p['c']) - low_noise_slice
        #     weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slice)
        #     return weighted_EDC_residual

        # mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        # result = mini.minimize(method='leastsq')
        # print(lmfit.fit_report(result))
        # scale = result.params.get('scale').value
        # T = result.params.get('T').value
        # dk = result.params.get('dk').value
        # s = result.params.get('s').value
        # a = result.params.get('a').value
        # c = result.params.get('c').value
        # plt.title("Norman fit")
        # plt.plot(low_noise_w, low_noise_slice, label='data')
        # plt.plot(low_noise_w, EDC_func(low_noise_w, scale, T, dk, s, a, c), label='fit')
        # plt.show()'

        import cmath

        def my_func(w, dk, T, T2, scale):
            sig_gap_num = complex(dk * dk, 0)
            if w > abs(dk) or w < -abs(dk):
                gamma0 = abs(T) + abs(T2) * cmath.sqrt(w ** 2 - T2 ** 2) / abs(w)
            else:
                gamma0 = abs(T)
            sig_gap_dom = complex(w, T2)
            sig_gap = sig_gap_num / sig_gap_dom
            sig = sig_gap - complex(0, 1) * gamma0
            Green = scale / (w - sig)
            return -Green.imag / np.pi

        mfv = np.vectorize(my_func)  # dk, T, T2, scale
        params, pcov = scipy.optimize.curve_fit(mfv, low_noise_w, low_noise_slice, p0=[5, 15, 0, 3e+05], sigma=np.sqrt(low_noise_slice), maxfev=2000)
        plt.plot(low_noise_w, low_noise_slice, label='data')
        plt.plot(low_noise_w, mfv(low_noise_w, *params), label='fit')
        plt.show()
        print(str(params[0]) + " +- " + str(np.sqrt(np.diag(pcov))[0]))

    @staticmethod
    def get_fitted_map(fileName, k, w, energy_conv_sigma, temperature, second_fit=False):
        data_file = open(fileName, "r")
        while '[[Variables]]' not in data_file.readline():
            pass
        if second_fit:
            while '[[Variables]]' not in data_file.readline():
                pass

        def extract_value(s):
            # regex for number
            match_number = re.compile('[-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?')
            return float(re.findall(match_number, s[s.index(':'):])[0])

        # Read scale parameters
        scale_values = []
        while True:
            temp = data_file.readline()
            if 'scale_' not in temp:
                break
            value = extract_value(temp)
            scale_values.append(value)
        # Read T params
        T_values = []
        while True:
            if 'T_' not in temp:
                break
            value = extract_value(temp)
            T_values.append(value)
            temp = data_file.readline()
        # Read secondary_electron_scale params
        secondary_electron_scale_values = []
        while True:
            if 'secondary_electron_scale_' not in temp:
                break
            value = extract_value(temp)
            secondary_electron_scale_values.append(value)
            temp = data_file.readline()
        dk = extract_value(temp)
        q = extract_value(data_file.readline())
        r = extract_value(data_file.readline())
        s = extract_value(data_file.readline())
        a = extract_value(data_file.readline())
        c = extract_value(data_file.readline())
        k_error = extract_value(data_file.readline())

        height = len(w)
        width = len(k)
        Z_fit_inverse = np.zeros((width, height))
        scale_polynomial = get_degree_polynomial(len(scale_values))
        T_polynomial = get_degree_polynomial(len(T_values))
        secondary_electron_scale_polynomial = get_degree_polynomial(len(secondary_electron_scale_values))
        for i in range(width):
            local_k = k[i] - k_error
            local_scale = scale_polynomial(local_k, *scale_values)
            local_T = T_polynomial(local_k, *T_values)
            local_secondary_electron_scale = secondary_electron_scale_polynomial(local_k,
                                                                                 *secondary_electron_scale_values)

            EDC = EDC_array_with_SE(w, local_scale, local_T, dk, local_secondary_electron_scale, q, r,
                                    s, a, c, local_k, energy_conv_sigma, temperature)
            Z_fit_inverse[i] = EDC
        fitted_Z = np.array([list(i) for i in zip(*Z_fit_inverse)])
        plt.title("Fitted map")
        im = plt.imshow(fitted_Z, cmap=plt.cm.RdBu, aspect='auto', interpolation='nearest',
                        extent=[min(k), max(k), min(w), max(w)])  # drawing the function
        plt.colorbar(im)

        plt.show()
        return fitted_Z

    @staticmethod
    def relative_error_map(Z, fitted_Z, k, w, DOF):
        error_map = np.abs(fitted_Z - Z)
        relative_error_map = error_map / np.sqrt(Z)
        plt.title("Relative error map")
        im = plt.imshow(relative_error_map, cmap=plt.cm.RdBu, aspect='auto',
                        extent=[min(k), max(k), min(w), max(w)])  # drawing the function
        plt.colorbar(im)
        plt.show()

        redchi = np.sum(error_map * error_map / Z) / DOF
        print("Reduced chi: " + str(redchi))

    def __init__(self, Z, k, w, a_estimate, c_estimate, dk_estimate, kf_estimate, temp, energy_conv_sigma, EDC_density,
                 fit_range_multiplier=2, override_index_to_fit=None, min_fit_count=25):
        self.Z = Z
        self.k = k
        self.w = w
        self.a_estimate = a_estimate
        self.c_estimate = c_estimate
        self.dk_estimate = dk_estimate
        self.energy_conv_sigma = energy_conv_sigma
        self.EDC_density = EDC_density
        self.temp = temp
        self.fit_range_multiplier = fit_range_multiplier
        self.min_fit_count = min_fit_count
        if override_index_to_fit is not None:
            self.index_to_fit = override_index_to_fit
        else:
            try:
                temp_dk = max(dk_estimate, self.energy_conv_sigma / 3)
                fit_start_k = math.sqrt(
                    (-fit_range_multiplier * temp_dk - c_estimate) / a_estimate)
            except ValueError:
                fit_start_k = 0
                print("Able to fit momenta through k=0")
            a = max(k_as_index(-kf_estimate, k), 0)
            b = k_as_index(-fit_start_k, k)
            c = k_as_index(fit_start_k, k)
            d = min(k_as_index(kf_estimate, k), k.size)
            if b == c:
                self.index_to_fit = [*range(a, d, EDC_density)]
            else:
                self.index_to_fit = [*range(a, b, EDC_density)] + [*range(c, d, EDC_density)]

    def fit(self, scale_values, T0_values, T1_values, secondary_electron_scale_values, plot_results=True, kdependent_fixed=False,
            ac_fixed=False, dk_0_fixed=False, q_estimate=-7.5, r_estimate=0.5, s_estimate=250, k_error_estimate=0):
        print("Fitting: " + str(self.index_to_fit))

        should_vary_kdependent = not kdependent_fixed
        should_vary_ac = not ac_fixed
        # Add parameters
        pars = lmfit.Parameters()
        for i in range(len(scale_values)):
            if scale_values[i] >= 0:
                pars.add('scale_' + str(i), value=scale_values[i],
                         # min=scale_values[i] / 1000, max=scale_values[i] * 1000,
                         vary=should_vary_kdependent)
            else:
                pars.add('scale_' + str(i), value=scale_values[i],
                         # min=scale_values[i] * 1000, max=scale_values[i] / 1000,
                         vary=should_vary_kdependent)
        for i in range(len(T0_values)):
            if T0_values[i] >= 0:
                pars.add('T0_' + str(i), value=T0_values[i],
                         # min=T0_values[i] / 1000, max=T0_values[i] * 1000,
                         vary=should_vary_kdependent)
            else:
                pars.add('T0_' + str(i), value=T0_values[i],
                         # min=T0_values[i] * 1000, max=T0_values[i] / 1000,
                         vary=should_vary_kdependent)
        for i in range(len(T1_values)):
            if T1_values[i] >= 0:
                pars.add('T1_' + str(i), value=T1_values[i],
                         # min=T1_values[i] / 1000, max=T1_values[i] * 1000,
                         vary=should_vary_kdependent)
            else:
                pars.add('T1_' + str(i), value=T1_values[i],
                         # min=T1_values[i] * 1000, max=T1_values[i] / 1000,
                         vary=should_vary_kdependent)
        for i in range(len(secondary_electron_scale_values)):
            if secondary_electron_scale_values[i] >= 0:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         # min=secondary_electron_scale_values[i] / 1000, max=secondary_electron_scale_values[i] * 1000,
                         vary=should_vary_kdependent)
            else:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         # min=secondary_electron_scale_values[i] * 1000, max=secondary_electron_scale_values[i] / 1000,
                         vary=should_vary_kdependent)
        pars.add('dk', value=(0 if dk_0_fixed else self.dk_estimate), min=0, max=100, vary=(not dk_0_fixed))
        pars.add('q', value=q_estimate, max=0, vary=True)
        pars.add('r', value=r_estimate, min=0, vary=True)
        pars.add('s', value=s_estimate, vary=True)
        pars.add('a', value=self.a_estimate, min=min(self.a_estimate / 1.5, 1750), max=max(self.a_estimate * 1.5, 3000),
                 vary=should_vary_ac)
        pars.add('c', value=self.c_estimate, min=min(self.c_estimate / 1.5, -50), max=max(self.c_estimate * 1.5, -15),
                 vary=should_vary_ac)
        pars.add('k_error', value=k_error_estimate, min=-0.03, max=0.03, vary=True)

        # Prepare EDCs to fit
        EDC_func_array = []
        low_noise_slices = []
        low_noise_ws = []
        for i in range(len(self.index_to_fit)):
            ki = self.index_to_fit[i]
            temp_low_noise_w, temp_low_noise_slice, _, _, _, _ = EDC_prep(ki, self.Z, self.w, self.min_fit_count,
                                                                          exclude_secondary=False)
            low_noise_ws.append(temp_low_noise_w)
            low_noise_slices.append(temp_low_noise_slice)
            EDC_func_array.append(partial(EDC_array_with_SE,
                                          energy_conv_sigma=self.energy_conv_sigma, temp=self.temp))
        # Fetch local polynomials
        scale_polynomial = get_degree_polynomial(len(scale_values))
        T0_polynomial = get_degree_polynomial(len(T0_values))
        T1_polynomial = get_degree_polynomial(len(T1_values))
        secondary_electron_scale_polynomial = get_degree_polynomial(len(secondary_electron_scale_values))

        def calculate_residual(p):
            residual = np.zeros(0)
            scale_polynomial_params = [p['scale_' + str(i)] for i in range(len(scale_values))]
            T0_polynomial_params = [p['T0_' + str(i)] for i in range(len(T0_values))]
            T1_polynomial_params = [p['T1_' + str(i)] for i in range(len(T1_values))]
            secondary_electron_scale_params = [p['secondary_electron_scale_' + str(i)] for i in
                                               range(len(secondary_electron_scale_values))]
            for i in range(len(self.index_to_fit)):
                ki = self.index_to_fit[i]
                local_k = self.k[ki] - p['k_error']
                local_scale = scale_polynomial(local_k, *scale_polynomial_params)
                local_T0 = T0_polynomial(local_k, *T0_polynomial_params)
                local_T1 = T1_polynomial(local_k, *T1_polynomial_params)
                local_secondary_electron_scale = secondary_electron_scale_polynomial(local_k,
                                                                                     *secondary_electron_scale_params)

                EDC_residual = EDC_func_array[i](
                    low_noise_ws[i], local_scale, local_T0, local_T1, p['dk'],
                    local_secondary_electron_scale, p['q'],
                    p['r'], p['s'], p['a'], p['c'], local_k) - \
                               low_noise_slices[i]
                weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slices[i])
                residual = np.concatenate((residual, weighted_EDC_residual))
            return residual

        mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        print(lmfit.fit_report(result))

        lmfit_scale_params = [result.params.get('scale_' + str(i)).value for i in range(len(scale_values))]
        lmfit_T0_params = [result.params.get('T0_' + str(i)).value for i in range(len(T0_values))]
        lmfit_T1_params = [result.params.get('T1_' + str(i)).value for i in range(len(T1_values))]
        lmfit_secondary_electron_scale_params = [result.params.get('secondary_electron_scale_' + str(i)).value for i
                                                 in range(len(secondary_electron_scale_values))]
        lmfit_dk = result.params.get('dk').value
        lmfit_q = result.params.get('q').value
        lmfit_r = result.params.get('r').value
        lmfit_s = result.params.get('s').value
        lmfit_a = result.params.get('a').value
        lmfit_c = result.params.get('c').value
        lmfit_k_error = result.params.get('k_error').value
        lmfit_T0 = result.params.get('T0').value

        if plot_results:
            print("FINAL DK: ")
            print(lmfit_dk)
            for i in range(0, len(self.index_to_fit), max(int(len(self.index_to_fit) / 10), 1)):
                ki = self.index_to_fit[i]
                plt.title("momentum: " + str(self.k[ki]))
                plt.plot(low_noise_ws[i], low_noise_slices[i], label='data')
                local_scale = get_degree_polynomial(len(scale_values))(self.k[ki], *lmfit_scale_params)
                local_T0 = get_degree_polynomial(len(T0_values))(self.k[ki], *lmfit_T0_params)
                local_T1 = get_degree_polynomial(len(T1_values))(self.k[ki], *lmfit_T1_params)
                local_secondary_electron_scale = get_degree_polynomial(len(secondary_electron_scale_values))(self.k[ki],
                                                                                                             *lmfit_secondary_electron_scale_params)
                plt.plot(low_noise_ws[i], EDC_func_array[i](low_noise_ws[i], local_scale, local_T0, local_T1, lmfit_dk,
                                                            local_secondary_electron_scale, lmfit_q, lmfit_r,
                                                            lmfit_s, lmfit_a, lmfit_c, self.k[ki] - lmfit_k_error),
                         label='fit')
                plt.show()
        return lmfit_scale_params, lmfit_T0_params, lmfit_T1_params, lmfit_secondary_electron_scale_params, lmfit_dk, lmfit_q, lmfit_r, lmfit_s, lmfit_a, lmfit_c, lmfit_k_error
