from functools import partial
import lmfit
import math
import matplotlib.pyplot as plt
import numpy as np

from extraction_functions import EDC_prep, spectrum_slice_array_SEC, spectrum_slice_array_2, spectrum_slice_array_SEC_2
from general import k_as_index, get_degree_polynomial


class Fitter:

    @staticmethod
    def NormanFit(Z, k, w, a_estimate, c_estimate, kf_index, energy_conv_sigma, temp, min_fit_count=25):
        pars = lmfit.Parameters()
        pars.add('scale', value=70000, min=60000, max=80000)
        pars.add('T', value=7, min=0, max=25)
        pars.add('dk', value=10, min=0, max=25)
        pars.add('p', value=1000, min=500, max=1500)
        pars.add('q', value=-10, min=-20, max=0)
        pars.add('r', value=0.5, min=0, max=1)
        pars.add('s', value=400, min=200, max=600)
        pars.add('a', value=a_estimate, min=a_estimate / 1.25, max=a_estimate * 1.25)
        pars.add('c', value=c_estimate, min=c_estimate * 1.25, max=c_estimate / 1.25)
        # low_noise_w, low_noise_slice, _, _, _, _ = EDC_prep(kf_index, Z, w, min_fit_count)
        low_noise_w = w
        low_noise_slice = [Z[i][kf_index] for i in range(len(w))]
        EDC_func = partial(spectrum_slice_array_SEC_2, fixed_k=math.fabs(k[kf_index]),
                           energy_conv_sigma=energy_conv_sigma, temp=temp)
        def calculate_residual(p):
            EDC_residual = EDC_func(
                low_noise_w, p['scale'], p['T'], p['dk'], p['p'], p['q'], p['r'], p['s'], p['a'], p['c']) - low_noise_slice
            weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slice)
            return weighted_EDC_residual

        mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        print(lmfit.fit_report(result))
        scale = result.params.get('scale').value
        T = result.params.get('T').value
        dk = result.params.get('dk').value
        p = result.params.get('p').value
        q = result.params.get('q').value
        r = result.params.get('r').value
        s = result.params.get('s').value
        a = result.params.get('a').value
        c = result.params.get('c').value
        plt.title("Norman fit")
        plt.plot()
        plt.plot(low_noise_w, low_noise_slice, label='data')
        plt.plot(low_noise_w, EDC_func(low_noise_w, scale, T, dk, p, q, r, s, a, c), label='fit')
        plt.show()

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
                fit_start_k = math.sqrt(
                    (-fit_range_multiplier * dk_estimate - c_estimate) / a_estimate)
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

    def fit(self, scale_values, T_values, secondary_electron_scale_values, plot_results=True):
        # Add parameters
        pars = lmfit.Parameters()
        for i in range(len(scale_values)):
            if scale_values[i] >= 0:
                pars.add('scale_' + str(i), value=scale_values[i], min=scale_values[i] / 10, max=scale_values[i] * 10,
                         vary=True)
            else:
                pars.add('scale_' + str(i), value=scale_values[i], min=scale_values[i] * 10, max=scale_values[i] / 10,
                         vary=True)
        for i in range(len(T_values)):
            if T_values[i] >= 0:
                pars.add('T_' + str(i), value=T_values[i], min=T_values[i] / 10, max=T_values[i] * 10, vary=True)
            else:
                pars.add('T_' + str(i), value=T_values[i], min=T_values[i] * 10, max=T_values[i] / 10, vary=True)
        for i in range(len(secondary_electron_scale_values)):
            if secondary_electron_scale_values[i] >= 0:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         min=secondary_electron_scale_values[i] / 10, max=secondary_electron_scale_values[i] * 10,
                         vary=True)
            else:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         min=secondary_electron_scale_values[i] * 10, max=secondary_electron_scale_values[i] / 10,
                         vary=True)
        pars.add('dk', value=15, min=0, max=75)
        pars.add('p', value=1, min=0)
        pars.add('q', value=-1, max=0)
        pars.add('r', value=1, min=0)
        pars.add('s', value=0)
        pars.add('a', value=self.a_estimate, min=0, max=self.a_estimate * 3)
        pars.add('c', value=self.c_estimate, min=self.c_estimate * 3, max=0)

        # Prepare EDCs to fit
        EDC_func_array = []
        low_noise_slices = []
        low_noise_ws = []
        for i in range(len(self.index_to_fit)):
            ki = self.index_to_fit[i]
            temp_low_noise_w, temp_low_noise_slice, _, _, _, _ = EDC_prep(ki, self.Z, self.w, self.min_fit_count, exclude_secondary=False)
            low_noise_ws.append(temp_low_noise_w)
            low_noise_slices.append(temp_low_noise_slice)
            EDC_func_array.append(partial(spectrum_slice_array_SEC, fixed_k=math.fabs(self.k[ki]),
                                          energy_conv_sigma=self.energy_conv_sigma, temp=self.temp))
        # Fetch local polynomials
        scale_polynomial = get_degree_polynomial(len(scale_values))
        T_polynomial = get_degree_polynomial(len(T_values))
        secondary_electron_scale_polynomial = get_degree_polynomial(len(secondary_electron_scale_values))

        def calculate_residual(p):
            residual = np.zeros(0)
            scale_polynomial_params = [p['scale_' + str(i)] for i in range(len(scale_values))]
            T_polynomial_params = [p['T_' + str(i)] for i in range(len(T_values))]
            secondary_electron_scale_params = [p['secondary_electron_scale_' + str(i)] for i in
                                               range(len(secondary_electron_scale_values))]
            for i in range(len(self.index_to_fit)):
                ki = self.index_to_fit[i]
                local_scale = scale_polynomial(self.k[ki], *scale_polynomial_params)
                local_T = T_polynomial(self.k[ki], *T_polynomial_params)
                local_secondary_electron_scale = secondary_electron_scale_polynomial(self.k[ki],
                                                                                     *secondary_electron_scale_params)

                EDC_residual = EDC_func_array[i](
                    low_noise_ws[i], local_scale, local_T, p['dk'],
                    local_secondary_electron_scale * p['p'], p['q'],
                    p['r'], p['s'], p['a'], p['c']) - \
                               low_noise_slices[i]
                weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slices[i])
                residual = np.concatenate((residual, weighted_EDC_residual))
            return residual

        mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        print(lmfit.fit_report(result))

        if plot_results:
            lmfit_scale_params = [result.params.get('scale_' + str(i)).value for i in range(len(scale_values))]
            lmfit_T_params = [result.params.get('T_' + str(i)).value for i in range(len(T_values))]
            lmfit_secondary_electron_scale_params = [result.params.get('secondary_electron_scale_' + str(i)).value for i
                 in range(len(secondary_electron_scale_values))]
            lmfit_dk = result.params.get('dk').value
            lmfit_p = result.params.get('p').value
            lmfit_q = result.params.get('q').value
            lmfit_r = result.params.get('r').value
            lmfit_s = result.params.get('s').value
            lmfit_a = result.params.get('a').value
            lmfit_c = result.params.get('c').value

            print("FINAL DK: ")
            print(lmfit_dk)

            for i in range(len(self.index_to_fit)):
                ki = self.index_to_fit[i]
                plt.title("momentum: " + str(self.k[ki]))
                plt.plot(low_noise_ws[i], low_noise_slices[i], label='data')
                local_scale = get_degree_polynomial(len(scale_values))(self.k[ki], *lmfit_scale_params)
                local_T = get_degree_polynomial(len(T_values))(self.k[ki], *lmfit_T_params)
                local_secondary_electron_scale = get_degree_polynomial(len(secondary_electron_scale_values))(self.k[ki],
                                                                                                             *lmfit_secondary_electron_scale_params)
                plt.plot(low_noise_ws[i], EDC_func_array[i](low_noise_ws[i], local_scale, local_T, lmfit_dk,
                                                            local_secondary_electron_scale * lmfit_p, lmfit_q, lmfit_r,
                                                            lmfit_s, lmfit_a, lmfit_c), label='fit')
                plt.show()

    '''
    def fit(self, scale_values, T_values, secondary_electron_scale_values, plot_results=True):
        # TODO: normalize parameters
        sc_n = [10 ** (-9), 10 ** (-7), 10 ** (-7), 10 ** (-5), 10 ** (-5)]
        T_n = [10 ** (-10), 10 ** (-8), 10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-4), 10 ** (-4), 10 ** (-2),
               10 ** (-2)]
        se_n = [10 ** (-9), 10 ** (-7), 10 ** (-7), 10 ** (-5), 10 ** (-5), 10 ** (-3), 10 ** (-3)]
        dk_n = 10 ** (-2)
        p_n = 1
        q_n = 10 ** (-1)
        r_n = 1
        s_n = 1
        a_n = 10 ** (-4)
        c_n = 10 ** (-2)

        scale_values = np.multiply(scale_values, sc_n)
        T_values = np.multiply(T_values, T_n)
        secondary_electron_scale_values = np.multiply(secondary_electron_scale_values, se_n)

        # Add parameters
        pars = lmfit.Parameters()
        for i in range(len(scale_values)):
            if scale_values[i] >= 0:
                pars.add('scale_' + str(i), value=scale_values[i], min=scale_values[i] / 10, max=scale_values[i] * 10,
                         vary=True)
            else:
                pars.add('scale_' + str(i), value=scale_values[i], min=scale_values[i] * 10, max=scale_values[i] / 10,
                         vary=True)
        for i in range(len(T_values)):
            if T_values[i] >= 0:
                pars.add('T_' + str(i), value=T_values[i], min=T_values[i] / 10, max=T_values[i] * 10, vary=True)
            else:
                pars.add('T_' + str(i), value=T_values[i], min=T_values[i] * 10, max=T_values[i] / 10, vary=True)
        for i in range(len(secondary_electron_scale_values)):
            if secondary_electron_scale_values[i] >= 0:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         min=secondary_electron_scale_values[i] / 10, max=secondary_electron_scale_values[i] * 10,
                         vary=True)
            else:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         min=secondary_electron_scale_values[i] * 10, max=secondary_electron_scale_values[i] / 10,
                         vary=True)
        pars.add('dk', value=15 * dk_n, min=0, max=75 * dk_n)
        pars.add('p', value=1 * p_n, min=0)
        pars.add('q', value=-1 * q_n, max=0)
        pars.add('r', value=1 * r_n, min=0)
        pars.add('s', value=0 * s_n)
        pars.add('a', value=self.a_estimate * a_n, min=0, max=self.a_estimate * 3 * a_n)
        pars.add('c', value=self.c_estimate * c_n, min=self.c_estimate * 3 * c_n, max=0)

        # Prepare EDCs to fit
        EDC_func_array = []
        low_noise_slices = []
        low_noise_ws = []
        for i in range(len(self.index_to_fit)):
            ki = self.index_to_fit[i]
            temp_low_noise_w, temp_low_noise_slice, _, _, _, _ = EDC_prep(ki, self.Z, self.w, self.min_fit_count)
            low_noise_ws.append(temp_low_noise_w)
            low_noise_slices.append(temp_low_noise_slice)
            EDC_func_array.append(partial(spectrum_slice_array_SEC, fixed_k=math.fabs(self.k[ki]),
                                          energy_conv_sigma=self.energy_conv_sigma, temp=self.temp))
        print(self.index_to_fit[len(self.index_to_fit)-1])
        print(self.k[self.index_to_fit[len(self.index_to_fit)-1]])
        quit()
        # Fetch local polynomials
        scale_polynomial = get_degree_polynomial(len(scale_values))
        T_polynomial = get_degree_polynomial(len(T_values))
        secondary_electron_scale_polynomial = get_degree_polynomial(len(secondary_electron_scale_values))

        def calculate_residual(p):
            residual = np.zeros(0)
            scale_polynomial_params = [p['scale_' + str(i)] for i in range(len(scale_values))]
            T_polynomial_params = [p['T_' + str(i)] for i in range(len(T_values))]
            secondary_electron_scale_params = [p['secondary_electron_scale_' + str(i)] for i in
                                               range(len(secondary_electron_scale_values))]
            for i in range(len(self.index_to_fit)):
                ki = self.index_to_fit[i]
                local_scale = scale_polynomial(self.k[ki], *np.divide(scale_polynomial_params, sc_n))
                local_T = T_polynomial(self.k[ki], *np.divide(T_polynomial_params, T_n))
                local_secondary_electron_scale = secondary_electron_scale_polynomial(self.k[ki],
                                                                                     *np.divide(
                                                                                         secondary_electron_scale_params,
                                                                                         se_n))

                EDC_residual = EDC_func_array[i](
                    low_noise_ws[i], local_scale, local_T, p['dk'] / dk_n,
                                                           local_secondary_electron_scale * p['p'] / p_n, p['q'] / q_n,
                                                           p['r'] / r_n, p['s'] / s_n, p['a'] / a_n, p['c'] / c_n) - \
                               low_noise_slices[i]
                weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slices[i])
                residual = np.concatenate((residual, weighted_EDC_residual))
            return residual

        mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        print(lmfit.fit_report(result))

        if plot_results:
            lmfit_scale_params = np.divide([result.params.get('scale_' + str(i)).value for i in range(len(scale_values))], sc_n)
            lmfit_T_params = np.divide([result.params.get('T_' + str(i)).value for i in range(len(T_values))], T_n)
            lmfit_secondary_electron_scale_params = np.divide([result.params.get('secondary_electron_scale_' + str(i)).value for i
                                                     in range(len(secondary_electron_scale_values))], se_n)
            lmfit_dk = result.params.get('dk').value / dk_n
            lmfit_p = result.params.get('p').value / p_n
            lmfit_q = result.params.get('q').value / q_n
            lmfit_r = result.params.get('r').value / r_n
            lmfit_s = result.params.get('s').value / s_n
            lmfit_a = result.params.get('a').value / a_n
            lmfit_c = result.params.get('c').value / c_n

            print("FINAL DK: ")
            print(lmfit_dk)

            for i in range(len(self.index_to_fit)):
                ki = self.index_to_fit[i]
                plt.title("momentum: " + str(self.k[ki]))
                plt.plot(low_noise_ws[i], low_noise_slices[i], label='data')
                local_scale = get_degree_polynomial(len(scale_values))(self.k[ki], *lmfit_scale_params)
                local_T = get_degree_polynomial(len(T_values))(self.k[ki], *lmfit_T_params)
                local_secondary_electron_scale = get_degree_polynomial(len(secondary_electron_scale_values))(self.k[ki],
                                                                                                             *lmfit_secondary_electron_scale_params)
                plt.plot(low_noise_ws[i], EDC_func_array[i](low_noise_ws[i], local_scale, local_T, lmfit_dk,
                                                            local_secondary_electron_scale * lmfit_p, lmfit_q, lmfit_r,
                                                            lmfit_s, lmfit_a, lmfit_c), label='fit')
                plt.show()
    '''

    def relative_error_map(self, fitted_Z):
        error_map = np.abs(fitted_Z - self.Z)
        relative_error_map = error_map / np.sqrt(self.Z)
        plt.title("Relative error map")
        im = plt.imshow(relative_error_map, cmap=plt.cm.RdBu, aspect='auto',
                        extent=[min(self.k), max(self.k), min(self.w), max(self.w)])  # drawing the function
        plt.colorbar(im)
        plt.show()

    def get_fitted_map(self, scale_values, T_values, secondary_electron_scale_values, show_progress=False, plot=True):
        """
        Requires manual editing of dk, p, q, r, s, a, c
        """
        height = len(self.Z)
        width = self.Z[0].size
        Z_fit_inverse = np.zeros((width, height))
        scale_polynomial = get_degree_polynomial(len(scale_values))
        T_polynomial = get_degree_polynomial(len(T_values))
        secondary_electron_scale_polynomial = get_degree_polynomial(len(secondary_electron_scale_values))
        for i in range(width):
            if show_progress:
                print(str(i) + " of " + str(width))
            local_scale = scale_polynomial(self.k[i], *scale_values)
            local_T = T_polynomial(self.k[i], *T_values)
            local_secondary_electron_scale = secondary_electron_scale_polynomial(self.k[i],
                                                                                 *secondary_electron_scale_values)
            dk = 11.5042512 # 0.11616861e+02  # 15.7193965
            p = 1.95276927 # 1.05881374  # 0.10047314
            q = -7.73249535 # -0.77553310e+01  # -7.13989334
            r = 0.45174549 # 0.45410424  # 3.46865626
            s = 244.609464 # 69.4525201  # -3.35522709
            a = 2562.94584 # 0.26037032e+04  # 2338.17857
            c = -22.2660626 # -0.22246008e+02  # -18.7353912

            EDC = spectrum_slice_array_SEC(self.w, local_scale, local_T, dk, local_secondary_electron_scale * p, q, r,
                                           s, a, c, self.k[i], self.energy_conv_sigma, self.temp)
            Z_fit_inverse[i] = EDC
        fitted_Z = np.array([list(i) for i in zip(*Z_fit_inverse)])
        if plot:
            plt.title("Fitted map")
            im = plt.imshow(fitted_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(self.k), max(self.k), min(self.w), max(self.w)])  # drawing the function
            plt.colorbar(im)
            plt.show()
        return fitted_Z


'''
            dk: 13.9502163 + / - 1.09874485(7.88 %)(init=15)
            p: 168.380391 + / - 21.0376189(12.49 %)(init=48)
            q: -8.64849864 + / - 0.29685599(3.43 %)(init=-18)
            r: 0.84498801 + / - 0.19696256(23.31 %)(init=0.38)
            s: -4.2384e-07 + / - 8.56175077(2020037063.72 %)(init=-14)
            a: 2536.60714 + / - 193.823970(7.64 %)(init=2555.655)
            c: -20.8313901 + / - 0.67114654(3.22 %)(init=-20.34252)
            '''
