from functools import partial
import lmfit
import math
import matplotlib.pyplot as plt
import numpy as np

from extraction_functions import EDC_prep, EDC_array_with_SE, symmetrize_EDC, Norman_EDC_array
from general import k_as_index, get_degree_polynomial


class Fitter:

    @staticmethod
    def NormanFit(Z, k, w, a_estimate, c_estimate, kf_index, energy_conv_sigma, temp):
        pars = lmfit.Parameters()
        pars.add('scale', value=70000, min=0, max=1000000)
        pars.add('T', value=7, min=0, max=25)
        pars.add('dk', value=10, min=0, max=25)
        pars.add('s', value=1500, min=1000, max=2000, vary=True)
        pars.add('a', value=a_estimate, min=a_estimate / 1.5, max=a_estimate * 1.5)
        pars.add('c', value=c_estimate, min=c_estimate * 1.5, max=c_estimate / 1.5)
        # low_noise_w, low_noise_slice, _, _, _, _ = EDC_prep(kf_index, Z, w, min_fit_count)
        low_noise_w = w
        low_noise_slice = [Z[i][kf_index] for i in range(len(w))]
        low_noise_w, low_noise_slice = symmetrize_EDC(low_noise_w, low_noise_slice)
        EDC_func = partial(Norman_EDC_array, fixed_k=math.fabs(k[kf_index]),
                           energy_conv_sigma=energy_conv_sigma, temp=temp)
        def calculate_residual(p):
            EDC_residual = EDC_func(
                np.asarray(low_noise_w), p['scale'], p['T'], p['dk'], p['s'], p['a'], p['c']) - low_noise_slice
            weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slice)
            return weighted_EDC_residual

        mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        print(lmfit.fit_report(result))
        scale = result.params.get('scale').value
        T = result.params.get('T').value
        dk = result.params.get('dk').value
        s = result.params.get('s').value
        a = result.params.get('a').value
        c = result.params.get('c').value
        plt.title("Norman fit")
        plt.plot()
        plt.plot(low_noise_w, low_noise_slice, label='data')
        plt.plot(low_noise_w, EDC_func(low_noise_w, scale, T, dk, s, a, c), label='fit')
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
                temp_dk = max(dk_estimate, self.energy_conv_sigma)
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

    def fit(self, scale_values, T_values, secondary_electron_scale_values, plot_results=True, hasty_fit=False):
        should_vary = not hasty_fit
        # Add parameters
        pars = lmfit.Parameters()
        for i in range(len(scale_values)):
            if scale_values[i] >= 0:
                pars.add('scale_' + str(i), value=scale_values[i],
                         # min=scale_values[i] / 1000, max=scale_values[i] * 1000,
                         vary=should_vary)
            else:
                pars.add('scale_' + str(i), value=scale_values[i],
                         # min=scale_values[i] * 1000, max=scale_values[i] / 1000,
                         vary=should_vary)
        for i in range(len(T_values)):
            if T_values[i] >= 0:
                pars.add('T_' + str(i), value=T_values[i],
                         # min=T_values[i] / 1000, max=T_values[i] * 1000,
                         vary=should_vary)
            else:
                pars.add('T_' + str(i), value=T_values[i],
                         # min=T_values[i] * 1000, max=T_values[i] / 1000,
                         vary=should_vary)
        for i in range(len(secondary_electron_scale_values)):
            if secondary_electron_scale_values[i] >= 0:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         # min=secondary_electron_scale_values[i] / 1000, max=secondary_electron_scale_values[i] * 1000,
                         vary=should_vary)
            else:
                pars.add('secondary_electron_scale_' + str(i), value=secondary_electron_scale_values[i],
                         # min=secondary_electron_scale_values[i] * 1000, max=secondary_electron_scale_values[i] / 1000,
                         vary=should_vary)
        pars.add('dk', value=self.dk_estimate, min=0, max=100, vary=True)
        pars.add('q', value=-7.5, max=0, vary=True)
        pars.add('r', value=0.5, min=0, vary=True)
        pars.add('s', value=250, vary=True)
        pars.add('a', value=self.a_estimate, min=min(self.a_estimate / 1.5, 1750), max=max(self.a_estimate * 1.5, 3000), vary=True)
        pars.add('c', value=self.c_estimate, min=min(self.c_estimate / 1.5, -50), max=max(self.c_estimate * 1.5, -15), vary=True)
        pars.add('k_error', value=0, min=-0.03, max=0.03, vary=True)

        # pars.add('dk', value=7.5733e-05, min=0, max=100, vary=True)
        # pars.add('q', value=-17.3566671, max=0, vary=True)
        # pars.add('r', value=0.13797258, min=0, vary=True)
        # pars.add('s', value=68.7445762, vary=True)
        # pars.add('a', value=1752.14770, min=0,
        #          vary=True)
        # pars.add('c', value=-38.2277233, max=0,
        #          vary=True)
        # pars.add('k_error', value=0, min=-0.03, max=0.03, vary=False)

        # Prepare EDCs to fit
        EDC_func_array = []
        low_noise_slices = []
        low_noise_ws = []
        for i in range(len(self.index_to_fit)):
            ki = self.index_to_fit[i]
            temp_low_noise_w, temp_low_noise_slice, _, _, _, _ = EDC_prep(ki, self.Z, self.w, self.min_fit_count, exclude_secondary=False)
            low_noise_ws.append(temp_low_noise_w)
            low_noise_slices.append(temp_low_noise_slice)
            EDC_func_array.append(partial(EDC_array_with_SE,
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
                    local_secondary_electron_scale, p['q'],
                    p['r'], p['s'], p['a'], p['c'], self.k[ki] - p['k_error']) - \
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
            lmfit_q = result.params.get('q').value
            lmfit_r = result.params.get('r').value
            lmfit_s = result.params.get('s').value
            lmfit_a = result.params.get('a').value
            lmfit_c = result.params.get('c').value
            lmfit_k_error = result.params.get('k_error')

            print("FINAL DK: ")
            print(lmfit_dk)

            for i in range(0, len(self.index_to_fit), int(len(self.index_to_fit) / 10)):
                ki = self.index_to_fit[i]
                plt.title("momentum: " + str(self.k[ki]))
                plt.plot(low_noise_ws[i], low_noise_slices[i], label='data')
                local_scale = get_degree_polynomial(len(scale_values))(self.k[ki], *lmfit_scale_params)
                local_T = get_degree_polynomial(len(T_values))(self.k[ki], *lmfit_T_params)
                local_secondary_electron_scale = get_degree_polynomial(len(secondary_electron_scale_values))(self.k[ki],
                                                                                                             *lmfit_secondary_electron_scale_params)
                plt.plot(low_noise_ws[i], EDC_func_array[i](low_noise_ws[i], local_scale, local_T, lmfit_dk,
                                                            local_secondary_electron_scale, lmfit_q, lmfit_r,
                                                            lmfit_s, lmfit_a, lmfit_c, self.k[ki] - lmfit_k_error), label='fit')
                plt.show()

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
        width = len(self.Z[0])
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
            # local_scale = 1000
            # local_secondary_electron_scale = 0
            # local_T = 10

            dk = 4.9076e-07
            q = -17.3558431
            r = 0.13776241
            s = 68.6551546
            a = 1750.73957
            c = -38.2093727
            k_error = -0.00386874

            EDC = EDC_array_with_SE(self.w, local_scale, local_T, dk, local_secondary_electron_scale, q, r,
                                    s, a, c, self.k[i] - k_error, self.energy_conv_sigma, self.temp)
            Z_fit_inverse[i] = EDC
        fitted_Z = np.array([list(i) for i in zip(*Z_fit_inverse)])
        if plot:
            plt.title("Fitted map")
            im = plt.imshow(fitted_Z, cmap=plt.cm.RdBu, aspect='auto', interpolation='nearest',
                            extent=[min(self.k), max(self.k), min(self.w), max(self.w)])  # drawing the function
            plt.colorbar(im)
            plt.show()
        return fitted_Z
