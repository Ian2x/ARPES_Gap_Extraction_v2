from functools import partial
import lmfit
import math
import matplotlib.pyplot as plt
import numpy as np
from extraction_functions import symmetrize_EDC, Norman_EDC_array, Norman_EDC_array2


class Fitter:
    @staticmethod
    def NormanFit(Z, k, w, k_index, energy_conv_sigma, simulated, params=None, print_results=False, plot_results=False):
        # w, EDC_slice, _, _, _, _ = EDC_prep(k_index, Z, w, min_fit_count)
        EDC_slice = [Z[i][k_index] for i in range(len(w))]
        w, EDC_slice = symmetrize_EDC(w, EDC_slice)
        while w[0] > (35 if simulated else 45):
            w = w[1:]
            EDC_slice = EDC_slice[1:]
        while w[-1] < (-35 if simulated else -45):
            w = w[:-1]
            EDC_slice = EDC_slice[:-1]
        pars = lmfit.Parameters()
        if simulated:
            pars.add('a', value=params[0] if params is not None else 3e+06, min=0, vary=True)
            pars.add('b', value=params[1] if params is not None else -24, min=-50, max=0.1, vary=True)
            pars.add('c', value=params[2] if params is not None else 8, min=0, max=np.inf, vary=True)
            pars.add('s', value=params[3] if params is not None else 350000, min=0, max=np.inf, vary=True)

            EDC_func = partial(Norman_EDC_array2, energy_conv_sigma=energy_conv_sigma, noConvolute=True)

            def calculate_residual(p):
                EDC_residual = EDC_func(np.asarray(w), p['a'], p['b'], p['c'], p['s']) - EDC_slice
                weighted_EDC_residual = EDC_residual / np.sqrt(EDC_slice)
                return weighted_EDC_residual
        else:
            pars.add('scale', value=params[0] if params is not None else 35000, min=0, vary=True)
            pars.add('loc', value=0, min=-100, max=0.1, vary=False)
            pars.add('dk', value=params[2] if params is not None else 10, min=-75, max=75, vary=True)
            pars.add('T1', value=params[3] if params is not None else 15, min=0, max=75, vary=True)
            pars.add('T0', value=params[4] if params is not None else 0, min=0, max=75, vary=True)
            pars.add('s', value=params[5] if params is not None else 1000, min=0, max=np.inf, vary=True)

            EDC_func = partial(Norman_EDC_array, energy_conv_sigma=energy_conv_sigma)

            def calculate_residual(p):
                EDC_residual = EDC_func(
                    np.asarray(w), p['scale'], p['loc'], p['dk'], p['T1'], p['T0'], p['s']) - EDC_slice
                weighted_EDC_residual = EDC_residual / np.sqrt(EDC_slice)
                return weighted_EDC_residual

        mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='least_squares', xtol=1e-07 if simulated else 1e-05)

        if simulated:
            dk = result.params.get('b').value
            dk_err = result.params.get('b').stderr
        else:
            dk = result.params.get('dk').value
            dk_err = result.params.get('dk').stderr

        if print_results:
            print(lmfit.fit_report(result))
            print(np.abs(dk), ",", dk_err, ",", result.redchi)

        if plot_results:
            plt.title("Norman fit (k=" + str(k[k_index])+")")
            plt.plot(w, EDC_slice, label='data')
            plt.plot(w, EDC_func(w, *result.params.values()), label='fit')
            plt.show()

        return np.abs(dk), dk_err, result.redchi, list(result.params.values())
