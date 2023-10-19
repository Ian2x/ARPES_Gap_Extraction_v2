from functools import partial
import lmfit
import matplotlib.pyplot as plt
import numpy as np

from data_reader import FileType
from extraction_functions import symmetrize_EDC, Norman_EDC_array, Norman_EDC_array2


class Fitter:
    @staticmethod
    def NormanFit(Z, k, w, k_index, energy_conv_sigma, fileType, params=None, plot=False, force_b_zero=True):
        EDC_slice = [Z[i][k_index] for i in range(len(w))]
        w, EDC_slice = symmetrize_EDC(w, EDC_slice)
        while w[0] > (35 if fileType == FileType.SIMULATED else (25 if fileType == FileType.ANTI_NODE else 45)):
            w = w[1:]
            EDC_slice = EDC_slice[1:]
        while w[-1] < (-35 if fileType == FileType.SIMULATED else (-25 if fileType == FileType.ANTI_NODE else -45)):
            w = w[:-1]
            EDC_slice = EDC_slice[:-1]
        pars = lmfit.Parameters()
        if fileType == FileType.SIMULATED:

            # if params is not None:
            #     params[1] = -0.1

            pars.add('a', value=params[0] if params is not None else 1900, min=0, vary=True)
            pars.add('b', value=0 if force_b_zero else (params[1] if params is not None else -18), min=-50, max=1,
                     vary=True)  # USE 0 GAP IF REDCHI IS NOT MUCH WORSE # 0 if params is not None else -24
            pars.add('c', value=params[2] if params is not None else 11, min=0, max=np.inf, vary=True)
            pars.add('s', value=params[3] if params is not None else 600, min=0, max=np.inf, vary=True)

            # pars.add('a', value=params[0] if params is not None else 1900, min=0, vary=True)
            # pars.add('b', value=0 if force_b_zero else (params[1] if params is not None else -24), min=-50, max=1, vary=True)  # USE 0 GAP IF REDCHI IS NOT MUCH WORSE # 0 if params is not None else -24
            # pars.add('c', value=params[2] if params is not None else 10, min=0, max=np.inf, vary=True)
            # pars.add('s', value=params[3] if params is not None else 600, min=0, max=np.inf, vary=True)

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
        result = mini.minimize(method='least_squares', xtol=1e-07 if fileType == FileType.SIMULATED else 1e-05)

        if fileType == FileType.SIMULATED:
            dk = result.params.get('b').value
            dk_err = result.params.get('b').stderr
        else:
            dk = result.params.get('dk').value
            dk_err = result.params.get('dk').stderr

        if plot:
            # print(lmfit.fit_report(result))
            # print(np.abs(dk), ",", dk_err, ",", result.redchi)

            plt.title("Norman fit (k=" + str(k[k_index]) + ")")
            plt.plot(w, EDC_slice, label='data')
            plt.plot(w, EDC_func(w, *result.params.values()), label='fit')
            plt.legend()
            plt.show()

        return np.abs(dk), dk_err, result.redchi, list(result.params.values())
