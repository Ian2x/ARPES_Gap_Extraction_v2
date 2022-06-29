from functools import partial

from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from fitter import Fitter
from general import k_as_index, polynomial_functions
import numpy as np
import matplotlib.pyplot as plt

from spectral_functions import A_BCS_3


def run():
    # k = np.linspace(-0.11, 0.11, 100)
    # w = np.linspace(-30, 30, 100)
    #
    # import cmath
    # def my_func(w, dk, T, T2, scale):
    #     sig_gap_num = complex(dk * dk, 0)
    #     if w > abs(dk) or w < -abs(dk):
    #         gamma0 = abs(T) + abs(T2) * cmath.sqrt(w ** 2 - T2 ** 2) / abs(w)
    #     else:
    #         gamma0 = abs(T)
    #     sig_gap_dom = complex(w, T2)
    #     sig_gap = sig_gap_num / sig_gap_dom
    #     sig = sig_gap - complex(0, 1) * gamma0
    #     Green = scale / (w - sig)
    #     return -Green.imag / np.pi
    # mfv = np.vectorize(my_func)
    # plt.plot(w, mfv(w, 0-2.26256197e-05, 3.13419155e+01, -4.59050228e-02, 2.90723816e+05))  # w, dk, T, T2, scale
    # plt.show()
    # quit()

    # T1 = [-1.1356e+08, -6304376.09, 1855969.52, 148538.699, -13901.3144, -735.819148, 44.1900628, 18.8245300]  # 0294, 0, 92.90
    # T2 = [-1.1023e+08, -2882554.08, 1819664.75, 112334.019, -13919.1620, -983.161657, 44.2946374, 19.1354002]  # 0294
    # T1 = [105943.581, 53473.4688, -5568.02695, -373.304689, 33.7738801, 19.3289678]  # 0289, 0, 103.63
    # T2 = [1305875.93, -121904.164, 25768.6255, -1162.86980, -715.477793, 10.8887960, 20.2237054]  # 0289
    # T1 = [-1.3515e+08, -9341583.92, 1703970.97, 177439.507, -10240.6321, -739.171243, 34.6533086, 17.8582023]  # 0299, 0, 82.73
    # T2 = [1605939.57, -141457.841, 8649.54705, -6195.83834, -383.361191, 41.1428536, 17.5893176]  # 0299

    # ks = np.linspace(-0.11, 0.11, 100)
    # plt.plot(ks, polynomial_functions[len(T1)-1](ks, *T1), label='dk=0')
    # plt.plot(ks, polynomial_functions[len(T2)-1](ks, *T2), label='dk!=0')
    # plt.title("Temperature = 92.9")
    # plt.legend()
    # plt.show()
    # quit()





    # Detector settings
    temperature = 81.71
    energy_conv_sigma = 8 / 2.35482004503
    data = DataReader(
        fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0300_nL.dat")

    # Get initial estimates (a, c, dk, kf, k_error) - Not too wide for lorentz+SEC fits
    data.getZoomedData(width=115, height=200, x_center=358, y_center=100)
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, initial_k_error = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        show_results=True)


    # Plot previously fit from file (fitted map, error map, and reduced-chi)
    # data.getZoomedData(width=145, height=140, x_center=355, y_center=70)
    # fitted_map = Fitter.get_fitted_map(r"/Users/ianhu/Documents/ARPES/final fit/2Step Fit Statistics/experimental2.txt", data.zoomed_k, data.zoomed_w, energy_conv_sigma, temperature, second_fit=False)
    # Fitter.relative_error_map(data.zoomed_Z, fitted_map, data.zoomed_k, data.zoomed_w, 20300 - 25) # data points - variable
    # quit()

    # initial_k_error = 0.0057
    # initial_a_estimate = 3300
    # initial_c_estimate = -23
    # Single EDC fit
    # data.getZoomedData(width=140, height=36, x_center=360, y_center=45)
    # data.zoomed_k -= initial_k_error
    # initial_kf_estimate = 0.090
    # print(initial_kf_estimate)
    # Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
    #                  k_as_index(initial_kf_estimate, data.zoomed_k), energy_conv_sigma)
    # quit()
    # Zoom in for k-dependent scale, T extraction - Regular height but narrower to dispersion, no weird T effects)
    data.getZoomedData(width=88, height=200, x_center=355, y_center=100)

    kde = KDependentExtractor(data.zoomed_Z, data.zoomed_w, data.zoomed_k, initial_a_estimate, initial_c_estimate,
                              energy_conv_sigma, temperature)
    kde.get_scale_T_trajectory()

    # Scale:
    scale_values, _ = kde.get_scale_polynomial_fit()

    # T:
    T_values, _ = kde.get_T_polynomial_fit()
    # T_values = [18]

    # Zoom in for 2D fit (and secondary electron polynomial extraction) - Tall and wide w/out extra bands
    data.getZoomedData(width=145, height=100, x_center=356, y_center=70)

    # Secondary electron:
    kde.Z = data.zoomed_Z
    kde.k = data.zoomed_k
    kde.w = data.zoomed_w
    kde.get_secondary_electron_scale_trajectory(99)
    secondary_electron_scale_values, _ = kde.get_secondary_electron_scale_polynomial_fit()

    # initial_a_estimate = 3500
    # initial_c_estimate = -27
    # Perform initial a,c-fixed 2D fit
    # fitter1 = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
    #                  initial_dk_estimate,
    #                  initial_kf_estimate, temperature, energy_conv_sigma, 1,
    #                  override_index_to_fit=range(0, len(data.zoomed_k))
    #                  )
    # lmfit_scale_params, lmfit_T_params, lmfit_secondary_electron_scale_params, lmfit_dk, lmfit_q, lmfit_r, lmfit_s, _, _, lmfit_k_error \
    #     = fitter1.fit(scale_values, T_values, secondary_electron_scale_values, kdependent_fixed=True, ac_fixed=True,
    #                   plot_results=False, dk_0_fixed=False)
    testFitter = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     initial_dk_estimate,
                     initial_kf_estimate, temperature, energy_conv_sigma, 1,
                     override_index_to_fit=range(0, len(data.zoomed_k))
                     )
    lmfit_scale_params, lmfit_T_params, lmfit_secondary_electron_scale_params, lmfit_dk, lmfit_q, lmfit_r, lmfit_s, _, _, lmfit_k_error \
        = testFitter.fit(scale_values, T_values, secondary_electron_scale_values, kdependent_fixed=True, ac_fixed=True,
                      plot_results=False, dk_0_fixed=False)

    quit()
    # Perform final 2D fit
    print("\nDOING SECOND FIT...\n")
    fitter2 = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     lmfit_dk if type(lmfit_dk) == int else lmfit_dk.item(),
                     initial_kf_estimate, temperature, energy_conv_sigma, 1,
                     override_index_to_fit=range(0, len(data.zoomed_k)))

    fitter2.fit(lmfit_scale_params, lmfit_T_params, lmfit_secondary_electron_scale_params, kdependent_fixed=False,
                ac_fixed=False,
                plot_results=True, q_estimate=lmfit_q.item(), r_estimate=lmfit_r.item(), s_estimate=lmfit_s.item(),
                k_error_estimate=lmfit_k_error.item(), dk_0_fixed=False)


if __name__ == '__main__':
    run()
