import os

import lmfit
import numpy as np

from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from fitter import Fitter
from general import reject_outliers

'''
New idea:
Combine lorentz fit with SEC over multiple files

'''

# 2.58672980e+03 -2.03344141e+01
#

def run():
    # Detector settings
    temperature = 103.63
    energy_conv_sigma = 8 / 2.35482004503
    data = DataReader(
        fileName=r"X20141210_far_off_node/OD50_0289_nL.dat")
    # data.getZoomedData(width=115, height=140, x_center=358, y_center=70)
    data.getZoomedData(width=110, height=140, x_center=360, y_center=70)

    initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, initial_k_error = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        temperature,
        show_results=True,
        plot_fits=False
    )

    # Plot previously fit from file (fitted map, error map, and reduced-chi)
    # data.getZoomedData(width=115, height=140, x_center=358, y_center=70)
    # fitted_map = Fitter.get_fitted_map(r"/Users/ianhu/Documents/ARPES/test.txt",
    #                                    data.zoomed_k, data.zoomed_w, energy_conv_sigma, temperature, second_fit=False,
    #                                    symmetrized=False)
    # Fitter.relative_error_map(data.zoomed_Z, fitted_map, data.zoomed_k, data.zoomed_w,
    #                           16100 - 21)  # data points - variable

    # Single EDC fit
    # data.getZoomedData(width=140, height=36, x_center=360, y_center=45)
    # data.zoomed_k -= initial_k_error
    # initial_kf_estimate = 0.090
    # print(initial_kf_estimate)
    # Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
    #                  k_as_index(initial_kf_estimate, data.zoomed_k), energy_conv_sigma)
    # quit()

    # Zoom in for kde
    data.getZoomedData(width=115, height=140, x_center=358, y_center=70)

    kde = KDependentExtractor(data.zoomed_Z, data.zoomed_w, data.zoomed_k, initial_a_estimate, initial_c_estimate,
                              energy_conv_sigma, temperature)
    kde.get_kdependent_trajectory(print_results=True, plot_fits=True)

    test = np.abs(kde.dk_trajectory)
    test = reject_outliers(test)
    print(f"=====\nEDC average dk and std: {np.average(test)}, {np.std(test)}\n=====")

    # Scale:
    scale_values, _ = kde.get_scale_polynomial_fit()

    # T:
    T0_values, _ = kde.get_T_polynomial_fit()

    # Secondary electron scale:
    secondary_electron_scale_values, _ = kde.get_secondary_electron_scale_polynomial_fit()

    # Zoom in for 2D fit
    data.getZoomedData(width=115, height=140, x_center=358, y_center=70)

    # Perform initial a,c-fixed 2D fit
    fitter1 = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     initial_dk_estimate,
                     initial_kf_estimate, temperature, energy_conv_sigma, 1,
                     override_index_to_fit=range(0, len(data.zoomed_k))
                     )
    lmfit_scale_params, lmfit_T0_params, lmfit_secondary_electron_scale_params, lmfit_dk, lmfit_q, lmfit_r, lmfit_s, _, _, lmfit_k_error, result \
        = fitter1.fit(scale_values, T0_values, secondary_electron_scale_values, scale_fixed=False, T_fixed=False,
                      SEC_fixed=False, ac_fixed=False,
                      plot_results=False, dk_0_fixed=False)

    ##################################################
    # SAVE TO FILES
    ##################################################

    script_dir = os.path.dirname(__file__)
    rel_path = f"Full-{temperature}K.txt"
    abs_file_path = os.path.join(script_dir, rel_path)

    file = open(abs_file_path, "w+")

    file.write(lmfit.fit_report(result))

    file.close()


if __name__ == '__main__':
    run()
