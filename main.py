from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from extraction_functions import spectrum_slice_array_SEC, extend_array
from fitter import Fitter
from general import k_as_index, d6_polynomial, d5_polynomial
from simulation import Simulator
import matplotlib.pyplot as plt

import numpy as np


def run():
    # tempz = Simulator().generate_spectra()
    # im = plt.imshow(tempz, cmap=plt.cm.RdBu, aspect='auto')  # drawing the function
    # plt.colorbar(im)
    # plt.show()
    # data = DataReader()
    # data.getZoomedData(width=134)
    # test = spectrum_slice_array_SEC(data.zoomed_w, 5000, 8, 15, 0.7556327, -7.27236, 0.4604658, 267.0537, 2200, -21.6158, 0.05, 8, 20)
    # plt.plot(test)
    # plt.show()
    # quit()

    energy_conv_sigma = 8 / 2.35482004503
    temp = 20.44
    data = DataReader()
    data.getZoomedData(width=130)
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, _ = extract_ac(data.zoomed_Z,
                                                                                                     data.zoomed_k,
                                                                                                     data.zoomed_w,
                                                                                                     show_results=True)
    # ax^2 + c

    '''
    data.getZoomedData(width=134, x_center=358)
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(initial_kf_estimate, data.zoomed_k), energy_conv_sigma, temp)
    '''
    # data.getZoomedData(width=140, height=54, x_center=360, y_center=67) # width=140, height=70, x_center=360, y_center=75, scaleup=17500
    data.getZoomedData(width=140, x_center=360)
    _, _, _, _, new_k = extract_ac(data.zoomed_Z, data.zoomed_k, data.zoomed_w)
    data.zoomed_k = new_k

    kde = KDependentExtractor(data.zoomed_Z, data.zoomed_w, data.zoomed_k, initial_a_estimate, initial_c_estimate,
                              energy_conv_sigma, temp, show_detailed_results=False)
    # kde.get_scale_T_trajectory()

    # Scale:
    # kde.get_scale_polynomial_fit()
    # scale_values = [3.94563418e+10, 2.50361076e+09, -3.85545197e+08, -1.47453871e+06, 4.03211205e+05, 3.24421285e+05]
    scale_values = [-1.4350e+10, -5.1734e+08, 1.3418e+08, -3813503.97, -382866.298, 99117.5984]

    # T:
    # kde.get_T_polynomial_fit()
    # T_values = [2.07152382e+08, -6.26312933e+07, 1.33239638e+06, 1.55071332e+06, -1.44518086e+04, -4.28136695e+03, 5.22045960e+01, 2.09819901e+01]
    T_values = [-2.4541e+08, -21862317.2, 2829319.35, 205927.854, -11008.5502, -825.737140, 19.7705303, 7.36352999]

    # Secondary electron:
    # kde.get_secondary_electron_scale_trajectory(53)
    # kde.get_secondary_electron_scale_polynomial_fit()

    # secondary_electron_scale_values = [-5.84437456e+09, -7.89026155e+08, 1.09275104e+08, 1.85089153e+07, -5.92610005e+05, -1.93976511e+05, -1.51030295e+03, 2.42464089e+03]
    secondary_electron_scale_values = [-6.0475e+10, -3.4179e+09, 1.2917e+09, 36265780.3, -7688688.72, 68628.6809, 6174.97196, 1854.56933]

    fitter = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                    initial_dk_estimate,
                    initial_kf_estimate, temp, energy_conv_sigma, 1,
                    override_index_to_fit=range(0, len(data.zoomed_k), 1))
    fitter.fit(scale_values, T_values, secondary_electron_scale_values, hasty_fit=False)

    # fitted_map = fitter.get_fitted_map(scale_values, T_values, secondary_electron_scale_values)
    # fitter.relative_error_map(fitted_map)

    # plt.plot(kde.scale_trajectory)
    # vectorized = np.vectorize(d6_polynomial)
    # plt.plot(vectorized(data.zoomed_k, *scale_values))
    # plt.show()


if __name__ == '__main__':
    run()
