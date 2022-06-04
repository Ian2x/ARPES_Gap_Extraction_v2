import math

from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from extraction_functions import EDC_array_with_SE, Norman_EDC_array
from fitter import Fitter
from general import k_as_index
from simulation import Simulator

import matplotlib.pyplot as plt
import numpy as np


def run():
    # Detector settings
    temp = 79.38
    energy_conv_sigma = 8 / 2.35482004503
    data = DataReader(
        fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0301_nL.dat",
        show_results=True)

    # Get initial estimates
    data.getZoomedData(width=115, height=70, x_center=356, y_center=75)
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, initial_k_error = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        show_results=True)

    # Single EDC fit
    data.getZoomedData(width=140, height=60, x_center=360, y_center=45)
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(initial_kf_estimate, data.zoomed_k), energy_conv_sigma, temp)

    # Zoom in for 2D fit
    data.getZoomedData(width=140, height=78, x_center=356, y_center=73)
    data.zoomed_k -= initial_k_error

    kde = KDependentExtractor(data.zoomed_Z, data.zoomed_w, data.zoomed_k, initial_a_estimate, initial_c_estimate,
                              energy_conv_sigma, temp, show_results=True, show_detailed_results=False)
    kde.get_scale_T_trajectory()

    # Scale:
    scale_values, _ = kde.get_scale_polynomial_fit()
    # scale_values = [3.94563418e+10, 2.50361076e+09, -3.85545197e+08, -1.47453871e+06, 4.03211205e+05, 3.24421285e+05] # initial estimate
    # scale_values = [6.1433e+09, -5.3653e+08, -73147548.7, 4180419.68, -50724.1242, 106819.989] # straight procedure # 0333
    # scale_values = [1.9666e+11, -91119497.0, -1.7062e+09, -2229997.43, 5012187.75] #0289 - round 1
    # scale_values = [1.8276e+11, -3.4123e+08, -1.5272e+09, -889944.591, 4436754.12] #0289 - round 1, fixed k error


    # T:
    T_values, _ = kde.get_T_polynomial_fit()
    # T_values = [2.07152382e+08, -6.26312933e+07, 1.33239638e+06, 1.55071332e+06, -1.44518086e+04, -4.28136695e+03, 5.22045960e+01, 2.09819901e+01] # initial estimate
    # T_values = [2.5048e+09, -88100967.7, -46658808.8, 1916500.55, 262460.826, -13661.7164, -835.216945, 33.1876436, 8.53026365]
    # T_values = [-3.5014e+09, 1.4812e+09, 1.1205e+08, -7027173.07, -862622.765, -21794.7264, 1966.45122, 217.650737]
    # T_values = [-3.9234e+09, 1.4551e+09, 1.1732e+08, -7843161.49, -897239.858, -8308.99786, 2053.21277, 171.485934]


    # Secondary electron:
    kde.get_secondary_electron_scale_trajectory(70)
    secondary_electron_scale_values, _ = kde.get_secondary_electron_scale_polynomial_fit()

    # secondary_electron_scale_values = [-5.84437456e+09, -7.89026155e+08, 1.09275104e+08, 1.85089153e+07, -5.92610005e+05, -1.93976511e+05, -1.51030295e+03, 2.42464089e+03] # initial estimate
    # secondary_electron_scale_values = [3.1719e+08, -60354835.4, 703864.584, 661442.780, -71886.6620, -4003.87839, 1846.36596]
    # secondary_electron_scale_values = [-8.5896e+09, -2.1848e+08, 2.6164e+08, 3060999.47, -2433494.19, -5559.29695, 5018.14244]
    # secondary_electron_scale_values = [-8.2045e+09, -2.4621e+08, 2.5319e+08, 3561700.54, -2373498.19, -7839.37567, 4884.47606]


    fitter = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                    initial_dk_estimate,
                    initial_kf_estimate, temp, energy_conv_sigma, 1)
    fitter.fit(scale_values, T_values, secondary_electron_scale_values, hasty_fit=False)

    # fitted_map = fitter.get_fitted_map(scale_values, T_values, secondary_electron_scale_values)
    # fitter.relative_error_map(fitted_map)


if __name__ == '__main__':
    run()
