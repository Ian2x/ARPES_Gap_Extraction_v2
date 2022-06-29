from functools import partial

from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from fitter import Fitter

def run():
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
    T0_values, _ = kde.get_T0_polynomial_fit()
    T1_values, _ = kde.get_T1_polynomial_fit()

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
    fitter1 = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     initial_dk_estimate,
                     initial_kf_estimate, temperature, energy_conv_sigma, 1,
                     override_index_to_fit=range(0, len(data.zoomed_k))
                     )
    lmfit_scale_params, lmfit_T0_params, lmfit_T1_params, lmfit_secondary_electron_scale_params, lmfit_dk, lmfit_q, lmfit_r, lmfit_s, _, _, lmfit_k_error \
        = fitter1.fit(scale_values, T0_values, T1_values, secondary_electron_scale_values, kdependent_fixed=True, ac_fixed=True,
                      plot_results=False, dk_0_fixed=False)

    quit()
    # Perform final 2D fit
    # print("\nDOING SECOND FIT...\n")
    # fitter2 = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
    #                  lmfit_dk if type(lmfit_dk) == int else lmfit_dk.item(),
    #                  initial_kf_estimate, temperature, energy_conv_sigma, 1,
    #                  override_index_to_fit=range(0, len(data.zoomed_k)))
    #
    # fitter2.fit(lmfit_scale_params, lmfit_T_params, lmfit_secondary_electron_scale_params, kdependent_fixed=False,
    #             ac_fixed=False,
    #             plot_results=True, q_estimate=lmfit_q.item(), r_estimate=lmfit_r.item(), s_estimate=lmfit_s.item(),
    #             k_error_estimate=lmfit_k_error.item(), dk_0_fixed=False)


if __name__ == '__main__':
    run()
