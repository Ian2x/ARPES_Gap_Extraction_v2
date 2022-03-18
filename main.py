from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from fitter import Fitter
from general import k_as_index


def run():
    energy_conv_sigma = 8 / 2.35482004503
    temp = 20.44
    data = DataReader()
    data.getZoomedData(width=130)
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, _ = extract_ac(data.zoomed_Z,
                                                                                                     data.zoomed_k,
                                                                                                     data.zoomed_w,
                                                                                                     show_results=True)
    '''
    data.getZoomedData(width=134, x_center=358)
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(initial_kf_estimate, data.zoomed_k), energy_conv_sigma, temp)
    '''
    data.getZoomedData(width=160, x_center=357)
    _, _, _, _, new_k = extract_ac(data.zoomed_Z, data.zoomed_k, data.zoomed_w)
    data.zoomed_k = new_k


    kde = KDependentExtractor(data.zoomed_Z, data.zoomed_w, data.zoomed_k, initial_a_estimate, initial_c_estimate,
                              energy_conv_sigma, temp)
    # kde.get_scale_T_trajectory()

    # Scale:
    # kde.get_scale_polynomial_fit()
    # Degree 7: [3.72745445e+10, -8.29803952e-02, -2.75205864e+08, -1.11602589e+07, -1.47606681e+06, 8.58483769e+04, 6.26965847e+04]
    # scale_values = [0.02996548e+09, -0.08989385e+07, -0.24741341e+07, -0.37697062e+05, 0.36064692e+05]
    scale_values = [2996548.00, -90000.1927, -7797152.50, -144384.938, 124542.461]
    # T:
    # kde.get_T_polynomial_fit()
    # Degree 8: [1.63855814e+09, -1.87097421e+08, -2.60864563e+07, 3.18392532e+06, 1.51744248e+05, -2.02966857e+04, -8.70145464e+02, 5.36146433e+01, 1.06426711e+01]
    # T_values = [0.19667778e+10, -1.40999484e+08, -0.57394822e+08, 0.22270583e+07, 0.46428775e+06, -1.27396243e+04, -0.16121280e+04,  0.28327974e+02, 0.09017932e+02]
    T_values = [2.5189e+09, -87234713.2, -61012381.6, 1639118.19, 452398.580, -10963.7699, -1525.70489,  26.8537528, 8.92120918]

    # Secondary electron:
    # kde.get_secondary_electron_scale_trajectory(53)
    # kde.get_secondary_electron_scale_polynomial_fit()
    # quit()
    # Degree 8: [-1.25680438e+09, -1.12568875e+08, 3.68760369e+07, 3.32541296e+06, -3.54704464e+05, -4.59811129e+04, 8.08980450e+02, 6.90401262e+02]
    # secondary_electron_scale_values = [-1.56786707e+09, 1.79439201e+07, 1.67003361e+07, -2.39011268e+05, -0.05309353e+05, -0.60779948e+03, 0.34521585e+03]
    secondary_electron_scale_values = [-2.2941e+09, 41339057.9, 25280285.4, -536736.973, -530.935300, -940.101870, 661.322323]

    fitter = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                    initial_dk_estimate,
                    initial_kf_estimate, temp, energy_conv_sigma, 15, override_index_to_fit=range(0, len(data.zoomed_k), 15))
    fitter.fit(scale_values, T_values, secondary_electron_scale_values)
    # fitted_map = fitter.get_fitted_map(scale_values, T_values, secondary_electron_scale_values)
    # fitter.relative_error_map(fitted_map)


if __name__ == '__main__':
    run()
