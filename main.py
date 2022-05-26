from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from fitter import Fitter
from general import k_as_index


def run():
    temp = 20.44
    energy_conv_sigma = 8 / 2.35482004503
    data = DataReader()
    data.getZoomedData(width=130)
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, _ = extract_ac(data.zoomed_Z,
                                                                                                     data.zoomed_k,
                                                                                                     data.zoomed_w,
                                                                                                     show_results=True)

    # data.getZoomedData(height=25, y_center=60)
    # Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate, k_as_index(initial_kf_estimate, data.zoomed_k), energy_conv_sigma, temp, symmetrize=True)
    data.getZoomedData(width=140, x_center=360)
    _, _, _, _, new_k = extract_ac(data.zoomed_Z, data.zoomed_k, data.zoomed_w)
    data.zoomed_k = new_k

    kde = KDependentExtractor(data.zoomed_Z, data.zoomed_w, data.zoomed_k, initial_a_estimate, initial_c_estimate,
                              energy_conv_sigma, temp, show_detailed_results=False)
    kde.get_scale_T_trajectory()

    # Scale:
    scale_values, _ = kde.get_scale_polynomial_fit()
    # scale_values = [3.94563418e+10, 2.50361076e+09, -3.85545197e+08, -1.47453871e+06, 4.03211205e+05, 3.24421285e+05] # initial estimate
    # scale_values = [-4.4329e+09, -3.1625e+08, 36661174.2, 2212823.76, -288337.786, 106836.717] # normal procedured

    # T:
    T_values, _ = kde.get_T_polynomial_fit()
    # T_values = [2.07152382e+08, -6.26312933e+07, 1.33239638e+06, 1.55071332e+06, -1.44518086e+04, -4.28136695e+03, 5.22045960e+01, 2.09819901e+01] # initial estimate
    # T_values = [-1.0822e+08, -5479703.07, 1674032.66, 66153.0222, -8893.91960, -533.918608, 20.1874522, 8.29715681]

    # Secondary electron:
    kde.get_secondary_electron_scale_trajectory(53)
    secondary_electron_scale_values, _ = kde.get_secondary_electron_scale_polynomial_fit()

    # secondary_electron_scale_values = [-5.84437456e+09, -7.89026155e+08, 1.09275104e+08, 1.85089153e+07, -5.92610005e+05, -1.93976511e+05, -1.51030295e+03, 2.42464089e+03] # initial estimate
    # secondary_electron_scale_values = [4.1450e+08, 2.3176e+08, 19130024.8, -143901.830, -258697.465, -59909.4128, -1656.56969, 1719.85017]

    fitter = Fitter(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                    initial_dk_estimate,
                    initial_kf_estimate, temp, energy_conv_sigma, 1
                    # , override_index_to_fit=range(0, len(data.zoomed_k), 1)
                    )
    fitter.fit(scale_values, T_values, secondary_electron_scale_values, hasty_fit=False)

    # fitted_map = fitter.get_fitted_map(scale_values, T_values, secondary_electron_scale_values)
    # fitter.relative_error_map(fitted_map)


if __name__ == '__main__':
    run()
