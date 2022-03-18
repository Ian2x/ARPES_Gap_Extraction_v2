from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

from extraction_functions import EDC_prep, spectrum_slice_array
from general import ONE_BILLION, d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial, \
    d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial, d10_polynomial, F_test


class KDependentExtractor:
    scale_trajectory = [
        86932.44135459, 84126.35940443, 77756.0578149,  73781.71609035,
        71960.41347968, 70396.54615677, 68924.04231627, 67206.19525277,
        65035.39860281, 64322.97445043, 63172.20755897, 62584.0565027,
        62051.6518976, 61214.00138863, 60421.5747216, 60300.22206521,
        59639.0427305, 58964.55162982, 57837.09672192, 57490.41892056,
        57307.6495832, 58836.48368233, 58797.29018218, 60143.55709575,
        60277.18769234, 59927.38624119, 59394.91465632, 58812.05229056,
        60042.6041593, 59962.07366629, 60175.64994826, 60834.92445235,
        60095.15626247, 59821.8407323, 60970.34102306, 60773.76864606,
        60670.25928008, 61587.74444921, 61166.21387541, 60860.8480431,
        60717.60821061, 61522.49404782, 61515.39105384, 61497.33678657,
        61250.51835746, 62086.93800287, 61919.2936947, 61717.81388878,
        61410.51313046, 61357.88231773, 60629.60547207, 61733.66428896,
        61500.54911906, 61194.87859996, 60783.11745674, 60814.08134285,
        61500.9634444, 61369.06139821, 61530.46815013, 62834.8084487,
        63200.5877807, 63317.97558076, 63367.71955595, 62906.131867,
        62852.03344388, 63709.03837937, 64435.10140252, 64508.45786434,
        64353.37266737, 64144.41048325, 64595.26423178, 64495.93950764,
        64152.88825006, 63764.95435521, 63504.74892683, 63187.29834727,
        63247.78685169, 63485.90275363, 63017.41189174, 61866.4063832,
        61882.96653416, 61698.14967169, 61498.84321245, 61143.17195301,
        61496.57566118, 61905.61257177, 62275.67389026, 61684.97298451,
        61102.20598305, 60543.89206635, 60573.14701209, 60500.10896242,
        60068.94203214, 59008.86145413, 59025.23808731, 58188.29622883,
        57889.28354083, 57557.62857844, 57217.78045924, 56940.7907421,
        56702.81507142, 56190.60985507, 55791.99652542, 55120.89382071,
        53543.48791814, 53314.46355244, 53708.16696607, 53714.83063212,
        52346.87461053, 51830.46975964, 51519.68773962, 51649.81300705,
        50793.91509544, 50657.61276788, 50230.05374714, 48669.0553728,
        48350.09296038, 47577.91537157, 47396.10869791, 47202.3918825,
        46334.31465631, 46319.40073842, 44830.86541935, 44367.80342754,
        42889.50635401, 42882.2734729, 43107.10127703, 42096.55956036,
        41786.03227416, 41517.98310979, 41819.68099481, 41565.68144793,
        41323.45501566, 41445.20844457, 41643.8703114, 41656.34549767,
        41945.52871067, 42507.52808938]

    T_trajectory = [
        7.30646979, 7.3851975, 6.07581627, 5.75451786, 5.85215645, 5.92840298,
        5.98950637, 5.96571795, 5.83903404, 5.98539357, 6.01826809, 6.14228169,
        6.25929465, 6.29444454, 6.2841076, 6.3674222, 6.38561713, 6.39267309,
        6.35984849, 6.45318053, 6.50297785, 6.85498225, 6.93578387, 7.24765438,
        7.34104576, 7.36547562, 7.3561925, 7.32534539, 7.59680173, 7.63260333,
        7.68958719, 7.84324074, 7.75985608, 7.79918837, 8.05763308, 8.08600666,
        8.12469461, 8.32607191, 8.29478317, 8.27424738, 8.28993332, 8.4423906,
        8.54200627, 8.63480125, 8.68549315, 8.84566974, 8.87186128, 8.88277422,
        8.91821527, 8.97879264, 8.89011139, 9.13135198, 9.1826873, 9.21173108,
        9.2442351, 9.30850021, 9.49202544, 9.53162633, 9.64734681, 9.98140694,
        10.08456681, 10.20744305, 10.31051563, 10.32645176, 10.41176188, 10.63562138,
        10.84343893, 10.94052801, 11.02540573, 11.09459767, 11.2690124, 11.29360616,
        11.30067553, 11.2897942, 11.27957614, 11.25201113, 11.32583807, 11.3760698,
        11.27846355, 11.05960242, 11.11671773, 11.14235398, 11.16476089, 11.15485365,
        11.23341466, 11.28815787, 11.3344849, 11.2032604, 11.0689252, 10.93744387,
        10.97913878, 10.9936374, 10.97531835, 10.83425015, 10.79482485, 10.49292286,
        10.37417081, 10.24504856, 10.17832042, 10.10482155, 10.02995278, 9.89687232,
        9.84120234, 9.78658555, 9.53633272, 9.45062399, 9.48710804, 9.44629342,
        9.16893564, 9.04620105, 8.96339304, 8.9653276, 8.76363155, 8.80482451,
        8.78869171, 8.51344015, 8.44149007, 8.26204754, 8.17458602, 8.08287296,
        7.92699977, 7.89488199, 7.57477808, 7.46739657, 7.14168773, 7.10520031,
        7.1272525, 6.88476056, 6.8081696, 6.73544943, 6.76666716, 6.62779984,
        6.4965835, 6.51268189, 6.53308579, 6.49903094, 6.35983045, 6.39863249]

    secondary_electron_scale_trajectory = None

    def __init__(self, Z, w, k, initial_a_estimate, initial_c_estimate, energy_conv_sigma, temp, show_results=True,
                 show_detailed_results=False, min_fit_count=1):
        self.scale_polynomial_degree = None
        self.T_polynomial_degree = None
        self.secondary_electron_scale_degree = None
        self.scale_polynomial_fit = None
        self.T_polynomial_fit = None
        self.secondary_electron_scale_fit = None
        self.Z = Z
        self.w = w
        self.k = k
        self.initial_a_estimate = initial_a_estimate
        self.initial_c_estimate = initial_c_estimate
        self.energy_conv_sigma = energy_conv_sigma
        self.temp = temp
        self.show_results = show_results
        self.show_detailed_results = show_detailed_results
        self.min_fit_count = min_fit_count

    def get_scale_T_trajectory(self):
        z_width = self.Z[0].size
        scale_trajectory = np.zeros(z_width)
        T_trajectory = np.zeros(z_width)
        scale_0, T_0, dk_0 = 1, 1, 1
        for i in range(z_width):
            low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index = \
                EDC_prep(i, self.Z, self.w, self.min_fit_count)

            params, pcov = scipy.optimize.curve_fit(
                partial(spectrum_slice_array, a=self.initial_a_estimate, c=self.initial_c_estimate, fixed_k=self.k[i],
                        energy_conv_sigma=self.energy_conv_sigma, temp=self.temp), low_noise_w, low_noise_slice,
                bounds=(
                    [0, 0, 0],
                    [ONE_BILLION, 75, 75]),
                p0=[scale_0, T_0, dk_0],
                sigma=fitting_sigma)
            scale_trajectory[i] = params[0]
            T_trajectory[i] = params[1]
            scale_0, T_0, dk_0 = params
            if self.show_detailed_results:
                plt.plot(low_noise_w, low_noise_slice)
                plt.plot(low_noise_w,
                         spectrum_slice_array(
                             low_noise_w, *params, self.initial_a_estimate, self.initial_c_estimate, self.k[i],
                             self.energy_conv_sigma, self.temp))
                plt.show()
            print(i)
        KDependentExtractor.scale_trajectory = scale_trajectory
        KDependentExtractor.T_trajectory = T_trajectory
        if self.show_results:
            print("Scale trajectory: ")
            print(", ".join(scale_trajectory))
            print("T trajectory: ")
            print(", ".join(T_trajectory))

    def get_secondary_electron_scale_trajectory(self, y_pos):
        if not (0 <= y_pos < self.w.size):
            raise ValueError("y_pos out of range of w")
        KDependentExtractor.secondary_electron_scale_trajectory = self.Z[y_pos]
        if self.show_results:
            plt.title("Secondary electron scale extraction line")
            im = plt.imshow(self.Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(self.k), max(self.k), min(self.w), max(self.w)])
            plt.hlines(self.w[y_pos], min(self.k), max(self.k))
            plt.colorbar(im)
            plt.show()
            plt.title("Secondary electron scale trajectory")
            plt.plot(self.k, KDependentExtractor.secondary_electron_scale_trajectory)
            plt.show()

    def get_scale_polynomial_fit(self):
        if KDependentExtractor.scale_trajectory is None:
            raise AttributeError("Uninitialized scale_trajectory.")

        polynomial_functions = [d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial,
                                d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial,
                                d10_polynomial]

        for i in range(9):
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k, self.scale_trajectory)
            outer_params, outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 1], self.k,
                                                                self.scale_trajectory)
            inner_fit = polynomial_functions[i](self.k, *inner_params)
            outer_fit = polynomial_functions[i + 1](self.k, *outer_params)
            if self.show_results:
                plt.plot(self.k, self.scale_trajectory)
                plt.plot(self.k, inner_fit)
                plt.show()
            F_statistic = F_test(self.scale_trajectory, inner_fit, i + 1, outer_fit, i + 2,
                                 np.ones(len(self.scale_trajectory)),
                                 len(self.scale_trajectory))
            critical_value = scipy.stats.f.ppf(q=1 - 0.5, dfn=1, dfd=len(self.scale_trajectory) - (i + 2))
            if F_statistic < critical_value:
                print("Optimal scale polynomial is of degree: " + str(i + 1))
                print("Fitted scale parameters are: " + str(inner_params))
                self.scale_polynomial_degree = i + 1
                self.scale_polynomial_fit = inner_fit
                return inner_params, polynomial_functions[i]
        raise RuntimeError("Unable to find optimal scale polynomial fit")

    def get_T_polynomial_fit(self):
        if KDependentExtractor.T_trajectory is None:
            raise AttributeError("Uninitialized T_trajectory.")
        polynomial_functions = [d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial,
                                d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial,
                                d10_polynomial]
        for i in range(9):
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k, self.T_trajectory)
            outer_params, outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 1], self.k,
                                                                self.T_trajectory)
            inner_fit = polynomial_functions[i](self.k, *inner_params)
            outer_fit = polynomial_functions[i + 1](self.k, *outer_params)
            if self.show_results:
                plt.plot(self.k, self.T_trajectory)
                plt.plot(self.k, inner_fit)
                plt.show()
            F_statistic = F_test(self.T_trajectory, inner_fit, i + 1, outer_fit, i + 2,
                                 np.ones(len(self.T_trajectory)),
                                 len(self.T_trajectory))
            critical_value = scipy.stats.f.ppf(q=1 - 0.5, dfn=1, dfd=len(self.T_trajectory) - (i + 2))
            if F_statistic < critical_value:
                print("Optimal T polynomial is of degree: " + str(i + 1))
                print("Fitted T parameters are: " + str(inner_params))
                self.T_polynomial_degree = i + 1
                self.T_polynomial_fit = inner_fit
                return inner_params, polynomial_functions[i]
        raise RuntimeError("Unable to find optimal T polynomial fit")

    def get_secondary_electron_scale_polynomial_fit(self):
        if KDependentExtractor.secondary_electron_scale_trajectory is None:
            raise AttributeError("Uninitialized secondary_electron_scale_trajectory.")
        polynomial_functions = [d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial,
                                d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial,
                                d10_polynomial]
        for i in range(9):
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k, self.secondary_electron_scale_trajectory)
            outer_params, outer_pcov = scipy.optimize.curve_fit(polynomial_functions[i + 1], self.k,
                                                                self.secondary_electron_scale_trajectory)
            inner_fit = polynomial_functions[i](self.k, *inner_params)
            outer_fit = polynomial_functions[i + 1](self.k, *outer_params)
            if self.show_results:
                plt.plot(self.k, self.secondary_electron_scale_trajectory)
                plt.plot(self.k, inner_fit)
                plt.show()
            F_statistic = F_test(self.secondary_electron_scale_trajectory, inner_fit, i + 1, outer_fit, i + 2,
                                 np.ones(len(self.secondary_electron_scale_trajectory)),
                                 len(self.secondary_electron_scale_trajectory))
            critical_value = scipy.stats.f.ppf(q=1 - 0.5, dfn=1, dfd=len(self.secondary_electron_scale_trajectory) - (i + 2))
            if F_statistic < critical_value:
                print("Optimal secondary electron scale polynomial is of degree: " + str(i + 1))
                print("Fitted secondary electron scale parameters are: " + str(inner_params))
                self.secondary_electron_scale_degree = i + 1
                self.secondary_electron_scale_fit = inner_fit
                return inner_params, polynomial_functions[i]
        raise RuntimeError("Unable to find optimal secondary electron scale polynomial fit")

    def plot(self):
        if self.scale_polynomial_fit is None or self.T_polynomial_fit is None:
            raise NotImplementedError("Must get scale and/or T polynomial fits before plotting")
        plt.title("Scale")
        plt.plot(self.k, KDependentExtractor.scale_trajectory)
        plt.plot(self.k, self.scale_polynomial_fit)
        plt.show()
        plt.title("T")
        plt.plot(self.k, KDependentExtractor.T_trajectory)
        plt.plot(self.k, self.T_polynomial_fit)
        plt.show()
