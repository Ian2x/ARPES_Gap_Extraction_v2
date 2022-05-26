from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

from extraction_functions import EDC_prep, EDC_array
from general import ONE_BILLION, d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial, \
    d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial, d10_polynomial, F_test


class KDependentExtractor:
    scale_trajectory = [
        474956.1741051247, 478385.8617540984, 484193.830870808, 491723.9139165234, 496478.1731756622, 500537.29263165325, 506648.1986712532, 509100.46341868147, 509824.1404839886, 512984.79876108863, 521144.6392398756, 526722.1639758267, 530290.9610901886, 544460.0857941564, 562505.400702504, 543034.3906688236, 525944.2103650015, 516955.76779038075, 499401.3012236205, 483177.81668883667, 466115.7675851548, 461555.45695089287, 454314.08059726685, 444362.37131899263, 434744.49549222103, 424116.559758521, 412912.6900625337, 409157.1890758903, 403129.5424847557, 396932.24988021184, 388687.0884116585, 380906.0982684169, 377537.66174268286, 370672.4541559977, 363673.6685861773, 362297.26609865547, 360397.8680834847, 358204.25353825, 356794.0797705772, 353100.4159182194, 348978.8300650019, 347369.59384272044, 345263.53303153766, 342808.1589652301, 341845.2776607741, 340741.48875747045, 338922.13080664276, 337614.4954666724, 336342.7294237602, 334896.6205181691, 334760.88788231026, 336360.388709656, 333857.99107026507, 332237.59280132165, 330361.78626750765, 328339.0152954343, 329103.021914399, 330104.82564402086, 329785.66998474125, 329274.6903874082, 326886.3178709185, 328442.30262240046, 329983.1891342405, 330463.5050763076, 330564.260253776, 328383.0174798045, 327605.36148817604, 330934.57413208694, 331110.8469127828, 331200.36268961075, 331399.2357026919, 331455.4038145798, 334038.1317507767, 334015.7398848778, 333443.46404122456, 332766.05411713215, 330464.1466204631, 328046.1157808527, 328278.37553783937, 330207.79991108994, 330002.9164470999, 329701.81689840456, 331337.4411615297, 330042.47181680857, 328601.9402850688, 327496.30158073077, 327898.478210603, 328864.44605205825, 329639.87980666535, 328218.98789208086, 326617.11098613904, 325320.71382158244, 325547.3316246134, 325196.0755609161, 323344.1669468309, 321035.3974416764, 315560.52898157964, 316504.33033322735, 316003.53956149984, 315533.94346782763, 315298.5292431265, 313424.1172556945, 311681.91087078856, 313440.30086500803, 313405.155541168, 312733.71063773806, 312104.919530102, 312486.67260696774, 314430.01837022044, 314587.7293654268, 313833.1314986076, 312641.2368050727, 311038.72289084754, 312410.81621992093, 314244.3658118772, 320127.22736744134, 321194.7519164237, 322955.12498152803, 327319.19514291553, 331846.8487692964, 334218.92269000475, 337066.89853667247, 344337.839972531, 354255.98382660944, 362883.9623999194, 371938.25071609125, 378831.3724063294, 384607.31833308464, 390126.07679259527, 407193.04296576645, 413792.14286727074, 423196.91283569863, 449908.90002698835, 478666.7168539967, 493591.0459321869, 493016.04969793657, 487078.8275362643, 483283.901518988, 483634.1925538333, 482198.7368400825,
    ]

    T_trajectory = [
        30.70537966616467, 32.253571454936164, 34.24317814683006, 35.94632994670707, 37.43540189380593, 38.93862900660417, 40.78339087007686, 41.59787088223215, 42.032372214610774, 43.058693510078314, 44.81797139989372, 46.135300167485674, 46.90051852432884, 49.986117253028546, 54.021921099547285, 50.4408420347617, 47.27671460656413, 45.343401310623356, 42.216753244779305, 39.479294807209484, 36.96582833563833, 36.18532465663794, 34.802764734871865, 33.050494481728265, 31.444230528763107, 29.884850849608053, 27.85332044279245, 27.07537119874575, 26.179264942248523, 25.31434708240766, 24.127695489347637, 23.04358632331181, 22.253963808413644, 21.408397697292717, 20.61309153008037, 20.44997339076736, 20.30086461938168, 20.135925329268392, 19.964658640062684, 19.618973803666012, 19.24465407699088, 19.065939738207973, 18.840226839702968, 18.588440360593346, 18.54378766780851, 18.49094257007762, 18.463006375785127, 18.31023435801468, 18.238446913422035, 18.15686850262301, 18.24519720089105, 18.358804149236178, 18.106574272057532, 18.06011950344624, 18.05908380943394, 18.042111512671045, 18.18755401654767, 18.273273390282366, 18.292533251066523, 18.289870407886227, 18.12348163245017, 18.280742652333824, 18.36944487307087, 18.56240282080403, 18.722654378243497, 18.67928282574699, 18.70046530178021, 18.907687472040156, 18.98168194407405, 19.042980416511224, 19.22009055354109, 19.379472522464663, 19.62036374036793, 19.64813799479936, 19.710572814487968, 19.7587547465159, 19.720522892753216, 19.667305044275572, 19.779165690708066, 19.932319120636716, 19.923098976024065, 19.9035578390191, 20.029621163787496, 20.158413551325406, 20.27637968332415, 20.419021463165564, 20.46976544213792, 20.51252985842062, 20.54647341170396, 20.561217488120956, 20.557728111072166, 20.529385747689744, 20.652292119184086, 20.722166870999, 20.76850193738643, 20.77391371369192, 20.430918094697233, 20.33993213802284, 20.326907808009786, 20.32372192497804, 20.446989821656647, 20.42995554545939, 20.406444822172233, 20.542493173424436, 20.71342236093424, 20.955685430133443, 21.198820439124407, 21.35911015132522, 21.587013171204614, 21.732969911901236, 21.949860416838117, 22.126155081558196, 22.20361585416783, 22.48606186905326, 22.850107568744313, 23.88240561690136, 24.4998681112005, 25.267613594179963, 26.382865269297664, 27.37012947829689, 27.97363826184752, 28.75261232532573, 30.579279516872063, 32.60134546665098, 34.621859874317636, 36.924102089887654, 39.31738660297334, 41.24556182346488, 43.014372168147816, 47.28444071814097, 49.999205731960934, 53.53822424706865, 61.15117943339948, 69.55714891667341, 74.21594524999972, 74.99999999974347, 74.99999999999315, 74.9999999999841, 74.99999999867305, 74.99999999999804,
    ]

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
                EDC_prep(i, self.Z, self.w, self.min_fit_count, exclude_secondary=False)

            params, pcov = scipy.optimize.curve_fit(
                partial(EDC_array, a=self.initial_a_estimate, c=self.initial_c_estimate, fixed_k=self.k[i],
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
                         EDC_array(
                             low_noise_w, *params, self.initial_a_estimate, self.initial_c_estimate, self.k[i],
                             self.energy_conv_sigma, self.temp))
                plt.show()
            print(i)
        KDependentExtractor.scale_trajectory = scale_trajectory
        KDependentExtractor.T_trajectory = T_trajectory
        if self.show_results:
            print("Scale trajectory: ")
            scale_trajectory_string = ""
            for i in range(scale_trajectory.size):
                scale_trajectory_string += str(scale_trajectory[i]) + ", "
            print(scale_trajectory_string)
            print("T trajectory: ")
            T_trajectory_string = ""
            for i in range(T_trajectory.size):
                T_trajectory_string += str(T_trajectory[i]) + ", "
            print(T_trajectory_string)

    def get_secondary_electron_scale_trajectory(self, y_pos):
        """
        :param y_pos: index of MDC to fit secondary electron polynomial to
        :return:
        """
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
            inner_params, inner_pcov = scipy.optimize.curve_fit(polynomial_functions[i], self.k,
                                                                self.secondary_electron_scale_trajectory)
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
            critical_value = scipy.stats.f.ppf(q=1 - 0.5, dfn=1,
                                               dfd=len(self.secondary_electron_scale_trajectory) - (i + 2))
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
