import csv
import math

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from functools import partial

from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
from data_reader import FileType, DataReader
from extraction_functions import Norman_EDC_array2, Norman_EDC_array, symmetrize_Z
from fitter import Fitter
from general import k_as_index
from simulation import Simulator
from spectral_functions import A_BCS
from statistics import mean, stdev
from math import log

k_str = 'Momentum ($\AA^{-1}$)'
w_str = 'Energy (mev)'
kF = '$k_F$'
eF = '$e_F$'


def substitute_zeroes(y, errors):
    for i in range(len(y)):
        if (errors[i] is not None and errors[i] > y[i]) or (errors[i] is None and y[i] < 0.1):
            y[i] = 0
            errors[i] = 0


def floatify(i):
    try:
        return float(i)
    except ValueError:
        return None


def cap_rss(arr, truth, cap):
    return mean([((x - truth) ** 2 if np.abs(x - truth) < cap else cap ** 2) for x in arr])


def bounded_rss(arr, truth, min=None, max=None):
    ret = mean([((x - truth) ** 2) for x in arr])
    if min is not None and ret < min:
        return min
    elif max is not None and ret > max:
        return max
    else:
        return ret


def figure1ab(gap):
    sim = Simulator(k_step=0.0005, dk=25 if gap else 0, w=np.arange(-80, 80, 1), width=0.03)

    Z = np.sqrt(np.flip(sim.generate_spectra(fermi=False, noise=False), 0))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(sim.k, sim.w)

    ax.plot_wireframe(X, Y, Z, rstride=0, cstride=5)
    ax.set_zlim(0, 35)

    ax.xaxis.set_major_locator(FixedLocator([0.4]))
    ax.xaxis.set_major_formatter(FixedFormatter([kF]))
    ax.set_xlabel(k_str)

    ax.yaxis.set_major_locator(FixedLocator([0]))
    ax.yaxis.set_major_formatter(FixedFormatter([eF]))
    ax.set_ylabel(w_str)

    ax.zaxis.set_major_locator(FixedLocator([]))
    ax.zaxis.set_major_formatter(FixedFormatter([]))
    ax.set_zlabel('Intensity')

    ax.set_title('Superconducting State Dispersion' if gap else 'Normal State Dispersion')

    ax.view_init(elev=40, azim=-40)

    plt.show()


def figure1cd(offset):
    def gapvtemp(t):
        if t < 20:
            return np.sqrt(100 - (t / 2) ** 2)
        else:
            return 0
    w = np.arange(-10, 10, 0.1)
    kf = 0.4
    c = -1000
    dk = 1
    T = 5
    plt.figure(figsize=(6.4, 12.8))
    above_labeled = False
    below_labeled = False
    dot_labeled = False
    for i in np.arange(18, 22, 0.1):
        y = np.array(2 * A_BCS(kf + (0.0008 if offset else 0), w, -c / (kf ** 2), c, gapvtemp(i), T))
        label = None
        if not above_labeled and not np.isclose([gapvtemp(i)], [0]):
            label = "above " + kF
            above_labeled = True
        elif not below_labeled and np.isclose([gapvtemp(i)], [0]):
            label = "below " + kF
            below_labeled = True
        plt.plot(w, y - 0.19 * i,
                 color="green" if np.isclose([gapvtemp(i)], [0]) else ("red" if offset else "blue"), label=label)
        dot_label = None
        if above_labeled and below_labeled and not dot_labeled:
            dot_label = "peak position"
            dot_labeled = True
        plt.plot(w[y.argmax()], y[y.argmax()] - 0.19 * i, 'ko', label=dot_label)
        if not offset:
            plt.plot(-w[y.argmax()], y[y.argmax()] - 0.19 * i, 'ko')
    plt.title(("Offset " if offset else "") + kF + " EDCs at Various Temperature")
    plt.ylabel("Temperature")
    plt.xlabel("Energy")
    plt.tick_params(labelleft=False, labelbottom=False)
    plt.legend()
    plt.show()


def figure2a():
    # Prep data
    simulated_file = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/Akw_Simulation_20221222_Fitting/Akw_dEdep_0017.dat"
    data = DataReader(fileName=simulated_file, plot=False, fileType=FileType.SIMULATED)
    data.getZoomedData(width=160, height=400, x_center=k_as_index(0, data.full_k), y_center=240, plot=False, scaleup=50)
    temp_w, temp_Z = symmetrize_Z(data.zoomed_w, data.zoomed_Z)
    Z = temp_Z
    k = data.zoomed_k
    w = temp_w
    energy_conv_sigma = 8 / 2.35482004503
    fileType = FileType.SIMULATED
    critical_is = range(5, 160, 10)
    kf_i = 145
    kf_neighbor_is = [125, 135, 155]

    # Prep fitting range
    z_width = Z[0].size
    super_state_trajectory = np.zeros(z_width)
    super_state_trajectory_errors = np.zeros(z_width)
    fitting_range = list(range(int(z_width / 2), -1, -1)) + list(range(int(z_width / 2) + 1, z_width, 1))

    params = None
    critical_params = []
    critical_locs = []
    ordered_critical_is = []

    # Fit EDCs
    for i in fitting_range:
        if i == int(z_width / 2) + 1:
            params = None

        loc, loc_std, _, params = Fitter.NormanFit(Z, k, w, i, energy_conv_sigma, fileType, params=params,
                                                   print_results=False, plot_results=False)
        super_state_trajectory[i] = -loc
        super_state_trajectory_errors[i] = loc_std

        if i in critical_is:
            critical_params.append(params)
            critical_locs.append(loc)
            ordered_critical_is.append(i)

    # Get trajectory fit
    def trajectory_form(x, a, c, dk, k_error):
        return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5

    simulated = fileType == FileType.SIMULATED
    pars = lmfit.Parameters()
    pars.add('a', value=1901.549259 if simulated else 2800, min=0)
    pars.add('c', value=-20.6267059166932 if simulated else -27, max=0)
    pars.add('dk', value=1)
    pars.add('k_error', value=0, min=-0.03, max=0.03, vary=not (simulated or fileType == FileType.NEAR_NODE))

    def calculate_residual(p, k, sst, sst_error):
        residual = (trajectory_form(k, p['a'], p['c'], p['dk'], p['k_error']) - sst) / sst_error
        return residual

    fit_function = partial(calculate_residual, k=k, sst=super_state_trajectory, sst_error=super_state_trajectory_errors)
    mini = lmfit.Minimizer(fit_function, pars, nan_policy='omit', calc_covar=True)
    result = mini.minimize(method='least_squares')

    # Plot EDC fits and locs
    if fileType == FileType.SIMULATED:
        EDC_func = partial(Norman_EDC_array2, energy_conv_sigma=energy_conv_sigma, noConvolute=True)
    else:
        EDC_func = partial(Norman_EDC_array, energy_conv_sigma=energy_conv_sigma)

    max_peak = -np.inf
    for critical_param in critical_params:
        max_peak = max(max_peak, *EDC_func(w, *critical_param))

    scale_factor = 10 * (k[1] - k[0]) / max_peak

    my_labeled = False
    multi_labeled = False
    norman_labeled = False
    point_labeled = False
    for i, critical_param in reversed(list(enumerate(critical_params))):
        EDC_color = "black" if ordered_critical_is[i] == kf_i else (
            "purple" if ordered_critical_is[i] in kf_neighbor_is else "green")

        label = None
        point_label = None
        if not norman_labeled:
            if EDC_color == "black":
                label = "Norman fit"
                norman_labeled = True
        elif not multi_labeled:
            if EDC_color == "purple":
                label = "Multi-EDC fit"
                multi_labeled = True
        elif not point_labeled:
            point_label = "EDC gap estimate"
            point_labeled = True
        elif not my_labeled:
            if EDC_color == "green":
                label = "1.5D fit EDCs"
                my_labeled = True

        plt.plot(k[ordered_critical_is[i]] + scale_factor * EDC_func(w, *critical_param), w, color=EDC_color,
                 label=label)
        # plt.plot(k[ordered_critical_is[i]] + scale_factor * EDC_func(loc, *critical_param), -critical_locs[i], marker='o', markersize=2, color="orange")
        plt.plot(k[ordered_critical_is[i]], -critical_locs[i], 'oy', markersize=2, label=point_label)

    # Plot trajectory
    plt.plot(k, trajectory_form(k, result.params.get('a').value, result.params.get('c').value,
                                result.params.get('dk').value, result.params.get('k_error').value), color="orange",
             label="1.5D fit trajectory")

    # Plot heat map
    plt.title("Norman method vs. Multi-EDC vs. 1.5D fit")
    im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])
    plt.colorbar(im, label='Intensity')

    # Show plot
    plt.legend()
    plt.xlabel(k_str)
    plt.ylabel(w_str)
    plt.show()


def figure2b(fileType="SNR"):
    file = "/Users/ianhu/Documents/ARPES/_ " + fileType + " 4 - Sheet1.csv"
    with open(file, 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        row = next(reader)
        xs = []
        my_dks = []
        my_errs = []
        N_dks = []
        N_errs = []
        true_gaps = []
        while row and row[0] != "END":
            row = row[:12]
            file, x, true_gap, my_dk, my_err, my_redchi, e_a, e_c, e_kf, N_dk, N_err, N_redchi = [float(i) for i in row]
            xs.append(x)
            true_gaps.append(true_gap)
            my_dks.append(my_dk)
            my_errs.append(my_err)
            N_dks.append(N_dk)
            N_errs.append(N_err)
            row = next(reader)

        substitute_zeroes(my_dks, my_errs)
        substitute_zeroes(N_dks, N_errs)

        plt.figure(figsize=(6.4, 4.8), dpi=144)
        plt.errorbar(xs, my_dks, yerr=my_errs, capsize=2, label="1.5D Fit", fmt='.:')
        plt.errorbar(xs, N_dks, yerr=N_errs, capsize=2, label=kF + " Fit", fmt='.:')
        plt.plot(xs, true_gaps, label="True Gap")

        if fileType == "SNR":
            title = kF + " Fit vs. 1.5D Fit Across Noise"
            xlabel = "Noise to signal factor"
        elif fileType == "dE":
            title = kF + " vs. 1.5D Fit Across Energy Resolution"
            xlabel = "Energy Resolution (mev)"
            plt.ylim(10, 14)
        elif fileType == "T":
            title = kF + " vs. 1.5D Fit Across Temperature"
            xlabel = "Temperature (K)"
            plt.ylim(0, 15)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Estimated Gap Size (mev)")
        plt.legend(loc="upper right")
        plt.show()


def figure2c(step=1):
    sim = Simulator(k_step=0.0005)

    if step == 1:
        Z = sim.generate_spectra(fermi=False, convolute=False, noise=False)
        plt.title("Spectral Function")
    elif step == 2:
        Z = sim.generate_spectra(fermi=True, convolute=False, noise=False)
        plt.title("Add Fermi-Dirac Distribution")
    elif step == 3:
        Z = sim.generate_spectra(fermi=True, convolute=True, noise=False)
        plt.title("Add Energy Convolution")
    elif step == 4:
        Z = sim.generate_spectra(fermi=True, convolute=True, noise=True)
        plt.title("Add Poisson Noise")

    im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto',
                    extent=[min(sim.k), max(sim.k), min(sim.w),
                            max(sim.w)])  # drawing the function
    plt.colorbar(im, label="Intensity")
    plt.xlabel(k_str)
    plt.ylabel(w_str)
    plt.show()


def figure3a(nodeType="NN"):
    if nodeType == "NN":
        fileName = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_near_node/OD50_0233_nL.dat"
        data = DataReader(fileName=fileName, plot=False, fileType=FileType.NEAR_NODE)
        data.getZoomedData(width=50, height=200, x_center=355, y_center=100, plot=False)
        plt.title("Near-Node Spectrum")
    elif nodeType == "FN":
        fileName = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0333_nL.dat"
        data = DataReader(fileName=fileName, plot=False, fileType=FileType.FAR_OFF_NODE)
        data.getZoomedData(width=150, height=140, x_center=k_as_index(0, data.full_k), y_center=70, plot=False)
        plt.title("Far-Node Spectrum")
    elif nodeType == "AN":
        fileName = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_antinode/OD50_0238_nL.dat"
        data = DataReader(fileName=fileName, plot=False, fileType=FileType.ANTI_NODE)
        data.getZoomedData(width=118, height=140, x_center=k_as_index(0, data.full_k) + 7, y_center=70, plot=False)
        plt.title("Anti-Node Spectrum")

    im = plt.imshow(data.zoomed_Z, cmap=plt.cm.RdBu, aspect='auto',
                    extent=[min(data.zoomed_k), max(data.zoomed_k), min(data.zoomed_w),
                            max(data.zoomed_w)])  # drawing the function
    plt.colorbar(im, label="Intensity")
    plt.xlabel(k_str)
    plt.ylabel(w_str)
    plt.show()


def figure3b(nodeType="NN"):
    if nodeType == "NN":
        ft = "Near"
        title = "Near-Node Experimental Data Fit"
    elif nodeType == "FN":
        ft = "Far Off"
        title = "Far-Node Experimental Data Fit"
    elif nodeType == "AN":
        ft = "Anti"
        title = "Anti-Node Experimental Data Fit"
    with open('/Users/ianhu/Documents/ARPES/_ ' + ft + ' Node 4 - Sheet1.csv', 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        for i in range(5):
            next(reader)
        row = next(reader)
        xs = []
        my_dks = []
        my_errs = []
        N_dks = []
        N_errs = []
        while row and row[0] != "END":
            row = row[:12]

            file, x, my_dk, my_err, my_redchi, e_a, e_c, e_kf, e_k_err, N_dk, N_err, N_redchi = [floatify(i) for i in
                                                                                                 row]
            xs.append(x)
            my_dks.append(my_dk)
            my_errs.append(my_err)
            N_dks.append(N_dk)
            N_errs.append(N_err)
            row = next(reader)

        substitute_zeroes(my_dks, my_errs)
        substitute_zeroes(N_dks, N_errs)

        plt.errorbar(xs, my_dks, yerr=my_errs, capsize=2, label="1.5D Fit", fmt='.:')
        plt.errorbar(xs, N_dks, yerr=N_errs, capsize=2, label=kF + " Fit", fmt='.:')

    plt.plot(xs, [8 for _ in xs], color="red", label="Resolution")

    plt.title(title)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Gap Size (mev)")
    plt.xlim(0, 105)
    plt.ylim(0, 15)
    plt.legend(loc="upper right")
    plt.show()


def figure4a():
    with open('/Users/ianhu/Documents/ARPES/big simulation fit 3.csv', 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        next(reader)

        situations = {}

        for i in range(675):
            try:
                file, true_gap, res, snr, my_dk, my_err, my_redchi, e_a, e_c, e_kf, N_dk, N_err, N_redchi = next(
                    reader)[:13]
                key = (float(true_gap), float(res), float(snr))
                if key not in situations:
                    situations[key] = [[], []]
                situations[key][0].append(float(my_dk))
                situations[key][1].append(float(N_dk))
            except ValueError:
                pass

        fig = plt.figure(figsize=(12.8, 9.6), dpi=144)
        ax = fig.add_subplot(projection='3d')
        cdict = {
            'red': (
                (0.0, 0.8, 0.8),
                (0.2, 0.8, 0.8),
                (0.5, 1.0, 1.0),
                (0.8, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            'green': (
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.5, 1.0, 1.0),
                (0.8, 0.6, 0.6),
                (1.0, 0.6, 0.6),
            ),
            'blue': (
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.5, 1.0, 1.0),
                (0.8, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            )
        }
        cmap = LinearSegmentedColormap('mycmap', cdict)
        for key in situations:
            print("=======")
            print("GAP, RES, SNR", key)
            print(mean(situations[key][0]), stdev(situations[key][0]), np.quantile(situations[key][0], [0,0.25,0.5,0.75,1]))
            print(mean(situations[key][1]), stdev(situations[key][1]), np.quantile(situations[key][1], [0,0.25,0.5,0.75,1]))

            # Uncapped rss
            my_rss = bounded_rss(situations[key][0], key[0], min=0.01)
            N_rss = bounded_rss(situations[key][1], key[0], min=0.01)

            size = my_rss ** (2/3)
            # # ratio = (N_rss / my_rss) ** (1 / 3)
            #
            # print(N_rss / my_rss)
            # print("                 " + str(math.log(my_rss)))
            # print("                 " + str(math.log(N_rss)))
            # print(1000 * my_rss ** (2/3), 1000 * N_rss ** (2/3))
            color = math.log(N_rss / my_rss) / 10.780852629365064 / 2 + 0.5
            print(N_rss / my_rss)
            ax.scatter3D(key[0], key[1], key[2], s=[1000 * my_rss ** (2/6)], c=color, cmap=cmap, vmin=0, vmax=1, edgecolors='black')
            ax.scatter3D(key[0], key[1], key[2], s=[1000 * N_rss ** (2/6)], color=(0, 0, 0, 0), edgecolors="black",
                         linestyle=':', linewidth=2)
            # if color < 0.49:
            #     ax.scatter3D(key[0], key[1], key[2], s=[2000 * size], color=(0, 0, 0, 0), edgecolors="orange",
            #                  linestyle=':', linewidth=2)
            # ax.scatter3D(key[0], key[1], key[2], s=[2000 * size], c=color, cmap=cmap, vmin=0, vmax=1, edgecolors='black')

        ax.set_xlabel("Gap Size")
        ax.set_ylabel("Resolution")
        ax.set_zlabel("Noise to Signal Factor", labelpad=15)
        ax.set_xticks([0, 6, 12])
        ax.set_yticks([2.5, 4.7, 6.8])
        ax.set_zticks([0.001, 0.011, 0.021])
        plt.title("RSS Comparison over Parameter Space")
        ax.view_init(elev=25, azim=-75)

        sm = plt.cm.ScalarMappable(cmap=cmap)
        cbar = plt.colorbar(sm, ticks=[0, 0.5, 1])

        cbar.ax.tick_params(size=0)
        cbar.ax.set_title("$\dfrac{RSS\:k_F Fit}{RSS\:1.5D fit}$", fontsize=12, pad=15)
        cbar.ax.set_yticklabels(['$\dfrac{1}{48000}$', '1', '48000'])

        plt.show()


def figure4b(bigGap=False):
    with open('/Users/ianhu/Documents/ARPES/big simulation fit 3.csv', 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        next(reader)

        # if set == 1:
        #     tkey = (0, 4.671269902, 0.011)
        # elif set == 2:
        #     tkey = (5.993, 4.671269902, 0.011)
        if bigGap:
            key1 = (12, 2.548, 0.001)
            key2 = (12, 4.671, 0.011)
            key3 = (12, 6.795, 0.021)
        else:
            key1 = (5.993, 2.548, 0.001)
            key2 = (5.993, 4.671, 0.011)
            key3 = (5.993, 6.795, 0.021)
        situations = {key1: [[], []], key2: [[], []], key3: [[], []]}

        for i in range(675):
            try:
                file, true_gap, res, snr, my_dk, my_err, my_redchi, e_a, e_c, e_kf, N_dk, N_err, N_redchi = next(
                    reader)[:13]
                key = (round(float(true_gap), 3), round(float(res), 3), round(float(snr), 3))
                if key in situations:
                    situations[key][0].append(float(my_dk))
                    situations[key][1].append(float(N_dk))
            except ValueError:
                pass

    my_data1 = situations[key1][0]
    my_data2 = situations[key2][0]
    my_data3 = situations[key3][0]

    N_data1 = situations[key1][1]
    N_data2 = situations[key2][1]
    N_data3 = situations[key3][1]

    data = [
        my_data1, my_data2, my_data3,
        N_data1, N_data2, N_data3
    ]

    fig, ax = plt.subplots()

    bp1 = ax.boxplot(data[:3], positions=np.array(range(3)) * 2.0 - 0.4, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot(data[3:], positions=np.array(range(3)) * 2.0 + 0.4, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="C2"))

    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(['Small Res & Noise', 'Medium Res & Noise', 'Big Res & Noise'])
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], Line2D([0], [0], color='red')], ['1.5D Fit', kF + ' Fit', 'True Gap'])
    plt.ylabel("Gap Size (mev)")
    plt.title("Various Fits for Gap Size " + str(key1[0]) + " mev")

    if bigGap:
        plt.ylim(11.5, 12.5)
        plt.plot([-1, 4.8], [12, 12], color="red")
    else:
        plt.ylim(-0.5, 8)
        plt.plot([-1, 4.8], [5.993, 5.993], color="red")

    plt.show()


def figure_1():
    figure1ab(False)
    figure1ab(True)
    figure1cd(False)
    figure1cd(True)


def figure_2():
    figure2a()
    figure2b("SNR")
    figure2b("dE")
    figure2b("T")
    figure2c(1)
    figure2c(2)
    figure2c(3)
    figure2c(4)


def figure_3():
    figure3a("NN")
    figure3a("FN")
    figure3a("AN")
    figure3b("NN")
    figure3b("FN")
    figure3b("AN")


def figure_4():
    figure4a()
    figure4b(False)
    figure4b(True)


def all_figures():
    figure_1()
    figure_2()
    figure_3()
    figure_4()
