import numpy as np

from data_reader import DataReader, FileType
from extract_ac import extract_ac, FittingOrder
from extraction_functions import estimated_peak_movement
from fitter import Fitter
from general import k_as_index, error_weighted_mean


def run():
    # Detector settings
    temperature = 76.94
    energy_conv_sigma = 8 / 2.35482004503

    """
    NEAR NODE: from OD50_0189 to OD50_0233
    """
    # fileName = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_near_node/OD50_0212_nL.dat"
    # x_center = (1 - (int(fileName[72:76]) - 189) / (233 - 189)) ** 2 * (375 - 362) + 362  # from 378 to 358
    # data = DataReader(
    #     fileName=fileName,
    #     plot=True, fileType=FileType.NEAR_NODE)
    # maxWidth = 40
    # minWidth = 20
    # data.getZoomedData(width=maxWidth, height=300, x_center=x_center, y_center=150, plot=True)

    """
        FAR OFF NODE: from OD50_0289 to OD50_0333
    """
    # data = DataReader(
    #     fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0302_nL.dat",
    #     plot=True, fileType=FileType.FAR_OFF_NODE)
    # maxWidth = 130
    # minWidth = 90
    # centering = 5  # Start at 6 and go to 10 as necessary
    # data.getZoomedData(width=maxWidth, height=140, x_center=k_as_index(0, data.full_k) + centering, y_center=70, plot=True)
    """
        ANTI NODE: from OD50_0238 to OD50_0286
    """
    # data = DataReader(
    #     fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_antinode/OD50_0286_nL.dat",
    #     plot=True, fileType=FileType.FAR_OFF_NODE)
    # maxWidth = 130
    # minWidth = 80
    # data.getZoomedData(width=maxWidth, height=140, x_center=k_as_index(0, data.full_k) + 7, y_center=70, plot=True)

    """
        SIMULATED: from dEdep_0000 to dEdep_0019, from SNRdep_0000 to SNRdep_0024, from Tdep_0000 to Tdep_0039
    """
    simulated_file = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/Akw_Simulation_20221222_Fitting/Akw_dEdep_0000.dat"
    data = DataReader(
        fileName=simulated_file, plot=False, fileType=FileType.SIMULATED)
    maxWidth = 160
    minWidth = 120
    data.getZoomedData(width=maxWidth, height=400, x_center=k_as_index(0, data.full_k), y_center=240, plot=True)

    temp_simulated_data_file = open(simulated_file, "r")
    res_reading = ""
    while not res_reading.startswith("Energy_Resolution = "):
        res_reading = temp_simulated_data_file.readline()
    energy_conv_sigma = 1000 * float(res_reading[res_reading.index(" = ") + 3:res_reading.index(" eV")]) / 2.35482004503
    temp_reading = ""
    while not temp_reading.startswith("Temperature = "):
        temp_reading = temp_simulated_data_file.readline()
    temperature = float(temp_reading[temp_reading.index(" = ") + 3:temp_reading.index(" K")])
    gap_reading = ""
    while not gap_reading.startswith("Gap_Size = "):
        gap_reading = temp_simulated_data_file.readline()
    true_gap = 1000 * float(gap_reading[gap_reading.index(" = ") + 3:gap_reading.index(" eV")])
    print("TRUE GAP: ", true_gap)
    print("TEMP: ", temperature)

    a, c, kf, k_error = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        temperature,
        minWidth,
        maxWidth,
        energy_conv_sigma,
        fullFunc=False,  # energy_conv_sigma,
        plot_trajectory_fits=True,
        plot_EDC_fits=True,
        fittingOrder=FittingOrder.center_out,
        noBackground=False,
        simulated=True
    )
    print("\na/c/kf,k_error")
    print(a, ",", c, ",", kf, ",", k_error, "\n")
    # a, c, kf, k_error = 1930.3306293461537 , -20.3878780752655 , 0.10277090073987355 , 0.001485560883442344

    if a is not None:
        Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, a, c,
                         k_as_index(kf + k_error, data.zoomed_k), energy_conv_sigma, simulated=True)

    # if a is not None:
    #     gaps = []
    #     errors = []
    #     i = 0
    #     next_index = k_as_index(kf+k_error, data.zoomed_k)
    #     while next_index >= 0 and next_index < len(data.zoomed_k) and (i < 3 or estimated_peak_movement(data.zoomed_k[next_index], a, c,
    #                                             np.median(gaps)) > 0.23 * np.median(gaps)):
    #         dk, dk_error, redchi = Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, a, c,
    #                                     next_index, energy_conv_sigma, simulated=False)
    #         if redchi > 15 or (i > 2 and np.abs((dk-np.median(gaps))) > 1.5 * np.std(gaps)):
    #             print("c", dk-np.median(gaps), 1.5 * np.std(gaps))
    #             break
    #         gaps.append(dk)
    #         errors.append(dk_error)
    #         i += 1
    #         next_index -= 1
    #     print("GAPS", i, gaps)
    #     if k_as_index(0+k_error, data.zoomed_k) <= next_index:
    #         j = 0
    #         next_index = k_as_index(-kf+k_error, data.zoomed_k)
    #         while next_index >= 0 and next_index < len(data.zoomed_k) and (
    #                 j < 3 or estimated_peak_movement(data.zoomed_k[next_index], a, c,
    #                                                  np.median(gaps)) > 0.23 * np.median(gaps)):
    #             dk, dk_error, redchi = Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, a, c,
    #                                                     next_index, energy_conv_sigma, simulated=False)
    #             if redchi > 15 or (j > 2 and np.abs((dk - np.median(gaps))) > 1.5 * np.std(gaps)):
    #                 print("c", dk - np.median(gaps), 1.5 * np.std(gaps))
    #                 break
    #             gaps.append(dk)
    #             errors.append(dk_error)
    #             j += 1
    #             next_index += 1
    #     print("GAPS2", j, gaps)
    #
    #     print("===FINAL RESULTS===")
    #     print(np.median(gaps), ",", 1.253 * np.mean(errors) / np.sqrt(i+j), ",", i)
    #     print(errors)

    # Plot previously fit from file (fitted map, error map, and reduced-chi)
    # data.getZoomedData(width=115, height=88, x_center=358, y_center=44)
    # data.symmetrize_data()
    # fitted_map = Fitter.get_fitted_map(r"/Users/ianhu/Documents/ARPES/Norman multiplied/Set 1/0300 - 1 Step, Flat SEC.txt",
    #                                    data.zoomed_k, data.zoomed_w, energy_conv_sigma, temperature, second_fit=False,
    #                                    symmetrized=True)
    # Fitter.relative_error_map(data.zoomed_Z, fitted_map, data.zoomed_k, data.zoomed_w,
    #                           10235 - 26)  # data points - variable


if __name__ == '__main__':
    run()
