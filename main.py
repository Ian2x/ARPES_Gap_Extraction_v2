from data_reader import DataReader, FileType
from extract_ac import extract_ac, FittingOrder
from general import k_as_index

import numpy as np

def run():
    # Detector settings
    temperature = 12
    energy_conv_sigma = 8 / 2.35482004503

    # data = DataReader(
    #     fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_near_node/OD50_0233_nL.dat",
    #     plot=True, fileType=FileType.NEAR_NODE)
    # maxWidth = 45
    # minWidth = 20
    # data.getZoomedData(width=maxWidth, height=250, x_center=365, y_center=150, plot=True)

    simulated_file = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/Superconductivity - A(k,w)/Akw_Tc067K_0166.dat"
    data = DataReader(
        fileName=simulated_file, plot=False, fileType=FileType.SIMULATED)
    maxWidth = 190
    minWidth = 150  # 120
    data.getZoomedData(width=maxWidth, height=200, x_center=k_as_index(0, data.full_k), y_center=240, plot=True)

    temp_simulated_data_file = open(simulated_file, "r")
    temp_reading = ""
    while not temp_reading.startswith("Temperature = "):
        temp_reading = temp_simulated_data_file.readline()
    temperature = float(temp_reading[temp_reading.index(" = ") + 3:temp_reading.index(" K")])

    # data = DataReader(
    #     fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0333_nL.dat",
    #     plot=True, fileType=FileType.FAR_OFF_NODE)
    # maxWidth = 140
    # minWidth = 90
    # data.getZoomedData(width=maxWidth, height=140, x_center=k_as_index(0, data.full_k), y_center=70, plot=True)

    extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        temperature,
        minWidth,
        maxWidth,
        energy_conv_sigma,
        fullFunc=True,  # energy_conv_sigma,
        plot_trajectory_fits=True,
        plot_EDC_fits=True,
        fittingOrder=FittingOrder.center_out,
        simulated=True
    )

    # Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, 1800, -24, k_as_index(0.12, data.zoomed_k), energy_conv_sigma)


    # print(k_as_index(0.09, data.zoomed_k))
    # print(k_as_index(-0.09, data.zoomed_k))
    # print(k_as_index(0, data.zoomed_k))

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
