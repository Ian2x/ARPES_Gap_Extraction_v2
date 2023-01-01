import numpy as np

from data_reader import DataReader, FileType
from extract_ac import extract_ac, FittingOrder
from fitter import Fitter
from general import k_as_index


def run():
    # Detector settings
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
    data = DataReader(
        fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0289_nL.dat",
        plot=True, fileType=FileType.FAR_OFF_NODE)
    maxWidth = 125
    minWidth = 85
    centering = 0  # Start at 0 and go to 10 as necessary
    data.getZoomedData(width=maxWidth, height=140, x_center=k_as_index(0, data.full_k) + centering, y_center=70, plot=True)
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
    # simulated_file = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/Akw_Simulation_20221222_Fitting/Akw_Tdep_0026.dat"
    # data = DataReader(
    #     fileName=simulated_file, plot=False, fileType=FileType.SIMULATED)
    # maxWidth = 195
    # minWidth = 155
    # data.getZoomedData(width=maxWidth, height=400, x_center=k_as_index(0, data.full_k), y_center=240, plot=True)
    #
    # temp_simulated_data_file = open(simulated_file, "r")
    # res_reading = ""
    # while not res_reading.startswith("Energy_Resolution = "):
    #     res_reading = temp_simulated_data_file.readline()
    # energy_conv_sigma = 1000 * float(res_reading[res_reading.index(" = ") + 3:res_reading.index(" eV")]) / 2.35482004503
    # gap_reading = ""
    # while not gap_reading.startswith("Gap_Size = "):
    #     gap_reading = temp_simulated_data_file.readline()
    # true_gap = 1000 * float(gap_reading[gap_reading.index(" = ") + 3:gap_reading.index(" eV")])
    # print("TRUE GAP: ", true_gap)

    simulated = data.fileType == FileType.SIMULATED
    a, c, kf, k_error = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        minWidth,
        maxWidth,
        energy_conv_sigma,
        plot_trajectory_fits=True,
        plot_EDC_fits=True,
        fittingOrder=FittingOrder.left_to_right if data.fileType == FileType.NEAR_NODE else FittingOrder.center_out,
        simulated=simulated
    )
    print("\na/c/kf,k_error")
    print(a, ",", c, ",", kf, ",", k_error, "\n")
    # a, c, kf, k_error = 2705.3688606280425 , -24.773435648254214 , 0.09569293532357644 , 0.006450556118251764

    if kf is not None and k_error is not None:
        data.getZoomedData(width=150, height=140, x_center=k_as_index(0, data.full_k) + centering, y_center=70,
                           plot=True)
        Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, k_as_index(kf + k_error, data.zoomed_k), energy_conv_sigma, simulated=simulated, print_results=True, plot_results=True, params=None)


if __name__ == '__main__':
    run()
