import matplotlib.pyplot as plt
import numpy as np
import csv

from data_reader import DataReader, FileType
from extract_ac import extract_ac, FittingOrder
from extraction_functions import symmetrize_Z
from fitter import Fitter
from general import k_as_index
from figures import figure2_a, figure1_ab, figure1_cd, figure2_b, substitute_zeroes, figure2_c, figure3_a, figure3_b, \
    figure4_a, figure4_b


def run(runData=None):
    # Detector settings
    energy_conv_sigma = 8 / 2.35482004503

    """
    NEAR NODE: from OD50_0189 to OD50_0233
    """
    # fileName = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_near_node/OD50_0199_nL.dat"
    # x_center = (1 - (int(fileName[72:76]) - 189) / (233 - 189)) ** 2 * (360 - 347) + 347  # from 360 to 347
    # data = DataReader(fileName=fileName, plot=True, fileType=FileType.NEAR_NODE)
    # data.getZoomedData(width=50, height=300, x_center=x_center, y_center=150, plot=True)

    """
        FAR OFF NODE: from OD50_0289 to OD50_0333
    """
    # fileName = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0333_nL.dat"
    # data = DataReader(fileName=fileName, plot=True, fileType=FileType.FAR_OFF_NODE)
    # data.getZoomedData(width=150, height=140, x_center=k_as_index(0, data.full_k), y_center=70, plot=True)

    """
        ANTI NODE: from OD50_0238 to OD50_0286
    """
    # fileName = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_antinode/OD50_0271_nL.dat"
    # data = DataReader(fileName=fileName, plot=True, fileType=FileType.ANTI_NODE)
    # data.getZoomedData(width=118, height=140, x_center=k_as_index(0, data.full_k)+7, y_center=70, plot=True)

    """
        SIMULATED: from dEdep_0000 to dEdep_0019, from SNRdep_0000 to SNRdep_0024, from Tdep_0000 to Tdep_0039
    """
    simulated_file = r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/Akw_Simulation_20221222_Fitting/Akw_Tdep_0025.dat"
    if runData is not None:
        simulated_file = runData
    data = DataReader(fileName=simulated_file, plot=False, fileType=FileType.SIMULATED)
    data.getZoomedData(width=220, height=400, x_center=k_as_index(0, data.full_k), y_center=240, plot=False, scaleup=50)

    temp_simulated_data_file = open(simulated_file, "r")
    snr_reading = ""
    while not snr_reading.startswith("SNR = "):
        snr_reading = temp_simulated_data_file.readline()
    snr = float(snr_reading[snr_reading.index(" = ") + 3:])
    res_reading = ""
    while not res_reading.startswith("Energy_Resolution = "):
        res_reading = temp_simulated_data_file.readline()
    energy_conv_sigma = 1000 * float(res_reading[res_reading.index(" = ") + 3:res_reading.index(" eV")]) / 2.35482004503
    gap_reading = ""
    while not gap_reading.startswith("Gap_Size = "):
        gap_reading = temp_simulated_data_file.readline()
    true_gap = 1000 * float(gap_reading[gap_reading.index(" = ") + 3:gap_reading.index(" eV")])
    # print("TRUE GAP: ", true_gap)

    """
        DATA FITTING
    """

    a, c, kf, k_error, my_dk, my_dk_err, my_redchi = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        energy_conv_sigma,
        data.fileType,
        plot_EDC_fits=False,
        fittingOrder=FittingOrder.right_to_left if data.fileType == FileType.NEAR_NODE else FittingOrder.center_out,
    )
    # print("\na/c/kf,k_error")
    # print(a, ",", c, ",", kf, ",", k_error, "\n")
    # a, c, kf, k_error = 2312.390942163413 , -5.486135939256181 , 0.04870826414559312 , 0.006540910672073074
    # kf = 0.07257183696
    # k_error = 0

    simulated = data.fileType == FileType.SIMULATED
    if kf is not None and k_error is not None:
        data.getZoomedData(width=len(data.full_k) - 2, height=len(data.full_w) - 2,
                           x_center=round(len(data.full_k) / 2), y_center=round(len(data.full_w) / 2), plot=False,
                           scaleup=50 if simulated else 17500)
        if simulated:
            params = [1000, -13, 9, 600]
        elif data.fileType == FileType.ANTI_NODE:
            params = [97000, 0, 0.1, 17, 0, 600]
        else:
            params = [35000, 0, 0.1, 15, 0, 1000]
        N_dk, N_dk_err, N_redchi, N_params = Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, k_as_index(
            (-kf if data.fileType == FileType.NEAR_NODE else kf) + k_error, data.full_k), energy_conv_sigma,
                                                              data.fileType, print_results=False, plot_results=False,
                                                              params=params)
        # print("NORMAN STUFF:")
        # print(N_dk, ",", N_dk_err, ",", N_redchi)

        return [true_gap, energy_conv_sigma, snr, my_dk, my_dk_err, my_redchi, a, c, kf, N_dk, N_dk_err, N_redchi]


if __name__ == '__main__':
    # figure4_a()
    figure4_b()

    # with open('/Users/ianhu/Documents/ARPES/T sim.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     # writer.writerow(["File", "True Gap", "Resolution", "SNR", "1.5D fit gap", "error", "redchi", "estimated_a", "estimated_c", "estimated_kf", "Norman gap", "Norman error", "Norman redchi"])
    #
    #     for i in range(2, 3):
    #         try:
    #             fileNum = '{:0>4}'.format(str(i))
    #             result = run(runData="/Users/ianhu/Documents/ARPES/Akw_Simulation_20230205_Fitting/Akw_Tdep_" + fileNum + ".dat")
    #             # writer.writerow([fileNum] + result)
    #             # print([fileNum] + result)
    #         except Exception as inst:
    #             print(inst)
    #             # writer.writerow([])

    # run()
