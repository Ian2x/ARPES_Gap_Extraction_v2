import csv

from data_reader import DataReader, FileType
from extract_ac import extract_ac, FittingOrder
from fitter import Fitter
from general import k_as_index
from figures import figure2a, figure1ab, figure1cd, figure2b, substitute_zeroes, figure2c, figure3a, figure3b, \
    figure4a, figure4b, all_figures, figure_1, figure_2, figure_3, figure_4


def run_fit(fileName, fileType):
    if fileType == FileType.NEAR_NODE:
        x_center = (1 - (int(fileName[fileName.find("_nL.dat")-4:fileName.find("_nL.dat")]) - 189) / (233 - 189)) ** 2 * (360 - 347) + 347  # from 360 to 347
        data = DataReader(fileName=fileName, plot=False, fileType=fileType)
        data.getZoomedData(width=50, height=300, x_center=x_center, y_center=150, plot=False)
        energy_conv_sigma = 8 / 2.35482004503
    elif fileType == FileType.FAR_OFF_NODE:
        data = DataReader(fileName=fileName, plot=False, fileType=fileType)
        data.getZoomedData(width=150, height=140, x_center=k_as_index(0, data.full_k), y_center=70, plot=False)
        energy_conv_sigma = 8 / 2.35482004503
    elif fileType == FileType.ANTI_NODE:
        data = DataReader(fileName=fileName, plot=False, fileType=fileType)
        data.getZoomedData(width=118, height=140, x_center=k_as_index(0, data.full_k) + 7, y_center=70, plot=False)
        energy_conv_sigma = 8 / 2.35482004503
    elif fileType == FileType.SIMULATED:
        data = DataReader(fileName=fileName, plot=False, fileType=fileType)
        data.getZoomedData(width=220, height=400, x_center=k_as_index(0, data.full_k), y_center=240, plot=False, scaleup=50)

        temp_simulated_data_file = open(fileName, "r")
        nextLine = ""
        while not nextLine.startswith("SNR = "):
            nextLine = temp_simulated_data_file.readline()
        snr = float(nextLine[nextLine.index(" = ") + 3:])
        while not nextLine.startswith("Energy_Resolution = "):
            nextLine = temp_simulated_data_file.readline()
        energy_conv_sigma = 1000 * float(nextLine[nextLine.index(" = ") + 3:nextLine.index(" eV")]) / 2.35482004503
        while not nextLine.startswith("Gap_Size = "):
            nextLine = temp_simulated_data_file.readline()
        true_gap = 1000 * float(nextLine[nextLine.index(" = ") + 3:nextLine.index(" eV")])

    """
        DATA FITTING
    """

    a, c, kf, k_error, my_dk, my_dk_err, my_redchi = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        energy_conv_sigma,
        data.fileType,
        plot=False,
        fittingOrder=FittingOrder.right_to_left if data.fileType == FileType.NEAR_NODE else FittingOrder.center_out,
    )

    # Do Norman Fit
    data.getZoomedData(width=len(data.full_k) - 2, height=len(data.full_w) - 2,
                       x_center=round(len(data.full_k) / 2), y_center=round(len(data.full_w) / 2), plot=False,
                       scaleup=50 if data.fileType == FileType.SIMULATED else 17500)
    if data.fileType == FileType.SIMULATED:
        params = [1000, -13, 9, 600]
        # return true_gap, energy_conv_sigma, snr, my_dk, my_dk_err, my_redchi, a, c, kf, N_dk, N_dk_err, N_redchi
    elif data.fileType == FileType.ANTI_NODE:
        params = [97000, 0, 0.1, 17, 0, 600]
    else:
        params = [35000, 0, 0.1, 15, 0, 1000]
    N_dk, N_dk_err, N_redchi, N_params = Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, k_as_index(
        (-kf if data.fileType == FileType.NEAR_NODE else kf) + k_error, data.full_k), energy_conv_sigma,
                                                          data.fileType, plot=False,
                                                          params=params)

    if data.fileType == FileType.SIMULATED:
        return true_gap, energy_conv_sigma, snr, my_dk, my_dk_err, my_redchi, a, c, kf, N_dk, N_dk_err, N_redchi
    return my_dk, my_dk_err, my_redchi, a, c, kf, k_error, N_dk, N_dk_err, N_redchi
    # print(my_dk, my_dk_err, my_redchi, a, c, kf, k_error, N_dk, N_dk_err, N_redchi, sep=", ")


if __name__ == '__main__':
    """
    ===== ===== FILE STRUCTURE ===== =====
    NEAR NODE: from OD50_0189 to OD50_0233
    "X20141210_near_node/OD50_0189_nL.dat"
    
    FAR OFF NODE: from OD50_0289 to OD50_0333
    "X20141210_far_off_node/OD50_0289_nL.dat"

    ANTI NODE: from OD50_0238 to OD50_0286
    "X20141210_antinode/OD50_0238_nL.dat"
    
    SIMULATED: from dEdep_0000 to dEdep_0019, from SNRdep_0000 to SNRdep_0024, from Tdep_0000 to Tdep_0039
    "Akw_Simulation_20221222_Fitting/Akw_Tdep_0025.dat"
    OR (from Akw_Tdep_0000 to Akw_Tdep_0674)
    "Akw_Simulation_20230205_Fitting/Akw_Tdep_0000.dat"
    """

    # run_fit("X20141210_near_node/OD50_0204_nL.dat", FileType.NEAR_NODE)
    figure_1()
    quit()
    # all_figures()

    with open('/Users/ianhu/Documents/ARPES/big simulation fit 3.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # FOR REAL
        # writer.writerow(["File", "1.5D fit gap", "error", "redchi", "estimated_a", "estimated_c", "estimated_kf", "estimated_k_error", "Norman gap", "Norman error", "Norman redchi"])

        # FOR SIMULATION
        # writer.writerow(["File", "True Gap", "Resolution", "SNR", "1.5D fit gap", "error", "redchi", "estimated_a", "estimated_c", "estimated_kf", "Norman gap", "Norman error", "Norman redchi"])

        for i in range(20, 675):
            print(i)
            try:
                fileNum = '{:0>4}'.format(str(i))
                result = run_fit(fileName="Akw_Simulation_20230205_Fitting/Akw_Tdep_" + fileNum + ".dat", fileType=FileType.SIMULATED)
                writer.writerow([fileNum] + list(result))
                # print([fileNum] + list(result))
            except Exception as inst:
                # print(inst)
                writer.writerow([])