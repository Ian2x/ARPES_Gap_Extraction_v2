import numpy as np

from data_reader import DataReader, FileType
from extract_ac import extract_ac, FittingOrder
from extract_k_dependent import KDependentExtractor
from fitter import Fitter
from general import reject_outliers, k_as_index


def run():
    # Detector settings
    temperature = 100
    energy_conv_sigma = 8 / 2.35482004503

    data = DataReader(
        fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_near_node/OD50_0189_nL.dat",
        plot=True, fileType=FileType.NEAR_NODE)
    maxWidth = 20
    minWidth = 20
    data.getZoomedData(width=maxWidth, height=200, x_center=378, y_center=150, plot=True)

    # data = DataReader(
    #     fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/Superconductivity - A(k,w)/Akw_Tc067K_0771.dat", plot=True, fileType=FileType.SIMULATED)
    # maxWidth = 170
    # minWidth = 120
    # data.getZoomedData(width=maxWidth, height=100, x_center=401, y_center=260, plot=True)

    # data = DataReader(
    #     fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0289_nL.dat",
    #     plot=True, fileType=FileType.FAR_OFF_NODE)
    # maxWidth = 140
    # minWidth = 90
    # data.getZoomedData(width=maxWidth, height=140, x_center=359, y_center=70, plot=True)

    extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        temperature,
        minWidth,
        maxWidth,
        fullFunc=False,  # energy_conv_sigma,
        hasBackground=True,
        plot_trajectory_fits=True,
        plot_EDC_fits=True,
        fittingOrder=FittingOrder.right_to_left
    )

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
