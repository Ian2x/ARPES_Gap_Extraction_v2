import numpy as np

from data_reader import DataReader
from extract_ac import extract_ac
from extract_k_dependent import KDependentExtractor
from fitter import Fitter
from general import reject_outliers

def run():
    # Detector settings
    temperature = 67.94
    energy_conv_sigma = 8 / 2.35482004503
    data = DataReader(
        fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0306_nL.dat", plot=False)
    maxWidth = 140
    minWidth = 90
    data.getZoomedData(width=maxWidth, height=140, x_center=359, y_center=70, plot=True)

    extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        temperature,
        minWidth,
        maxWidth,
        plot_trajectory_fits=False,
        plot_EDC_fits=True
    )

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
