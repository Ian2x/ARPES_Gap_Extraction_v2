import matplotlib.pyplot as plt
import numpy as np

from extraction_functions import symmetrize_EDC
from enum import Enum


class FileType(Enum):
    FAR_OFF_NODE = 0
    NEAR_NODE = 1
    SIMULATED = 2
    ANTI_NODE = 3


class DataReader:
    def __init__(self,
                 fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0333_nL.dat",
                 plot=True, fileType=FileType.FAR_OFF_NODE):

        self.fileType = fileType

        # Handle simulated files
        if fileType == FileType.SIMULATED:
            Simulated_data_file = open(fileName, "r")
            momentumAxis = ""
            while not momentumAxis.startswith("Momentum_Axis_Scaling"):
                momentumAxis = Simulated_data_file.readline()
            k_start, k_step, k_end = momentumAxis[
                                     momentumAxis.index(" = ") + 3:momentumAxis.index(" invAngstrom")].split(":")
            self.full_k = np.arange(float(k_start), float(k_end) + float(k_step) / 2, float(k_step))
            energyAxis = Simulated_data_file.readline()
            w_start, w_step, w_end = energyAxis[energyAxis.index(" = ") + 3:energyAxis.index(" eV")].split(":")
            self.full_w = np.arange(float(w_start), float(w_end) + float(w_step) / 2, float(w_step))
            self.full_w = 1000 * np.flip(self.full_w)
            w_dim = len(self.full_w)
            k_dim = len(self.full_k)
            while Simulated_data_file.readline() != '[Data]\n':
                pass
            Simulated_data_file.readline()
            Simulated_data = np.zeros((w_dim, k_dim))

            for i in range(k_dim):
                temp = Simulated_data_file.readline()
                temp_split = temp.split()
                for j in range(w_dim):
                    Simulated_data[w_dim - j - 1][k_dim - i - 1] = temp_split[j]  # fill in opposite
            self.full_Z = Simulated_data
            self.zoomed_w = None
            self.zoomed_k = None
            self.zoomed_Z = None
            if plot:
                plt.title("Raw Eugen data")
                im = plt.imshow(Simulated_data, cmap=plt.cm.RdBu, aspect='auto',
                                extent=[min(self.full_k), max(self.full_k), min(self.full_w), max(self.full_w)])  # drawing the function
                plt.colorbar(im)
                plt.show()
            Simulated_data_file.close()
            return

        Eugen_data_file = open(fileName, "r")
        Eugen_data_file.readline()  # skip blank starting line
        temp = Eugen_data_file.readline()  # energy?
        temp_split = temp.split()

        if fileType == FileType.FAR_OFF_NODE or fileType == FileType.ANTI_NODE:
            w_dim = 201
            k_dim = 695
        elif fileType == FileType.NEAR_NODE:
            w_dim = 401
            k_dim = 690

        Eugen_data = np.zeros((w_dim, k_dim))
        k = np.zeros(k_dim)
        w = np.zeros(w_dim)
        for i in range(w_dim):
            w[i] = float(temp_split[i]) * 1000
        self.full_w = np.flip(w)

        Eugen_data_file.readline()  # empty 0.0164694505526385 / 0.515261371488587
        if fileType == FileType.FAR_OFF_NODE or fileType == FileType.ANTI_NODE:
            Eugen_data_file.readline()  # unfilled 0.513745070571566 (FOR FAR OFF NODE ONLY)
            Eugen_data_file.readline()  # unfilled 0.512228769654545 (FOR FAR OFF NODE ONLY)

        for i in range(k_dim):
            temp = Eugen_data_file.readline()
            temp_split = temp.split()
            k[i] = float(temp_split[0])  # flip to positive --> removed negative
            for j in range(w_dim):
                Eugen_data[w_dim - j - 1][k_dim - i - 1] = temp_split[j + 1]  # fill in opposite
        self.full_k = np.flip(k)
        self.full_Z = Eugen_data
        self.zoomed_w = None
        self.zoomed_k = None
        self.zoomed_Z = None
        if plot:
            plt.title("Raw Eugen data")
            im = plt.imshow(Eugen_data, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(k), max(k), min(w), max(w)])  # drawing the function
            plt.colorbar(im)
            plt.show()
        Eugen_data_file.close()

    def getZoomedData(self, width=140, height=70, x_center=360, y_center=75, scaleup=17500, plot=True):
        """
        Zoom in onto a part of the spectrum. Sets zoomed_k, zoomed_w, and zoomed_Z
        :param width:
        :param height:
        :param x_center: measured from top left
        :param y_center: measured from top left
        :param scaleup:
        :param plot:
        :return:
        """
        height_offset = round(y_center - 0.5 * height)
        width_offset = round(x_center - 0.5 * width)
        print(height_offset, "+")
        print(width_offset, "-")

        zoomed_k = np.zeros(width)
        zoomed_w = np.zeros(height)

        zoomed_Z = np.zeros((height, width))

        for i in range(height):
            zoomed_w[i] = self.full_w[i + height_offset]
            for j in range(width):
                zoomed_Z[i][j] = self.full_Z[i + height_offset][j + width_offset]
                zoomed_k[j] = self.full_k[j + width_offset]

        zoomed_Z = np.multiply(zoomed_Z, scaleup)
        zoomed_Z = np.around(zoomed_Z)

        self.zoomed_k = zoomed_k
        self.zoomed_w = zoomed_w
        self.zoomed_Z = zoomed_Z

        if plot:
            plt.title("Raw Eugen data (Reduced Window)")
            im = plt.imshow(zoomed_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(zoomed_k), max(zoomed_k), min(zoomed_w), max(zoomed_w)])  # drawing the function
            plt.colorbar(im)
            plt.show()

    def symmetrize_data(self, plot=True):
        new_Z = []
        for ki in range(len(self.zoomed_k)):
            EDC = [self.zoomed_Z[i][ki] for i in range(len(self.zoomed_w))]
            _, EDC = symmetrize_EDC(self.zoomed_w, EDC)
            new_Z.append(EDC)
        new_w, _ = symmetrize_EDC(self.zoomed_w, [self.zoomed_Z[i][0] for i in range(len(self.zoomed_w))])

        self.zoomed_w = new_w
        self.zoomed_Z = np.array(new_Z).T
        if plot:
            plt.title("Symmetrized data")

            im = plt.imshow(self.zoomed_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(self.zoomed_k), max(self.zoomed_k), min(self.zoomed_w), max(self.zoomed_w)])
            plt.colorbar(im)
            plt.show()
