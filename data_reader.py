import matplotlib.pyplot as plt
import numpy as np


class DataReader:
    def __init__(self, fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0333_nL.dat", w_dim=201, k_dim=695, show_results=False):
        Eugen_data_file = open(fileName, "r")
        Eugen_data_file.readline()  # skip blank starting line
        temp = Eugen_data_file.readline()  # energy?
        temp_split = temp.split()

        w_dim = w_dim  # 401 for near node # 201 for far off node
        k_dim = k_dim  # 690 for near node # 695 for far off node

        Eugen_data = np.zeros((w_dim, k_dim))
        k = np.zeros(k_dim)
        w = np.zeros(w_dim)
        for i in range(w_dim):
            w[i] = float(temp_split[i]) * 1000
        self.full_w = np.flip(w)

        Eugen_data_file.readline()  # empty 0.0164694505526385 / 0.515261371488587
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
        self.show_results = show_results
        if show_results:
            plt.title("Raw Eugen data")
            im = plt.imshow(Eugen_data, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(k), max(k), min(w), max(w)])  # drawing the function
            plt.colorbar(im)
            plt.show()
        Eugen_data_file.close()

    def getZoomedData(self, width=140, height=70, x_center=360, y_center=75, scaleup=17500):
        """
        Zoom in onto a part of the spectrum. Sets zoomed_k, zoomed_w, and zoomed_Z
        :param width:
        :param height:
        :param x_center: measured from top left
        :param y_center: measured from top left
        :param scaleup:
        :return:
        """
        height_offset = int(y_center - 0.5 * height)
        width_offset = int(x_center - 0.5 * width)

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

        if self.show_results:
            plt.title("Raw Eugen data (Reduced Window)")
            im = plt.imshow(zoomed_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(zoomed_k), max(zoomed_k), min(zoomed_w), max(zoomed_w)])  # drawing the function
            plt.colorbar(im)
            plt.show()
