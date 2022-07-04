import matplotlib.pyplot as plt
import numpy as np

from extraction_functions import symmetrize_EDC
from general import n, n_vectorized
from spectral_functions import E


class DataReader:
    def __init__(self, fileName=r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0333_nL.dat", w_dim=201, k_dim=695, plot=True):
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

        if plot:
            plt.title("Raw Eugen data (Reduced Window)")
            # zoomed_Z = zoomed_Z - 700
            # zoomed_Z = np.clip(zoomed_Z, a_min=0, a_max=None)
            # zoomed_Z /= n_vectorized(zoomed_w, 100.86)[:, None]
            im = plt.imshow(zoomed_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(zoomed_k), max(zoomed_k), min(zoomed_w), max(zoomed_w)])  # drawing the function
            plt.colorbar(im)
            # plt.plot(zoomed_k, -E((zoomed_k - 0.00608974), 3563.07808, -26.1965687, 7.00118984))  # 0290
            # plt.plot(zoomed_k, 3563.07808 * (zoomed_k - 0.00608974) ** 2 - 26.1965687)
            # plt.plot(zoomed_k, -E((zoomed_k - 0.01248904), 2568.30255, -21.0992210, 13.5027728))  # 0333
            # plt.plot(zoomed_k, 2568.30255 * (zoomed_k - 0.01248904) ** 2 - 21.0992210)
            plt.show()

    def symmetrize_data(self, plot=True):
        print(self.zoomed_w)
        new_Z = []
        for ki in range(len(self.zoomed_k)):
            EDC = [self.zoomed_Z[i][ki] for i in range(len(self.zoomed_w))]
            _, EDC = symmetrize_EDC(self.zoomed_w, EDC)
            new_Z.append(EDC)
        new_w, _ = symmetrize_EDC(self.zoomed_w, [self.zoomed_Z[i][0] for i in range(len(self.zoomed_w))])

        self.zoomed_w = new_w
        self.zoomed_Z = np.array(new_Z).T
        if plot:
            print(self.zoomed_w)
            print(self.zoomed_Z)
            plt.title("Raw Eugen data (Reduced Window)")

            im = plt.imshow(self.zoomed_Z, cmap=plt.cm.RdBu, aspect='auto',
                            extent=[min(self.zoomed_k), max(self.zoomed_k), min(self.zoomed_w), max(self.zoomed_w)])
            plt.colorbar(im)
            plt.show()

