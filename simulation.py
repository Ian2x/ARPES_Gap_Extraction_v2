import math
import numpy as np

from general import H_BAR, ELECTRON_MASS, LIGHT_SPEED, add_noise
from spectral_functions import I, I_nofermi, Io_n_A_BCS, A_BCS


class Simulator:

    def __init__(self, dk=15, energy_convolution_sigma=(8 / 2.355), T=5, scaleup_factor=10000, kf=0.4, c=-1000,
                 w=np.arange(-50, 50, 1), d_theta=(0.045 * math.pi / 180), temp=20.44, k_step=None, width=0.016):
        self.dk = dk
        self.energy_convolution_sigma = energy_convolution_sigma
        self.T = T
        self.scaleup_factor = scaleup_factor
        self.kf = kf
        self.c = c
        self.a = -c / (kf ** 2)
        self.w = w
        if k_step is None:
            k_step = (1 / H_BAR) * math.sqrt(2 * ELECTRON_MASS / LIGHT_SPEED / LIGHT_SPEED * 6176.5840329647) * d_theta / (
                    10 ** 10)
        else:
            k_step = k_step
        self.k = np.arange(kf - width / 2, kf + width / 2, k_step)
        self.temp = temp

    def generate_spectra(self, fermi=True, convolute=True, noise=True):
        w = np.flip(self.w)
        X, Y = np.meshgrid(self.k, w)

        # The Spectrum

        if convolute:
            if fermi:
                Z = I(X, Y, self.a, self.c, self.dk, self.T, self.scaleup_factor, self.energy_convolution_sigma, self.temp)
            else:
                Z = I_nofermi(X, Y, self.a, self.c, self.dk, self.T, self.scaleup_factor, self.energy_convolution_sigma)
        else:
            if fermi:
                Z = self.scaleup_factor * Io_n_A_BCS(X, Y, self.a, self.c, self.dk, self.T, self.temp)
            else:
                Z = self.scaleup_factor * A_BCS(X, Y, self.a, self.c, self.dk, self.T)

        if noise:
            add_noise(Z)

        return Z
