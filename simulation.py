import math
import numpy as np

from general import H_BAR, ELECTRON_MASS, LIGHT_SPEED, add_noise
from spectral_functions import I


class Simulator:

    def __init__(self, dk=15, energy_convolution_sigma=(8 / 35482004503), T=5, raw_scaleup_factor=2000, kf=0.4, c=-1000,
                 w=np.arange(-125, 50, 1), d_theta=(0.045 * math.pi / 180)):
        self.dk = dk
        self.energy_convolution_sigma = energy_convolution_sigma
        self.T = T
        self.scaleup_factor = raw_scaleup_factor * (energy_convolution_sigma + T)
        self.kf = kf
        self.c = c
        self.a = -c / (kf ** 2)
        self.w = w
        k_step = (1 / H_BAR) * math.sqrt(2 * ELECTRON_MASS / LIGHT_SPEED / LIGHT_SPEED * 6176.5840329647) * d_theta / (
                10 ** 10)
        self.k = np.arange(kf - 0.04 * kf, kf + 0.04 * kf, k_step)

    def generate_spectra(self):
        w = np.flip(self.w)
        X, Y = np.meshgrid(self.k, w)

        # The Spectrum
        Z = I(X, Y, self.a, self.c, self.dk, self.T, self.scaleup_factor)
        add_noise(Z)

        return Z
