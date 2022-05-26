from functools import partial
import math
import numpy as np

from general import n_vectorized, energy_conv_map


def e(k, a, c):
    """
    Normal Dispersion
    """
    return a * k ** 2 + c


def E(k, a, c, dk):
    """
    Superconducting Dispersion
    """
    return (e(k, a, c) ** 2 + dk ** 2) ** 0.5


def u(k, a, c, dk):
    """
    Coherence Factor (relative intensity of BQP bands above EF)
    """
    if dk == 0:
        if a * k ** 2 + c > 0:
            return 1
        elif a * k ** 2 + c < 0:
            return 0
        else:
            return 0.5
    return 0.5 * (1 + e(k, a, c) / E(k, a, c, dk))


def v(k, a, c, dk):
    """
    Coherence Factors (relative intensity of BQP bands below EF)
    """
    return 1 - u(k, a, c, dk)


def A_BCS(k, w, a, c, dk, T):
    """
    BCS Spectral Function (https://arxiv.org/pdf/cond-mat/0304505.pdf) (non-constant gap)
    """
    local_T = max(T, 0)
    # print(k)
    # return 1 / ((w + E(k, a, c, dk)) ** 2 + local_T ** 2)

    return (1 / math.pi) * (u(k, a, c, dk) * local_T / ((w - E(k, a, c, dk)) ** 2 + local_T ** 2) + v(k, a, c, dk) * local_T / (
                (w + E(k, a, c, dk)) ** 2 + local_T ** 2))


def A_BCS_2(k, w, a, c, dk, T):
    """
    Alternative Spectral Function - broken
    (http://ex7.iphy.ac.cn/downfile/32_PRB_57_R11093.pdf)
    """
    local_T = max(3, T)
    return local_T / (math.pi * ((w - e(k, a, c) - (dk ** 2) / (w + e(k, a, c))) ** 2 + local_T ** 2))


def Io(k):
    """
    Intensity Pre-factor. Typically a function of k but approximate as 1
    """
    return 1


def Io_n_A_BCS(k, w, true_a, true_c, true_dk, true_T, temp):
    """
    Full Composition Function. Knows true a, c, dk, and T (ONLY meant to be used with simulated data)
    """
    return Io(k) * n_vectorized(w, temp) * A_BCS(k, w, true_a, true_c, true_dk, true_T)


def I(k, w, true_a, true_c, true_dk, true_T, scaleup_factor, energy_conv_sigma, temp):
    """
    Final Intensity (ONLY meant to be used with simulated data)
    """
    convolution_function = partial(Io_n_A_BCS, true_a=true_a, true_c=true_c, true_dk=true_dk, true_T=true_T, temp=temp)
    return energy_conv_map(k, w, convolution_function, energy_conv_sigma, scaleup_factor)


def norm_state_Io_n_A_BCS(k, w, true_a, true_c, true_T, temp):
    """
    Normal-state Composition Function (dk=0, knows a, c, and T) (ONLY meant to be used with simulated data)
    """
    return Io(k) * n_vectorized(w, temp) * A_BCS(k, w, true_a, true_c, 0, true_T)


def norm_state_I(k, w, true_a, true_c, true_dk, true_T, scaleup_factor, energy_conv_sigma, temp):
    """
    Final Normal-state Intensity (dk=0, knows a, c, and T) (ONLY meant to be used with simulated data)
    """
    convolution_function = partial(norm_state_Io_n_A_BCS, true_a=true_a, true_c=true_c, true_dk=true_dk, true_T=true_T, temp=temp)
    return energy_conv_map(k, w, convolution_function, energy_conv_sigma, scaleup_factor)
