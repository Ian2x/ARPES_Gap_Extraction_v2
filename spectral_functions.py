import math
from functools import partial

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
        def f(k2):
            if a * k2 ** 2 + c > 0:
                return 1
            elif a * k2 ** 2 + c < 0:
                return 0
            else:
                return 0.5
        try:
            return np.array([f(k3) for k3 in k])
        except TypeError:
            return f(k)
    return 0.5 * (1 + e(k, a, c) / E(k, a, c, dk))


def final_u(loc, dk):
    if dk == 0:
        if loc > 0:
            return 1
        elif loc < 0:
            return 0
        else:
            return 0.5
    return 0.5 * (1 + loc / (loc ** 2 + dk ** 2) ** 0.5)


def v(k, a, c, dk):
    """
    Coherence Factors (relative intensity of BQP bands below EF)
    """
    return 1 - u(k, a, c, dk)


def final_v(loc, dk):
    return 1 - final_u(loc, dk)


def A_BCS(k, w, a, c, dk, T):
    """
    BCS Spectral Function (https://arxiv.org/pdf/cond-mat/0304505.pdf) (non-constant gap)
    """
    local_T = max(T, 0)
    return (1 / math.pi) * (
            u(k, a, c, dk) * local_T / ((w - E(k, a, c, dk)) ** 2 + local_T ** 2) + v(k, a, c, dk) * local_T / (
            (w + E(k, a, c, dk)) ** 2 + local_T ** 2))


def final_A_BCS(w, loc, dk, T):
    return (1 / math.pi) * (
                final_u(loc, dk) * T / ((w - (loc ** 2 + dk ** 2) ** 0.5) ** 2 + T ** 2) + final_v(loc, dk) * T / (
                    (w + (loc ** 2 + dk ** 2) ** 0.5) ** 2 + T ** 2))


def A_BCS_2(k, w, a, c, dk, T):
    """
    Alternative Spectral Function - broken
    (http://ex7.iphy.ac.cn/downfile/32_PRB_57_R11093.pdf)
    """
    local_T = max(T, 0)
    return local_T / (math.pi * ((w - e(k, a, c) - (dk ** 2) / (w + e(k, a, c))) ** 2 + local_T ** 2))


def final_A_BCS_2(w, loc, dk, T):
    local_T = max(T, 0)
    return local_T / (math.pi * ((w - loc - (dk ** 2) / (w + loc)) ** 2 + local_T ** 2))


def final_A_BCS_3(w, loc, dk, T1, T0):
    local_T1 = max(T1, 0)
    local_T0 = max(T0, 0)
    denom = (w + loc) * (w + loc) + local_T0 * local_T0
    real_pt = (dk * dk * (w + loc)) / denom
    imag_pt = -local_T1 - (dk * dk * local_T0) / denom
    return (1 / np.pi) * (-imag_pt) / ((w - loc - real_pt) * (w - loc - real_pt) + imag_pt * imag_pt)


def sigma_real(k, w, a, c, dk, T0):
    T0 = max(T0, 0)
    return dk * dk * (w + E(k, a, c, dk)) / ((w + E(k, a, c, dk)) * (w + E(k, a, c, dk)) + T0 * T0)


def sigma_imaginary(k, w, a, c, dk, T0, T1):
    T0 = max(T0, 0)
    return -T1 - T0 * dk * dk / ((w + E(k, a, c, dk)) * (w + E(k, a, c, dk)) + T0 * T0)


def A_BCS_3(k, w, a, c, dk, T0, T1):
    """
    Same as A_BCS_2, but complex version
    (http://ex7.iphy.ac.cn/downfile/32_PRB_57_R11093.pdf)
    """
    return 1 / np.pi * sigma_imaginary(k, w, a, c, dk, T0, T1) / (
            (w - E(k, a, c, dk) - sigma_real(k, w, a, c, dk, T0)) * (
            w - E(k, a, c, dk) - sigma_real(k, w, a, c, dk, T0)) + sigma_imaginary(k, w, a, c, dk, T0,
            T1) * sigma_imaginary(k, w, a, c, dk, T0, T1))


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


def I_nofermi(k, w, true_a, true_c, true_dk, true_T, scaleup_factor, energy_conv_sigma):
    convolution_function = partial(A_BCS, a=true_a, c=true_c, dk=true_dk, T=true_T)
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
    convolution_function = partial(norm_state_Io_n_A_BCS, true_a=true_a, true_c=true_c, true_dk=true_dk, true_T=true_T,
                                   temp=temp)
    return energy_conv_map(k, w, convolution_function, energy_conv_sigma, scaleup_factor)
