import math
import numpy as np
import scipy.integrate

from general import R, n, secondary_electron_contribution_array
from spectral_functions import A_BCS, A_BCS_2


def energy_conv_integrand(integration_w, fixed_w, T, dk, a, c, fixed_k, energy_conv_sigma, temp):
    """
    Integrand of energy convolution integral
    :param integration_w: w to integrate over
    :param fixed_w: w to calculate convolution about
    :param T:
    :param dk:
    :param a:
    :param c:
    :param fixed_k: momentum of EDC
    :param energy_conv_sigma:
    :param temp:
    :return:
    """
    return A_BCS(fixed_k, integration_w, a, c, dk, T) * R(math.fabs(integration_w - fixed_w), energy_conv_sigma) * n(
        integration_w, temp)


def spectrum_slice_array_SEC(w_array, scale, T, dk, p, q, r, s, a, c, fixed_k, energy_conv_sigma, temp):
    """
    EDC slice function with secondary electron contribution
    :param w_array: energy array
    :param scale: scaling factor
    :param T:
    :param dk:
    :param p: SEC scale
    :param q: SEC horizontal shift
    :param r: SEC steepness
    :param s: SEC vertical shift
    :param a:
    :param c:
    :param fixed_k: momentum of EDC
    :param energy_conv_sigma:
    :param temp:
    :return:
    """
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250, args=(
            w_array[i], T, dk, a, c, fixed_k, energy_conv_sigma, temp))[0]
    # add in secondary electrons
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(w_array.size):
        return_array[i] = return_array[i] + secondary[i]
    return return_array


def spectrum_slice_array(w_array, scale, T, dk, a, c, fixed_k, energy_conv_sigma, temp):
    """
    EDC slice function
    :param w_array: energy array
    :param scale: scaling factor
    :param T:
    :param dk:
    :param a:
    :param c:
    :param fixed_k: momentum of EDC
    :param energy_conv_sigma:
    :param temp:
    :return:
    """
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250, args=(
            w_array[i], T, dk, a, c, fixed_k, energy_conv_sigma, temp))[0]
    return return_array


def energy_conv_integrand_2(integration_w, fixed_w, T, dk, a, c, fixed_k, energy_conv_sigma, temp):
    return A_BCS_2(fixed_k, integration_w, a, c, dk, T) * R(math.fabs(integration_w - fixed_w), energy_conv_sigma) * n(
        integration_w, temp)


def spectrum_slice_array_SEC_2(w_array, scale, T, dk, p, q, r, s, a, c, fixed_k, energy_conv_sigma, temp):
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand_2, w_array[i] - 250, w_array[i] + 250, args=(
            w_array[i], T, dk, a, c, fixed_k, energy_conv_sigma, temp))[0]
    # add in secondary electrons
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(w_array.size):
        return_array[i] = return_array[i] + secondary[i]
    return return_array


def spectrum_slice_array_2(w_array, scale, T, dk, a, c, fixed_k, energy_conv_sigma, temp):
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand_2, w_array[i] - 250, w_array[i] + 250, args=(
            w_array[i], T, dk, a, c, fixed_k, energy_conv_sigma, temp))[0]
    return return_array


def EDC_prep(curr_index, Z, w, min_fit_count, exclude_secondary=True):
    """
    Prepares relevant variables for EDC calculations
    :param curr_index: index of EDC
    :param Z:
    :param w:
    :param min_fit_count: minimum electron count to fit at
    :param exclude_secondary:
    :return: (low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index)
    """
    z_height = len(Z)
    # Energy Distribution Curve (slice data)
    EDC = np.zeros(z_height)

    # Ignore noisy data
    fit_start_index = -1
    fit_end_index = -1

    if exclude_secondary:
        peak = 0
        peak_index = 0
        one_side_min = np.inf

        for i in range(z_height):

            # Build EDC
            EDC[i] = Z[i][curr_index]

            # Start fit at first index greater than min_fit_count
            if fit_start_index == -1:
                if EDC[i] >= min_fit_count:
                    fit_start_index = i
            # End fit at at last index less than min_fit_count
            if EDC[i] >= min_fit_count:
                fit_end_index = i

            if EDC[i] > peak:
                peak = EDC[i]
                peak_index = i

        for i in range(peak_index, z_height):
            if EDC[i] < one_side_min:
                one_side_min = EDC[i]

        for i in range(peak_index, z_height):
            if EDC[i] > (peak + one_side_min) / 2:
                peak_index += 1
        fit_end_index = min(peak_index, fit_end_index)
    else:
        for i in range(z_height):
            # Build EDC
            EDC[i] = Z[i][curr_index]
        fit_start_index = 0
        fit_end_index = z_height - 1
    points_in_fit = fit_end_index - fit_start_index + 1  # include end point
    if points_in_fit < 5:
        print("Accepted points: ", points_in_fit)
        print("fit_start_index: ", fit_start_index)
        print("fit_end_index: ", fit_end_index)
        raise RuntimeError(
            "ERROR: Not enough points to do proper EDC fit. Suggestions: expand upper/lower energy bounds or increase gap size")

    # Create slice w/ low noise points
    low_noise_slice = np.zeros(points_in_fit)
    low_noise_w = np.zeros(points_in_fit)
    for i in range(points_in_fit):
        low_noise_slice[i] = EDC[i + fit_start_index]
        low_noise_w[i] = w[i + fit_start_index]
    # Remove 0s from fitting sigma
    fitting_sigma = np.sqrt(low_noise_slice)
    for i in range(len(fitting_sigma)):
        if fitting_sigma[i] <= 0:
            fitting_sigma[i] = 1
    return low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index
