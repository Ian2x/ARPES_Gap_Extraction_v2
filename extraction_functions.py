import numpy as np
import math

from scipy.interpolate import interp1d

from general import secondary_electron_contribution_array, n_vectorized, energy_conv_to_array, extend_array
from spectral_functions import A_BCS, A_BCS_2, A_BCS_3


def Norman_EDC_array(w_array, scale, T, dk, s, a, c, fixed_k, energy_conv_sigma, convolution_extension=None):
    """
        EDC slice function
        :param w_array: energy array
        :param scale: scaling factor
        :param T:
        :param dk:
        :param s:
        :param a:
        :param c:
        :param fixed_k: momentum of EDC
        :param energy_conv_sigma:
        :param temp:
        :param convolution_extension:
        :return:
        """
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 2.5)  # between 96% and 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    temp_array = energy_conv_to_array(temp_w_array, np.multiply(
        A_BCS(fixed_k, temp_w_array, a, c, dk, T), scale),
                                      energy_conv_sigma)
    return_array = temp_array[convolution_extension:convolution_extension + len(w_array)] + s
    return return_array

def test_EDC_array_with_SE(w_array, scale, T, T0, dk, p, q, r, s, a, c, fixed_k, energy_conv_sigma, temp,
                      convolution_extension=None):
    return_array = test_EDC_array(w_array, scale, T, T0, dk, a, c, fixed_k, energy_conv_sigma, temp,
                             convolution_extension=convolution_extension)
    # add in secondary electrons
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(len(w_array)):
        return_array[i] = return_array[i] + secondary[i]
    return return_array


def test_EDC_array(w_array, scale, T, T0, dk, a, c, fixed_k, energy_conv_sigma, temp, convolution_extension=None):
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 2.5)  # between 96% and 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    temp_array = energy_conv_to_array(temp_w_array, np.multiply(
        A_BCS_3(fixed_k, temp_w_array, a, c, dk, T, T0) * n_vectorized(temp_w_array, temp), scale),
                                      energy_conv_sigma)
    return_array = temp_array[convolution_extension:convolution_extension + len(w_array)]
    return return_array


def EDC_array_with_SE(w_array, scale, T, dk, p, q, r, s, a, c, fixed_k, energy_conv_sigma, temp,
                      convolution_extension=None):
    """
    EDC slice function with secondary electron contribution
    :param w_array: energy array
    :param scale: scaling factor
    :param T:
    :param dk:
    :param p: SE scale
    :param q: SE horizontal shift
    :param r: SE steepness
    :param s: SE vertical shift
    :param a:
    :param c:
    :param fixed_k: momentum of EDC
    :param energy_conv_sigma:
    :param temp:
    :param convolution_extension:
    :return:
    """
    return_array = EDC_array(w_array, scale, T, dk, a, c, fixed_k, energy_conv_sigma, temp,
                             convolution_extension=convolution_extension)
    # add in secondary electrons
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(len(w_array)):
        return_array[i] = return_array[i] + secondary[i]
    return return_array


def EDC_array(w_array, scale, T, dk, a, c, fixed_k, energy_conv_sigma, temp, convolution_extension=None):
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
    :param convolution_extension:
    :return:
    """
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 2.5)  # between 96% and 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    temp_array = energy_conv_to_array(temp_w_array, np.multiply(
        A_BCS(fixed_k, temp_w_array, a, c, dk, T) * n_vectorized(temp_w_array, temp), scale),
                                      energy_conv_sigma)
    return_array = temp_array[convolution_extension:convolution_extension + len(w_array)]
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


def symmetrize_EDC(axis_array, data_array):
    """
    Symmetrize an EDC by copying its values over w=0
    Returns new axis_array and new data_array
    :param axis_array: energy array
    :param data_array: EDC
    :return:
    """
    # count how many extra positive or negative axis indices there are
    extra_positive = 0
    for i in range(len(axis_array)):
        if axis_array[i] > 0:
            extra_positive += 1
        elif axis_array[i] < 0:
            extra_positive -= 1
    if extra_positive >= 0:
        cropped_axis_array = axis_array[extra_positive:]
    else:
        cropped_axis_array = axis_array[:extra_positive]

    one_side_length = min(cropped_axis_array[0], math.fabs(cropped_axis_array[len(cropped_axis_array) - 1]))
    step_size = (2 * one_side_length) / len(cropped_axis_array)
    new_axis_array = np.arange(one_side_length, -one_side_length-0.01, -step_size)
    new_data_array = np.zeros(len(new_axis_array))

    f = interp1d(axis_array, data_array, kind='quadratic')
    return f(new_axis_array), new_axis_array

    # def interpolate_point(value):
    #     # Assumes value is greater than smallest axis_array value, and smaller than largest axis_array value
    #     for i in range(len(axis_array)):
    #         if math.fabs(value - axis_array[i]) < 0.00001:
    #             return data_array[i]
    #         elif (axis_array[i] < value < axis_array[i + 1]) or (axis_array[i] > value > axis_array[i + 1]):
    #             total_distance = math.fabs(axis_array[i+1] - axis_array[i])
    #             return data_array[i] * math.fabs(axis_array[i+1] - value) / total_distance + data_array[i+1] * math.fabs(axis_array[i] - value) / total_distance
    #
    # for i in range(len(new_data_array)):
    #     new_data_array[i] = interpolate_point(new_axis_array[i]) + interpolate_point(-new_axis_array[i])
    # return new_axis_array, new_data_array