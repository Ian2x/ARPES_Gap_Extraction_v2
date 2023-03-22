import numpy as np
from math import fabs
from general import secondary_electron_contribution_array, n_vectorized, energy_conv_to_array, extend_array, lorentz, \
    gaussian
from spectral_functions import A_BCS, A_BCS_3, final_A_BCS, A_BCS_2, final_A_BCS_2, final_A_BCS_3


def Norman_EDC_array(w_array, scale, loc, dk, T1, T0, s, energy_conv_sigma, convolution_extension=None):
    """
        EDC slice function
        :param w_array: energy array
        :param scale: scaling factor
        :param loc:
        :param dk:
        :param T1:
        :param T0:
        :param s:
        :param energy_conv_sigma:
        :param convolution_extension:
        :return:
        """
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 4)  # >> 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    temp_array = energy_conv_to_array(temp_w_array, np.multiply(
        final_A_BCS_3(temp_w_array, loc, dk, T1, T0) + final_A_BCS_3(-temp_w_array, loc, dk, T1, T0), scale),
                                      energy_conv_sigma)
    return_array = temp_array[convolution_extension:convolution_extension + len(w_array)] + s
    return return_array


def Norman_EDC_array3(w_array, scale, loc, dk, T1, s, energy_conv_sigma, convolution_extension=None):
    """
        EDC slice function
        :param w_array: energy array
        :param scale: scaling factor
        :param loc:
        :param dk:
        :param T1:
        :param s:
        :param energy_conv_sigma:
        :param convolution_extension:
        :return:
        """
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 4)  # >> 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    temp_array = energy_conv_to_array(temp_w_array, np.multiply(
        final_A_BCS(temp_w_array, loc, dk, T1) + final_A_BCS(-temp_w_array, loc, dk, T1), scale),
                                      energy_conv_sigma)
    return_array = temp_array[convolution_extension:convolution_extension + len(w_array)] + s
    return return_array


def Norman_EDC_array2(w_array, a, b, c, s, energy_conv_sigma, convolution_extension=None, noConvolute=False):
    """
        EDC slice function
        :param w_array: energy array
        :param a:
        :param b:
        :param c:
        :param s:
        :param energy_conv_sigma:
        :param convolution_extension:
        :param noConvolute:
        :return:
        """
    if noConvolute:
        return gaussian(w_array, a, b, c, 0) + gaussian(-w_array, a, b, c, 0) + s
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 4)  # >> 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    temp_array = energy_conv_to_array(temp_w_array, gaussian(temp_w_array, a, b, c, 0) + gaussian(-temp_w_array, a, b, c, 0), energy_conv_sigma)
    return_array = temp_array[convolution_extension:convolution_extension + len(w_array)] + s
    return return_array


# def EDC_array_with_SE(w_array, scale, T0, T1, dk, p, q, r, s, a, c, fixed_k, energy_conv_sigma, temp,
#                       convolution_extension=None):
#     return_array = EDC_array(w_array, scale, T0, T1, dk, a, c, fixed_k, energy_conv_sigma, temp,
#                              convolution_extension=convolution_extension)
#     # add in secondary electrons
#     secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
#     for i in range(len(w_array)):
#         return_array[i] = return_array[i] + secondary[i]
#     return return_array
#
#
# def EDC_array(w_array, scale, T0, T1, dk, a, c, fixed_k, energy_conv_sigma, temp, convolution_extension=None):
#     if convolution_extension is None:
#         convolution_extension = int(
#             energy_conv_sigma / (w_array[0] - w_array[1]) * 2.5)  # between 96% and 99% ? maybe...
#     temp_w_array = extend_array(w_array, convolution_extension)
#     temp_array = energy_conv_to_array(temp_w_array, np.multiply(
#         A_BCS_3(fixed_k, temp_w_array, a, c, dk, T0, T1) * n_vectorized(temp_w_array, temp), scale),
#                                       energy_conv_sigma)
#     return_array = temp_array[convolution_extension:convolution_extension + len(w_array)]
#     return return_array


def EDC_array_with_SE(w_array, scale, T, dk, p, q, r, s, a, c, fixed_k, energy_conv_sigma, temp,
                      convolution_extension=None, symmetrized=False, flat_SEC=False):
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
    base = EDC_array(w_array, scale, T, dk, a, c, fixed_k, energy_conv_sigma, temp,
                     convolution_extension=convolution_extension, symmetrized=symmetrized)

    # add in secondary electrons
    if flat_SEC:
        secondary = np.full(len(w_array), p)
    else:
        secondary = secondary_electron_contribution_array(w_array, p, q, r, s)

    height = len(w_array)
    result = np.zeros(len(base))
    for i in range(height):
        result[i] = base[i] + secondary[i]
        if symmetrized:
            result[i] += base[height - i - 1] + secondary[height - i - 1]
    return result


def final_EDC_array_with_SE(w_array, scale, T1,T0, dk, p, q, r, s, loc, energy_conv_sigma, temp):
    base = final_EDC_array(w_array, scale, T1, T0, dk, loc, energy_conv_sigma, temp)
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    result = np.zeros(len(base))
    for i in range(len(w_array)):
        result[i] = base[i] + secondary[i]
    return result


def EDC_array(w_array, scale, T, T2, dk, a, c, fixed_k, energy_conv_sigma, temp, convolution_extension=None,
              symmetrized=False):
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 4)  # >> 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    if symmetrized:
        temp_array = energy_conv_to_array(temp_w_array, np.multiply(
            A_BCS(fixed_k, temp_w_array, a, c, dk, T), scale),
                                          energy_conv_sigma)
    else:
        temp_array = energy_conv_to_array(temp_w_array, np.multiply(
            A_BCS(fixed_k, temp_w_array, a, c, dk, T) * n_vectorized(temp_w_array, temp), scale),
                                          energy_conv_sigma)
    return_array = temp_array[convolution_extension:convolution_extension + len(w_array)]
    return return_array


def final_EDC_array(w_array, scale, T1, T0, dk, loc, energy_conv_sigma, temp):
    # return scale * final_A_BCS(w_array, loc, dk, T)
    convolution_extension = int(energy_conv_sigma / (w_array[0] - w_array[1]) * 2.5)  # between 96% and 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    temp_array = energy_conv_to_array(temp_w_array, np.multiply(
            final_A_BCS_3(temp_w_array, loc, dk, T1, T0) * n_vectorized(temp_w_array, temp), scale), energy_conv_sigma)
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
    orig_arr_size = len(axis_array)
    orig_step_size = axis_array[0] - axis_array[1]

    # count how many extra positive or negative axis indices there are
    extra_positive = 0
    for i in range(orig_arr_size):
        if axis_array[i] > 0:
            extra_positive += 1
        elif axis_array[i] < 0:
            extra_positive -= 1
    if extra_positive >= 0:
        cropped_axis_array = axis_array[extra_positive:]
    else:
        cropped_axis_array = axis_array[:extra_positive]
    new_arr_size = len(cropped_axis_array)

    one_side_length = min(fabs(cropped_axis_array[0]), fabs(cropped_axis_array[new_arr_size - 1]))
    step_size = (2 * one_side_length) / (new_arr_size - 1)
    new_axis_array = np.arange(one_side_length, -one_side_length - 0.001, -step_size)
    new_data_array = np.zeros(new_arr_size)
    assert len(new_data_array) == len(new_axis_array) == new_arr_size

    curr_top = 0
    for i in range(new_arr_size):
        target = new_axis_array[i]
        while curr_top + 2 < orig_arr_size and axis_array[curr_top + 1] >= target:
            curr_top += 1
        if target == axis_array[curr_top + 1]:
            new_data_array[i] += data_array[curr_top + 1]
        elif axis_array[curr_top + 1] < target <= axis_array[curr_top]:
            new_data_array[i] += (data_array[curr_top] * abs(axis_array[curr_top + 1] - target) +
                                  data_array[curr_top + 1] * abs(axis_array[curr_top] - target)) / orig_step_size

    curr_bot = orig_arr_size - 1
    for i in range(new_arr_size):
        target = -new_axis_array[i]
        while curr_bot - 2 >= 0 and axis_array[curr_bot - 1] <= target:
            curr_bot -= 1
        if target == axis_array[curr_bot - 1]:
            new_data_array[i] += data_array[curr_bot - 1]
        elif axis_array[curr_bot] <= target < axis_array[curr_bot - 1]:
            new_data_array[i] += (data_array[curr_bot] * fabs(axis_array[curr_bot - 1] - target) +
                                  data_array[curr_bot - 1] * fabs(axis_array[curr_bot] - target)) / orig_step_size
    return new_axis_array, new_data_array

def symmetrize_Z(axis_array, Z):
    inv_Z = np.array([list(i) for i in zip(*Z)])
    new_Z = []
    new_axis_array = None
    for EDC in inv_Z:
        new_axis_array = symmetrize_EDC(axis_array, EDC)[0]
        new_Z.append(symmetrize_EDC(axis_array, EDC)[1])
    return new_axis_array, np.array(new_Z).T


def estimated_peak_movement(k, a, c, dk):
    norm = a * k ** 2 + c
    sc = (norm ** 2 + dk ** 2) ** 0.5
    return np.abs(norm - sc)


