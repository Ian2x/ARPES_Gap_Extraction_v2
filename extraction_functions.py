import numpy as np

from general import secondary_electron_contribution_array, n_vectorized, energy_conv_to_array, extend_array
from spectral_functions import A_BCS, A_BCS_2


def EDC_array_with_SE(w_array, scale, T, dk, p, q, r, s, a, c, fixed_k, energy_conv_sigma, temp,
                      convolution_extension=None, use_Norman=False, symmetrize=False):
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
    :param use_Norman:
    :return:
    """
    return_array = EDC_array(w_array, scale, T, dk, a, c, fixed_k, energy_conv_sigma, temp,
                             convolution_extension=convolution_extension, use_Norman=use_Norman, symmetrize=symmetrize)
    # add in secondary electrons
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(len(w_array)):
        return_array[i] = return_array[i] + secondary[i]
    return return_array


def EDC_array(w_array, scale, T, dk, a, c, fixed_k, energy_conv_sigma, temp, convolution_extension=None,
              use_Norman=False, symmetrize=False):
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
    :param use_Norman:
    :return:
    """
    if convolution_extension is None:
        convolution_extension = int(
            energy_conv_sigma / (w_array[0] - w_array[1]) * 2.5)  # between 96% and 99% ? maybe...
    temp_w_array = extend_array(w_array, convolution_extension)
    if use_Norman:
        # Use A_BCS_2
        if symmetrize:
            # Don't include Fermi effect
            temp_array = energy_conv_to_array(temp_w_array, np.multiply(
                A_BCS_2(fixed_k, temp_w_array, a, c, dk, T), scale), energy_conv_sigma)
        else:
            # Include Fermi effect
            temp_array = energy_conv_to_array(temp_w_array, np.multiply(
                A_BCS_2(fixed_k, temp_w_array, a, c, dk, T) * n_vectorized(temp_w_array, temp), scale),
                                              energy_conv_sigma)
    else:
        # Use A_BCS
        temp_array = energy_conv_to_array(temp_w_array, np.multiply(
            A_BCS(fixed_k, temp_w_array, a, c, dk, T) * n_vectorized(temp_w_array, temp), scale), energy_conv_sigma)
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


def symmetrize_EDC(axis_array, data_array, temp, ignore_w_above=-4):
    """
    Symmetrize an EDC by copying its values over w=0
    Returns new axis_array and new data_array
    :param axis_array: energy array
    :param data_array: EDC
    :param temp:
    :param ignore_w_above: w values to ignore (because large Fermi effect)
    :return:
    """
    # ignore small temp values
    bad_indexes = []
    for i in range(len(axis_array)):
        if axis_array[i] > ignore_w_above:
            bad_indexes.append(i)
    bad_indexes.reverse()
    for i in bad_indexes:
        axis_array = np.delete(axis_array, i)
        data_array = np.delete(data_array, i)
    # remove fermi effect from data_array
    data_array = data_array / n_vectorized(axis_array, temp)
    # symmetrize values
    new_array_size = 2 * len(data_array)
    if 0 in axis_array:
        new_array_size - 1
    new_data_array = np.zeros(new_array_size)
    new_axis_array = np.zeros(new_array_size)
    for i in range(len(data_array)):
        new_data_array[2*i] = data_array[i]
        new_axis_array[2*i] = axis_array[i]
        if axis_array[i] != 0:
            new_data_array[2*i + 1] = data_array[i]
            new_axis_array[2*i+1] = -axis_array[i]
    # sort by axis
    return sorted(new_axis_array, reverse=True), [x for _, x in sorted(zip(new_axis_array, new_data_array), reverse=True)]