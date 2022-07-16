import math
import matplotlib.pyplot as plt
import numpy as np

##################################################
# GENERAL FUNCTIONS AND CONSTANTS
##################################################

# Constants
H_BAR = 6.582 * 10 ** (-13)  # meV*s
ELECTRON_MASS = 5.1100 * 10 ** 8  # mev/c^2
LIGHT_SPEED = 2.998 * 10 ** 8  # m/s
ONE_BILLION = 1000000000


# Gaussian Function
def R(dw, sigma):
    """
    Gaussian distribution for energy convolution
    :param dw: distance on energy axis between energy convolution point and nearby point
    :param sigma: energy resolution (not FWHM)
    :return: a float representing convolution factor
    """
    try:
        return (1 / sigma / math.sqrt(2 * math.pi)) * np.exp(-0.5 * dw ** 2 / sigma ** 2)
    except OverflowError:
        return 0


def energy_conv_map(k, w, spectrum, energy_conv_sigma, scale):
    """
    Applies energy convolution to a map
    :param k: array of momentum
    :param w: array of energy
    :param spectrum: function (k,w) representing map
    :param energy_conv_sigma: energy resolution (sigma not FWHM)
    :param scale: factor to scale-up resulting map by
    :return: convoluted map
    """
    # k should be 2d
    height = int(k.size / k[0].size)
    width = k[0].size

    results = np.zeros((height, width))

    # Extract vertical arrays from 2d array to convolve over w
    inv_k = np.array([list(i) for i in zip(*k)])
    inv_w = np.array([list(i) for i in zip(*w)])
    # Flip vertically
    rev_inv_w = np.flip(inv_w)

    for i in range(height):
        for j in range(width):
            # 1D array of w at point (i, j) (in numpy coordinates); w is used to find dw
            curr_w = np.full(inv_w[j].size, w[i][j])
            # Energy convolution to find final intensity at point (i, j)
            res = np.convolve(
                R(rev_inv_w[j] - curr_w, energy_conv_sigma), scale * spectrum(inv_k[j], inv_w[j]), mode='valid')
            results[i][j] = res

    return results


def energy_conv_to_array(w, spectral_slice, energy_conv_sigma):
    """
    Energy Convolution to an array
    :param w: energy array
    :param spectral_slice: array to be convolved
    :param energy_conv_sigma: energy resolution (sigma not FWHM)
    :return: convoluted array
    """
    results = np.zeros(w.size)

    # Flip vertically to convolve properly
    rev_w = np.flip(w)

    for i in range(w.size):
        curr_w = np.full(w.size, w[i])
        res = np.convolve(spectral_slice, R(rev_w - curr_w, energy_conv_sigma), mode='valid')
        results[i] = res

    return results


def add_noise(spectrum):
    """
    Adds poisson noise to a map
    :param spectrum: the map to add noise to
    :return: the map with noise
    """
    height = math.floor(spectrum.size / spectrum[0].size)
    width = spectrum[0].size
    for i in range(height):
        for j in range(width):
            spectrum[i][j] = np.random.poisson(spectrum[i][j])
    return


def n(w, temp):
    """
    # Fermi-Dirac Function (assumes uP=0)
    :param w: energy at point
    :param temp: temperature of experiment
    :return: Fermi-Dirac factor
    """
    # Boltzmann's constant (meV/K)
    kB = 8.617333262 * 10 ** (-2)
    uP = 0
    # h-bar: 6.582 * 10 ** (-13) (mev*s) # (Implicit bc already expressing energy)
    if ((w - uP) / kB / temp) > 100:
        return 0  # Rounding approximation
    if (w - uP) / kB / temp < -100:
        return 1
    return 1 / (math.exp((w - uP) / kB / temp) + 1)


n_vectorized = np.vectorize(n)


def secondary_electron_contribution_array(w_array, p, q, r, s):
    """
    Model for secondary electron effect as sigmoid function
    :param w_array: energy array
    :param p: scale
    :param q: horizontal shift
    :param r: steepness
    :param s: vertical shift
    :return:
    """
    return_array = np.zeros(len(w_array))

    # p is scale-up factor (0, inf), q is horizontal shift (-inf, inf), r is steepness (-inf, 0]
    for i in range(len(w_array)):
        if r * w_array[i] - r * q > 100:
            return_array[i] = s
        else:
            return_array[i] = p / (1 + math.exp(r * w_array[i] - r * q)) + s

    return return_array


def reduced_chi(data, fit, absolute_sigma_squared, DOF):
    """
    Reduced-Chi Calculation
    :param data: true data
    :param fit: fitted predictions
    :param absolute_sigma_squared: variance (standard deviation squared)
    :param DOF: degrees of freedom (number points - number parameters)
    :return: reduced chi value (1 is good, >1 is bad, <1 is overfit)
    """
    res = 0
    for i in range(len(data)):
        res += (data[i] - fit[i]) ** 2 / absolute_sigma_squared[i]
    return res / DOF


def F_test(data, fit1, para1, fit2, para2, absolute_sigma_squared, n):
    """
    F-Test Calculation for comparing nested models
    :param data: true data
    :param fit1: smaller model fitted predictions
    :param para1: number of parameters in smaller model
    :param fit2: larger model fitted predictions
    :param para2: number of parameters in larger model
    :param absolute_sigma_squared: variance (standard deviation squared)
    :param n: number of data points
    :return:
    """
    # fit1 should be 'nested' within fit2
    if para2 <= para1:
        return ValueError
    rss1 = reduced_chi(data, fit1, absolute_sigma_squared, 1)
    rss2 = reduced_chi(data, fit2, absolute_sigma_squared, 1)
    return ((rss1 - rss2) / (para2 - para1)) / (rss2 / (n - para2))


def gaussian_form_normalized(x, sigma, mu):
    """
    Gaussian Function (Normalized)
    :param x: input
    :param sigma: Gaussian width
    :param mu: horizontal shift
    :return: Normalized Gaussian evaluated at input
    """
    return 1 / (sigma * (2 * math.pi) ** 0.5) * math.e ** (-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_form(x, a, b, c):
    """
    Gaussian Function (General)
    :param x: input
    :param a: scale
    :param b: horizontal shift
    :param c: width
    :return: Gaussian evaluated at input
    """

    return a * math.e ** ((- (x - b) ** 2) / (2 * c ** 2))


def lorentz_form(x, a, b, c, d):
    """
    Lorentz Function with vertical shift
    :param x: input
    :param a: scale
    :param b: horizontal shift
    :param c: width
    :param d: vertical shift
    :return: Lorentz with vertical shift evaluated at input
    """

    return a * c / ((x - b) ** 2 + c ** 2) + d


def lorentz_form_with_secondary_electrons(x, a, b, c, p, q, r, s, temp):
    lorentz = lorentz_form(x, a, b, c, 0) * n_vectorized(x, temp)
    secondary = secondary_electron_contribution_array(x, p, q, r, s)
    # output = np.zeros(len(x))
    # for i in range(len(x)):
    #     output[i] = lorentz[i] + secondary[i]
    return lorentz+secondary


def parabola(x, a, b, c):
    return a * (x - b) ** 2 + c


def w_as_index(input_w, w):
    """
    Convert w in meV to corresponding index
    :param input_w: w to convert
    :param w: energy array
    :return: index in w corresponding to input_w
    """
    return int(round((input_w - min(w)) / (max(w) - min(w)) * (w.size - 1)))


def k_as_index(input_k, k):
    """
    Convert k in A^-1 to corresponding index
    :param input_k: k to convert
    :param k: momentum array
    :return: index in k corresponding to input_k
    """
    return int(round((input_k - min(k)) / (max(k) - min(k)) * (k.size - 1)))


def d1_polynomial(x, a):
    return np.full(len(x), a)


def d2_polynomial(x, a, b):
    return a * x + b


def d3_polynomial(x, a, b, c):
    return a * x ** 2 + b * x + c


def d4_polynomial(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def d5_polynomial(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def d6_polynomial(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


def d7_polynomial(x, a, b, c, d, e, f, g):
    return a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2 + f * x + g


def d8_polynomial(x, a, b, c, d, e, f, g, h):
    return a * x ** 7 + b * x ** 6 + c * x ** 5 + d * x ** 4 + e * x ** 3 + f * x ** 2 + g * x + h


def d9_polynomial(x, a, b, c, d, e, f, g, h, i):
    return a * x ** 8 + b * x ** 7 + c * x ** 6 + d * x ** 5 + e * x ** 4 + f * x ** 3 + g * x ** 2 + h * x + i


def d10_polynomial(x, a, b, c, d, e, f, g, h, i, j):
    return a * x ** 9 + b * x ** 8 + c * x ** 7 + d * x ** 6 + e * x ** 5 + f * x ** 4 + g * x ** 3 + h * x ** 2 + i * x + j


def d11_polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
    return a * x ** 10 + b * x ** 9 + c * x ** 8 + d * x ** 7 + e * x ** 6 + f * x ** 5 + g * x ** 4 + h * x ** 3 + i * x ** 2 + j * x + k


def d12_polynomial(x, a, b, c, d, e, f, g, h, i, j, k, l):
    return a * x ** 11 + b * x ** 10 + c * x ** 9 + d * x ** 8 + e * x ** 7 + f * x ** 6 + g * x ** 5 + h * x ** 4 + i * x ** 3 + j * x ** 2 + k * x + l


def get_degree_polynomial(i):
    options = {1: d1_polynomial,
               2: d2_polynomial,
               3: d3_polynomial,
               4: d4_polynomial,
               5: d5_polynomial,
               6: d6_polynomial,
               7: d7_polynomial,
               8: d8_polynomial,
               9: d9_polynomial,
               10: d10_polynomial,
               11: d11_polynomial,
               12: d12_polynomial}
    return options[i]


def plot_map(Z, x, y):
    im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(x), max(x), min(y), max(y)])  # drawing the function
    plt.colorbar(im)


def rss(arr1, arr2):
    return np.sum(np.square(arr1 - arr2))


def extend_array(array, one_side_extension):
    """
    Extends an array with constant step size by one_side_extension on both sides
    :param array: must have length of at least 2
    :param one_side_extension: how many indexes to extend array by
    :return:
    """
    array_size = len(array)
    assert array_size >= 2
    step_size = array[1] - array[0]
    a = np.arange(array[0] - one_side_extension * step_size, array[0], step_size)
    b = np.arange(array[array_size - 1] + step_size, array[array_size - 1] + (one_side_extension + 1) * step_size,
                  step_size)
    return np.concatenate((a, array, b))


polynomial_functions = [d1_polynomial, d2_polynomial, d3_polynomial, d4_polynomial,
                        d5_polynomial, d6_polynomial, d7_polynomial, d8_polynomial, d9_polynomial,
                        d10_polynomial, d11_polynomial, d12_polynomial]

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]