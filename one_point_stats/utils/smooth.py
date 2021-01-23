import numpy as np
from astropy.convolution import convolve_fft, Gaussian2DKernel

__all__ = ['gaussian_smooth', 'smooth_healpix', 'sum_gaussian_smoothed_maps']


def smooth_healpix():
    pass


def gaussian_smooth(arr, kernel_stddev, boundary='wrap',
                    normalize_kernel=None):
    """Convolve 2D data array with a 2D Gaussian kernel.

    Parameters
    ----------
    arr: array-like
        Data array
    kernel_stddev: float
        Standard deviation of the Gaussian kernel
    boundary: {'fill', 'wrap'}, optional
        A flag indicating how to handle boundaries:
        * 'fill': set values outside the array boundary to fill_value
        * 'wrap': periodic boundary (default)
    normalize_kernel: {'peak', 'integral'}, optional
        Normalization of the Gaussian kernel. Default is 'integral'

    """
    k = Gaussian2DKernel(kernel_stddev)
    if normalize_kernel:
        k.normalize(normalize_kernel)
    return convolve_fft(arr, k, boundary=boundary, allow_huge=True)


def sum_gaussian_smoothed_maps(maps, heights, sigmas):
    """
    Sum Gaussian smoothed maps weighted by their smoothing kernels.

    Parameters
    ----------
    maps: tuple of 2D ndarray
        Input maps passed inside a tuple. Each map must have the same dimension.
    heights: scalar or tuple of scalar
        Heights of the Gaussian kernels used to smooth the input maps.
        If scalar, the same height will be used for all maps.
    sigmas: scalar, tuple of scalar, or tuple of tuple of scalar
        Standard deviation of the Gaussian kernels used to smooth the
        input maps.
        If scalar, assume the same standard deviation in both x and y
        dimensions for all input maps.
        If tuple of scalar, assume that each input map uses the same standard
        deviation for both x and y dimension, and each of these values are
        the element of a tuple.
        If tuple of tuple of scalar, i.e. 2D tuple, each row represents the
        standard deviation used for each map, where the first column is for
        the x dimension and the second column is for the y dimension.

    Return
    ------
    out: ndarray
        The weighted sum of the input maps.

    """
    nmaps = len(maps)
    if isinstance(heights, (float, int)):
        h = np.ones(nmaps) * heights
    else:
        h = np.array(heights)
    if isinstance(sigmas, (float, int)):
        s = np.ones(nmaps) * sigmas
    else:
        s = np.array(sigmas)
    if s.ndim == 2:
        vol = 2 * np.pi * h * s[:, 0] * s[:, 1]
    else:  # assume scalar
        vol = 2 * np.pi * h * s * s
    return np.sum(np.dstack(maps) * vol, axis=2) / np.sum(vol)
