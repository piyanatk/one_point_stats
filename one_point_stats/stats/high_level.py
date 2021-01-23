"""Calculate moment statistics (variance, skewness and kurtosis) from
data sample. Errors on the moments are analytically calculate by assuming
Gaussian random noise in the data. The noise is assumed to be dominated
by the sky temperature."""
import numpy as np
from scipy.stats import moment

from .noise import propagate_noise_error

__all__ = ['get_stats']


def get_stats(data, noise_err=None, noise_err_npix=None):
    """ Calculate variance, skewness and kurtosis of the data with
    an option to propagate noise error.

    Flatten data array is used in the calculation.

    The noise is analytically added, assuming if the "true" data is
    corrupted by some random noise with standard deviation of `noise_err`.

    Propagated error of the kurtosis is not implemented at the moment.

    Parameters
    ----------
    data : array-like
        Data to calculate statistics.
    noise_err : float or array-like, optional
        Noise error to propagate to the statistics.
        If array-like, each noise error in the array will be propagate
        individually, returning multiple propagated errors for each
        noise error.
    noise_err_npix : float, optional
        Assumed number of independent npix in the noise calculation.
        Default is every pixel is independent.
    Returns
    -------
    ndarray([min, max, mean, variance, skewness, kurtosis,
            (noise1, var_err1, skew_err1, noise2, var_err2, skew_err2,
             noise3, var_err3, skew_err3, ... (if any))]

    """
    npix = np.size(data)
    dmin = np.min(data)
    dmax = np.max(data)
    dmean = np.mean(data)
    m2 = moment(data, moment=2, axis=None)
    m3 = moment(data, moment=3, axis=None)
    m4 = moment(data, moment=4, axis=None)
    skew = m3 / m2 ** (3. / 2.)
    kurt = (m4 / m2 ** 2) - 3.
    stat_vals = (dmin, dmax, dmean, m2, skew, kurt)
    if noise_err is not None:
        if noise_err_npix is None:
            noise_err_npix = npix
        m6 = moment(data, moment=6, axis=None)
        m2_err, skew_err, kurt_err = propagate_noise_error(
            noise_err, m2, m3, m4, m6, noise_err_npix
        )
        prop_noise_err = np.vstack(
            (noise_err, m2_err, skew_err, kurt_err)
        ).T.ravel()
        out = np.hstack((stat_vals, prop_noise_err))
    else:
        out = np.array(stat_vals)
    return out
