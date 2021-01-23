import numpy as np
from scipy.stats import skew, kurtosis

__all__ = ['sky_noise_error', 'propagate_noise_error', 'mcnoise']


def sky_noise_error(nu_obs, nu_emit, nu_ch_bw, tint, a_eff, n_station, bmax):
    """Calculate instrument noise error of an interferometer.

    This assume that Tsys is dominated by Tsky.
    (see Furlanetto et al. (2006) section 9)

    Parameters
    ----------
    nu_obs : float or array-like
        Observing frequency in [MHz].
        Can be array-like to compute noise at multiple frequencies.
    nu_emit : float
        Emitted frequency of the observed spectral line in [MHz].
    nu_ch_bw : float
        Observed frequency channel bandwidth in [MHz].
    tint : float, optional
        Integration time. Default is 1000. [hours]
    a_eff : float
        Effective area of a station in [m**2].
    n_station : integer
        Number of antennas (or stations in case of a phase-array).
    bmax : float
        Maximum baseline length of the array in [wavelength].
    Returns
    -------
    Noise error (standard deviation) in [mK] in the same format as nu_obs.

    """
    nu_obs = np.asarray(nu_obs)
    a_tot = a_eff * n_station
    z = (nu_emit / nu_obs) - 1.
    theta = 1.22 / bmax * 60 * 180 / np.pi
    err = 2.9 * (1.e5 / a_tot) * (10. / theta) ** 2 * \
        ((1 + z) / 10.0) ** 4.6 * np.sqrt(100. / (nu_ch_bw * tint))
    return err


def propagate_noise_error(noise_err, m2, m3, m4, m6, npix):
    """Analytically propagate error to variance and skewness.

    Based on error propagation described in the appendix of
    Watkinson & Pritchard (2014)

    Parameters
    ----------
    noise_err : float or array-like
        Noise error.
    m2 : float
        2nd moment of the data
    m3 : float
        3rd moment of the data
    m4 : float
        4th moment of the data
    m6 : float
        6th moment of the data
    npix : int
        Number of pixels in the data
    Returns
    -------
    Error of 2nd moment, 3rd moment and skewness.

    """
    noise_var = np.asarray(noise_err) ** 2
    m2_var = (2. / npix) * (2 * m2 * noise_var + noise_var ** 2)
    m3_var = (3. / npix) * (3 * noise_var * m4 + 12 * m2 * noise_var ** 2 +
                            5 * noise_var ** 3)
    m4_var = (8. / npix) * (2 * m6 * noise_var + 21 * m4 * noise_var ** 2 +
                            48 * m2 * noise_var ** 3 + 12 * noise_var ** 4)
    m2m3_cov = (6 / npix) * m3 * noise_var
    m2m4_cov = (4. / npix) * (2 * m4 * noise_var + 9 * m2 * noise_var ** 2 +
                              3 * noise_var ** 3)
    skew_var = (m3_var / (m2 ** 3)) + \
               ((9 * m3 ** 2 * m2_var) / (4 * m2 ** 5)) - \
               (3 * m3 * m2m3_cov / (m2 ** 4))
    kurt_var = (1. / m2 ** 4) * m4_var + 4 * (m4 ** 2 / m2 ** 6) * m2_var - \
        4 * (m4 / m2 ** 5) * m2m4_cov
    return np.sqrt(m2_var), np.sqrt(skew_var), np.sqrt(kurt_var)


def mcnoise(data, noise_std, n, noise_scaling=1.):
    """
    Parameters
    ----------
    data : ndarray
        Array of data.
    noise_std : float
        Standard deviation of the noise
    n : int
        Number of repetition
    noise_scaling: float
        Scaling factor for noise

    Returns
    -------
    variance, variance error, skewness, skewness error, kurtosis, kurtosis error

    """
    noise_arr = np.random.normal(0, noise_std, (n, data.size)) * noise_scaling
    var_sample = np.var(data + noise_arr, axis=1)
    skew_sample = skew(data + noise_arr, axis=1)
    kurt_sample = kurtosis(data + noise_arr, axis=1)
    var_val = np.mean(var_sample)
    skew_val = np.mean(skew_sample)
    kurt_val = np.mean(kurt_sample)
    var_err = np.std(var_sample)
    skew_err = np.std(skew_sample)
    kurt_err = np.std(kurt_sample)
    return var_val, var_err, skew_val, skew_err, kurt_val, kurt_err
