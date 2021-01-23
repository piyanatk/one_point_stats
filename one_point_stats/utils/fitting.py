import numpy as np
from astropy.modeling import fitting, models

__all__ = ['fit_gaussian1d', 'fit_multi_gaussian1d']


def fit_gaussian1d(x, y, init_amp=1.0, init_mean=0.0, init_std=0.45,
                   xcrop=None):
    """Fit a Gaussian 1D kernel to the data.

    Parameters
    ----------
    x: array-like
        x values.
    y: array-like
        y values.
    init_amp: float, optional
        Initial amplitude parameter for the fit. Default 1.0
    init_mean: float, optional
        Initial mean parameter for the fit. Default 0.0
    init_std: float, optional
        Initial standard deviation parameter for the fit. Default 0.45.
    xcrop: (min, max) or None, optional
        Crop input to min <= angle <= max.

    Returns
    -------
    amp: float
        Amplitude of the fitted Gaussian.
    mean: float
        Mean of the fitted Gaussian.
    std: float
        Standard deviation of the fitted Gaussian.

    """
    fitter = fitting.LevMarLSQFitter()
    g1_init = models.Gaussian1D(init_amp, init_mean, init_std)
    if xcrop is not None:
        mask = (x >= xcrop[0]) & (x <= xcrop[1])
    else:
        mask = np.ones_like(x, dtype=bool)
    g1 = fitter(g1_init, x[mask], y[mask])
    return g1.amplitude.value, g1.mean.value, g1.stddev.value


def fit_multi_gaussian1d(x, y, init_params):
    """
    Fit PSF radial profile with a compound Gaussian model.

    Parameters
    ----------
    x: array-like
        Radial profile of the PSF.
    y: array-like
        Radius for the profile in arc minute.
    init_params: tuple of tuples
        (height, mean, std) for each Gaussian modeling given as a tuple inside
        a tuple. For example:
        A single Gaussian: init_params=((h1, m1, std1),)
        Two gaussian: init_params=((h1, m1, std1), (h2, m2, std2))

    Returns
    -------
    out: list
        List of astropy Gaussian1D model from the fit.

    """
    fitter = fitting.LevMarLSQFitter()
    g_init = []
    for p in init_params:
        g_init.append(models.Gaussian1D(p[0], p[1], p[2]))
    g_comp_init = np.sum(g_init)
    g_comp = fitter(g_comp_init, np.hstack((-y[::-1], y)),
                    np.hstack((x[::-1], x)))
    out = []
    for i in range(np.size(g_comp)):
        out.append([g_comp[i].amplitude.value, g_comp[i].mean.value,
                    g_comp[i].stddev.value])
    return out
