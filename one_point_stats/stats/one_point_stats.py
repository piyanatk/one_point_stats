import numpy as np
import xarray as xr
from scipy.stats import moment, skew, kurtosis

__all__ = ['sample_mean', 'sample_skewness', 'sample_kurtosis', 'xr_moment']


def xr_moment(x, dim, order=1):
    """Calculate statistical moment of an XArray DataArray object.

    Parameters
    ----------
    x : xarray object
    dim : str
        Dimension over which to calculate moment.
        Note: Only support calculation over a single dimension.
    order : int or array_like of ints, optional
        Order of central moment that is returned. Default is 1 (mean).

    Returns
    -------
    moment : Calculated moment as an XArray object.

    """
    return xr.apply_ufunc(
        moment, x, input_core_dims=[[dim]],
        kwargs={'moment': order, 'axis': -1, 'nan_policy': 'omit'},
        dask='parallelized', output_dtypes=[float]
    )


def sample_mean(data, error=True):
    dmean = np.mean(data)
    if error:
        mean_err = np.std(data, ddof=1) / np.sqrt(np.size(data))
        result = (dmean, mean_err)
    else:
        result = dmean
    return result


def sample_skewness(data, axis=None, bias=True, error=False):
    """
    Calculate sample skewness of data.

    Parameters
    ----------
    data : array-like
        Data array.
    axis : int or None, optional
        axis of the data array to computer skewness.
        Compute from the whole array if None.
    bias : bool, optional
        Computer biased of unbiased skewness. Default is True (biased).
    error : bool, optional
        Return standard error of the skewness if True. Default is False.
        Note that standard error is derived from the unbiased skewness.

    Returns
    -------
    skewness and standard error of skewness (if any)

    """
    dskew = skew(data, axis=axis, bias=bias)
    if error:
        nval = np.size(data)
        err = np.sqrt((6. * nval * (nval - 1)) /
                      ((nval - 2) * (nval + 1) * (nval + 3)))
        out = (dskew, err)
    else:
        out = dskew
    return out


def sample_kurtosis(data, axis=None, bias=True, error=False):
    """
    Calculate sample kurtosis of data.

    Parameters
    ----------
    data : array-like
        Data array.
    axis : int or None, optional
        axis of the data array to computer skewness.
        Compute from the whole array if None.
    bias : bool, optional
        Computer biased of unbiased skewness. Default is True (biased).
    error : bool, optional
        Return standard error of the skewness if True. Default is False.
        Note that standard error is derived from the unbiased skewness.

    Returns
    -------
    kurtosis and standard error of kurtosis (if any)

    """
    dkurt = kurtosis(data, axis=axis, bias=bias)
    if error:
        nval = np.size(data)
        err = np.sqrt((24. * nval * (nval - 1) ** 2) /
                      ((nval - 3) * (nval - 2) * (nval + 3) * (nval + 5)))
        out = (dkurt, err)
    else:
        out = dkurt
    return out
