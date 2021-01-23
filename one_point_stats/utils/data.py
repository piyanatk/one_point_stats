import numpy as np
from .settings import MWA_FREQ_EOR_ALL_80KHZ

__all__ = ['bin_freqs', 'crop_arr', 'gen_radius_array', 'gen_radial_mask',
           'is_empty_list', 'radial_profile', 'LazyProperty',
           'get_channel_indexes_per_bin']


class LazyProperty(object):
    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, klass=None):
        if obj is None:
            return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result


def radial_profile(data, center, scale=(1, 1), normalized=False,
                   max_r='circle'):
    """
    Compute radial sum and radial average of a 2D array.

    Parameters
    ----------
    data: 2D numpy ndarray
        Data to calculate the profile
    center: (float, float)
        (x-center, y-center) of data
    scale: (float, float), optional
        Scaling factor to be multiplied to x and y. default is (1, 1)
    normalized: bool, optional
        Normalize the outputs by their maxima. default is false
    max_r: {float, 'circle', 'all'}
        Maximum radial distance.
        float will limit the radial distance to that number.
        'circle' will limit the radius to full concentric circle
        'all' will includes all radius, including those not in a full circle
        Everything else will use 'all'

    Returns
    -------
    radial_sum: numpy ndarray of ints
        Radial sum profile. lenght = numpy.amax(r) + 1
    radial_avg: numpy ndarray of ints
        Radial average profile. length = numpy.amax(r) + 1
    r: numpy ndarray of ints
        Radius corresponding to the profiles

    Note
    ----
    Due to the use of numpy.bincount, the length of an output array is limited
    to numpy.amax(r) + 1, where:
        r = np.sqrt(((x - center[0]) * scale[0]) ** 2
                    + ((y - center[1]) * scale[1]) ** 2)
    """
    y, x = np.indices(data.shape)
    r = np.sqrt(((x - center[0]) * scale[0]) ** 2 +
                ((y - center[1]) * scale[1]) ** 2)
    r = r.astype('int')
    radial_sum = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radial_avg = radial_sum / nr
    if normalized:
        radial_sum /= radial_sum.max()
        radial_avg /= radial_avg.max()
    if isinstance(max_r, (float, int)):
        cut = slice(0, max_r)
    elif max_r == 'circle':
        max_r_ = np.min(
            [np.array([center[0] - 0, data.shape[0] - center[0]]) * scale[0],
             np.array([center[1] - 0, data.shape[1] - center[1]]) * scale[1]]
        )
        cut = slice(0, max_r_)
    else:
        cut = slice(0, -1)
    return radial_sum[cut], radial_avg[cut], np.arange(np.amax(r) + 1)[cut]


def is_empty_list(inlist):
    if isinstance(inlist, list):
        out = all(map(is_empty_list, inlist))
    else:
        out = False
    return out


def crop_arr(in_arr, mask):
    crop_length = np.ceil(mask.sum(0).max() / 2.)
    xcen, ycen = np.array(in_arr.shape) / 2.
    crop_slice = np.s_[xcen-crop_length:xcen+crop_length,
                       ycen-crop_length:ycen+crop_length]
    in_arr[~mask] = np.nan
    out_arr = in_arr[crop_slice]
    return out_arr


def gen_radius_array(shape, center, xy_scale=None, r_scale=None):
    """
    Make a 2D array of radius values from a specific center.

    Parameters
    ----------
    shape: (int, int)
        Size for the x and y dimensions of the array in pixels.
    center: (float, float)
        Center for the x and y dimensions of the array in pixels.
    xy_scale: float or (float, float), optional
        Scale factor to apply to x and y dimension before generating a mask.
        If a single number is given, this is a scale factor for both x and y.
        If a (float, float) is given, these are (x scale, y scale).
    r_scale: float, optional
        Scale factor to apply to radius.
        Overwrite `xy_scale`, i.e. the radius will be calculate in pixels unit
        before applying this scaling factor.

    Returns
    -------
    r: array
        Radius array

    """
    # Figure out all the scaling complexity
    if r_scale is not None:
        rscale = r_scale
        xscale = 1
        yscale = 1
    else:
        if isinstance(xy_scale, (tuple, list, np.ndarray)):
            rscale = 1
            xscale = xy_scale[0]
            yscale = xy_scale[1]
        elif isinstance(xy_scale, (float, int)):
            rscale = 1
            xscale = xy_scale
            yscale = xy_scale
        else:
            rscale = 1
            xscale = 1
            yscale = 1
    x = (np.arange(shape[0]) - center[0]) * xscale
    y = (np.arange(shape[1]) - center[1]) * yscale
    r = np.sqrt(x[:, np.newaxis] ** 2 + y ** 2) * rscale
    return r


def gen_radial_mask(shape, center, radius, mask=True,
                    xy_scale=None, r_scale=None):
    """
    Generate a 2D radial mask array.

    Pixels within the radius=(rmin, rmax) from a specified center will
    be masked by to the value in `mask`.

    Parameters
    ----------
    shape: (int, int)
        Size for the x and y dimensions of an array in pixels.
    center: (float, float)
        Center for the x and y dimensions of an array in pixels.
    radius: (float, float)
        Minimum and maximum radius of the masking region from the center
        of an array. This region will be masked with the value specified in
        `mask`. xy_scale` and/or `r_scale` will be applied before masking.
    xy_scale: float or (float, float), optional
        Scale factor to apply to x and y dimension before generating a mask.
        If a single number is given, it will be used for both x and y.
        If a (float, float) is given, these are (x scale, y scale).
    r_scale: float, optional
        Scale factor to apply to radius.
        Overwrite `xy_scale`, i.e. the radius will be calculate in pixels unit
        before applying this scaling factor.
    mask: {True, False}, optional
        Weather to set the mask region to `True` or `False`.
        Default is True.

    Returns
    -------
    mask: bool array
        A boolean array of shape=`shape` with pixels within `radius` from the
        center pixels set to True, and False elsewhere.

    """
    r = gen_radius_array(shape, center, xy_scale=xy_scale, r_scale=r_scale)
    out = (r >= radius[0]) & (r <= radius[1])
    return out if mask else np.logical_not(out)


def bin_freqs(bin_width, native_channel_width=0.08, freqs_list=None):
    """Group frequency channels into bins.

    Parameters
    ----------
    bin_width : float
        Frequency bandwidth of the bin
    native_channel_width : float
        Native frequency bandwidth of the channel
    freqs_list : array-like
        List of frequency channels. Must be in ascending order.
        Assume continuous frequency channels.

    Returns
    -------
    ch_list_per_bin: list
        List of ndarray containing frequency channels per bin.
    bin_centers: ndarray
        Center frequency of each bin

    """
    if freqs_list is None:
        freqs_list = MWA_FREQ_EOR_ALL_80KHZ
    if bin_width == native_channel_width:
        ch_list_per_bin = [[f] for f in freqs_list]
    else:
        nchannel_per_bin = int(np.ceil(bin_width / native_channel_width))
        # nbins = int(np.ceil(len(freqs) / float(nchannel_per_bin)))
        ch_list_per_bin = np.array_split(
            freqs_list, np.arange(len(freqs_list), 0, -nchannel_per_bin)
            [-1:0:-1]
        )
    bin_centers = np.array([(f[0] + f[-1]) / 2. for f in ch_list_per_bin]).ravel()
    out = (ch_list_per_bin, bin_centers)
    return out


def get_channel_indexes_per_bin(bin_width, native_channel_width, nchannels):
    nchannel_per_bin = int(np.ceil(bin_width / native_channel_width))
    channel_indexes_per_bin = np.array_split(
        np.arange(nchannels), np.arange(nchannels, 0, -nchannel_per_bin)
        [-1:0:-1]
    )
    return channel_indexes_per_bin
