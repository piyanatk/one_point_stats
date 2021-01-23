import numpy as np
from numpy.fft import ifftshift, fftshift, fftn
from astropy.io import fits
import astropy.constants as const
from astropy.modeling import models, fitting

from ..utils.conversion import gaussian_fwhm2std
from ..utils.data import gen_radial_mask, gen_radius_array


def gen_psf(uv, pad=True, unnormalize=True):
    """
    Generate a point spread function map from a UV weights map.

    Parameters
    ----------
    uv: array-like or string
        UV weights map.
        If string, assume a FITS file of UV weights.
        If array-like, assume an array of UV weights map.
    pad: bool, optional
        Padding the uv by its size before FFT.
        Default is True
    unnormalize: bool, optional
        Re-normalize the FFT result to un-normalized values.

    Note
    ----
        MAPS use the the forward transform in FFTW library (negative exponent)
        to FFT from an image plane to a UV plane. We use numpy.fft.ifft
        to do a reverse transform from the UV plane to an image plane.
        This keep the FFT sign consistent.
        We also re-normalize the numpy ifft to make it returns
        an un-normalized result.

    """
    if isinstance(uv, str):
        uv_arr = fits.getdata(uv)
    else:
        uv_arr = uv
    if pad:
        s = np.array(uv_arr.shape) / 2
        uv_arr = np.pad(uv_arr, pad_width=((s[0], s[1]), (s[0], s[1])),
                        mode='constant')
    psf = np.fft.ifftshift(np.fft.ifft2(uv_arr))
    if unnormalize:
        psf *= psf.size
    return psf


def gen_psf2(uv, pad=False, multiplier=None):
    """Generate a point spread function map from a UV map.

    Parameters
    ----------
    uv: array-like or str
        UV weights map.
        If string, assume a FITS file of UV weights.
        If array-like, assume an array of UV weights map.
    pad: bool, optional
        Padding the uv by its size before FFT.
        Default is False
    multiplier: float, optional
        Multiplier to scale the PSF

    Note
    ----
        MAPS use the the forward transform in FFTW library (negative exponent)
        to FFT from an image plane to a UV plane. We use numpy.fft.ifft
        to do a reverse transform from the UV plane to an image plane.
        This keep the FFT sign consistent.
        We also re-normalize the numpy ifft to make it returns
        an un-normalized result.

    """
    if isinstance(uv, str):
        uv_arr = fits.getdata(uv)
    else:
        uv_arr = uv
    if multiplier:
        uv_arr *= multiplier
    if pad:
        xpad = uv_arr.shape[0] / 2
        ypad = uv_arr.shape[0] / 2
        uv_arr = np.pad(uv_arr, pad_width=((xpad, ypad), (xpad, ypad)),
                        mode='constant')
    return fftshift(fftn(ifftshift(uv_arr))).real


def fit_psf_radial(psf, r, init_params=((6000., 0., gaussian_fwhm2std(5.)),
                                        (1000., 0., gaussian_fwhm2std(60.)))):
    """
    Fit PSF radial profile with a compound Gaussian model.

    Parameters
    ----------
    psf: array-like
        Radial profile of the PSF.
    r: array-like
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
    g_comp = fitter(g_comp_init, np.hstack((-r[::-1], r)),
                    np.hstack((psf[::-1], psf)))
    n_g = np.size(g_comp)
    g_out = []
    for i in range(n_g):
        g_out.append(g_comp[i])
    return g_out


def fit_psf_cross(psf, center=(2048, 2048), direction='x', vmin=0.1,
                  x_scale=1, y_scale=1):
    if direction == 'x':
        data = psf[center[0], :]
        x = (np.arange(data.size) - center[0]) * x_scale
    else:
        data = psf[:, center[1]]
        x = (np.arange(data.size) - center[1]) * y_scale
    m = data >= vmin
    g_init = models.Gaussian1D(data[m].max(), 0, data[m].std())
    fitter = fitting.LevMarLSQFitter()
    g = fitter(g_init, x[m], data[m])
    g.stddev.value = np.abs(g.stddev.value)
    return g


def radial_profile_fast(data, center=(2048, 2048), r_scale=1,
                        norm=False, max_r='circle'):
    """
    Faster way to compute radial profile with some limitations.

    Parameters
    ----------
    data: 2D numpy array
        Data to calculate the profile
    center: (float, float)
        (x-center, y-center) for the profile
    r_scale: float. optional
        scaling factor for radius.
        Default is None, i.e. no scaling, result in pixels.
    norm: {'max', 'sum', None}, optional
        Normalization options.
        'max': normalize the profile by maximas.
        'sum': normalize the integral of the profile to 1.
        None: return the counts.
    max_r: {float, 'circle', 'all'}, optional
        Maximum radius for the profile.
        float: limit the radius to that number.
        'circle': limit the radius to a full concentric circle from `center`.
        'all': includes all radius, including those outside a full concentric
        circle from `center`.
        Everything else will use 'all'

    Returns
    -------
    radial_avg: numpy array
        Radial average profile.
    r: numpy array
        Radius corresponding to the profiles.

    See Notes below about lengths of returns.

    Notes
    -----
    This is a faster algorithm for radial profile, but due to the use of
    `numpy.bincount`, the length of an output array is limited
    to numpy.amax(rpix) + 1, where
        rpix = np.sqrt(((xpix - center[0])) ** 2 + ((ypix - center[1])) ** 2)
    i.e. 1 + the maximum of the radial coordinates of data in pixels,
    before applying `r_scale`

    """
    r = gen_radius_array(data.shape, center, xy_scale=None, r_scale=None)
    r = r.astype('int')
    radial_sum = np.bincount(r.ravel(), weights=data.ravel())
    nr = np.bincount(r.ravel())
    radial_avg = radial_sum / nr
    if norm == 'max':
        radial_sum /= radial_sum.max()
        radial_avg /= radial_avg.max()
    elif norm == 'sum':
        radial_sum /= radial_sum.sum()
        radial_avg /= radial_avg.sum()
    if isinstance(max_r, (float, int)):
        cut = slice(0, max_r)
    if max_r == 'circle':
        max_r_ = np.min(
            [np.array([center[0] - 0, data.shape[0] - center[0]]),
             np.array([center[1] - 0, data.shape[1] - center[1]])])
        cut = slice(0, max_r_)
    else:
        cut = slice(0, -1)
    return radial_avg[cut], \
           np.arange(np.amax(r) + 1)[cut] * r_scale


def radial_profile(data, center=(2048, 2048), bins='unique', xy_scale=None,
                   r_scale=None, max_r=None, data_range=None, norm=None):
    """
    Compute radial profile with some limitations.

    Parameters
    ----------
    data: 2D numpy array
        Data to calculate the profile
    center: (float, float)
        (x-center, y-center) for the profile
    bins : {int, sequence of scalars, 'max'}, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default).
        If `bins` is a sequence, it defines the bin edges, including the
        rightmost edge, allowing for non-uniform bin widths.
        If `bin` is 'unique', the maximum number of bins possible, i.e. the
        unique radius will be use.
    xy_scale: float or (float, float), optional
        Scale factor to apply to x and y dimension before generating a mask.
        If a single number is given, this is a scale factor for both x and y.
        If a (float, float) is given, these are (x scale, y scale).
    r_scale: float. optional
        scaling factor for radius.
        Default is None, i.e. no scaling, result in pixels.
    max_r: {None, float, 'circle'}, optional
        Maximum radius for the profile.
        None will use all radius [default]
        float: limit the radius to that number.
        'circle': limit the radius to a full concentric circle from `center`.
        'all': includes all radius, including those outside a full concentric
        circle from `center`.
    data_range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored.
    norm: {'max', 'sum', None}, optional
        Normalization options.
        'max': normalize the profile by maximas.
        'sum': normalize the integral of the profile to 1.
        None: return the counts.

    Returns
    -------
    radial_avg: numpy array
        Radial average profile.
    bin_edges: numpy array
        Return the bin edges of the profile.

    """
    r = gen_radius_array(data.shape, center, xy_scale=xy_scale, r_scale=r_scale)
    if max_r is None:
        max_r = r.max()
    elif max_r == 'circle':
        max_r = (0, np.min([r[0, center[1]], r[center[0], 0],
                            r[center[0], -1], r[-1, center[1]]]))
    mask = r <= max_r
    if bins == 'unique':
        bins = np.unique(r[mask])
    elif isinstance(bins, (int, float)):
        bins = np.linspace(0, max_r, bins)
    radial_counts, bin_edges = np.histogram(
        r[mask], bins=bins, range=data_range)
    radial_sum, bin_edges = np.histogram(
        r[mask], bins=bins, weights=data[mask], range=data_range)
    radial_avg = radial_sum / radial_counts
    if norm == 'max':
        radial_sum /= radial_sum.max()
        radial_avg /= radial_avg.max()
    elif norm == 'sum':
        radial_sum /= radial_sum.sum()
        radial_avg /= radial_avg.sum()
    return radial_avg, bin_edges[:-1]


def radial_crop_uv(uv_arr, center=(1024, 1024),
                   crop=((0, 50), (50, 370), (375, None)),
                   xy_scale=(0.477465, 0.477465)):
    """
    Crop UV map into multiple maps.

    """
    out = []
    for c in crop:
        mask = gen_radial_mask(uv_arr.shape, center, c, xy_scale=xy_scale,
                               mask=False)
        tmp = np.copy(uv_arr)
        tmp[mask] = 0
        out.append(tmp)
    return out


def hanning2d(m, n):
    """
    A 2D Hanning window.

    See numpy.hanning for the 1d description

    Parameters
    ----------
    m: int
        Size of x dimension.
    n: int
        Size of y dimension.
    Return
    ------
    out: numpy array of m by n
        The 2D Hanning window.

    """
    return np.outer(np.hanning(m), np.hanning(n))


def gen_hanning_mask(shape=(2048, 2048), center=(1024, 1024),
                     taper_edges=(80, 120)):
    """
    Generate a 2D Hanning Mask.

    Parameters
    ----------
    shape: tuple of 2 integer
        X and Y dimension of the mask
    center: tuple of 2 integer
        X and Y center of the mask
    taper_edges: tuple of 2 integer
        The pixels in which to start and stop the Hanning tapering

    """
    x = np.zeros(shape[0])
    y = np.zeros(shape[1])
    wsize = taper_edges[1] - taper_edges[0]
    w = np.hanning(wsize * 2 + 1)
    x[center[0] + taper_edges[0]:center[0] + taper_edges[1]] = w[-wsize:]
    y[center[1] + taper_edges[0]:center[1] + taper_edges[1]] = w[-wsize:]
    x[center[0] - taper_edges[1]:center[0] - taper_edges[0]] = w[:wsize]
    y[center[1] - taper_edges[1]:center[1] - taper_edges[0]] = w[:wsize]
    x[center[0] - taper_edges[0]:center[0] + taper_edges[0]] = 1
    y[center[1] - taper_edges[0]:center[1] + taper_edges[0]] = 1
    return np.outer(x, y)


def gaussian_std2fwhm(sigma):
    """
    Convert Gaussian STD to FWHM. Unit is the same as input.

    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def gaussian_fwhm2std(fwhm):
    """
    Convert Gaussian FWHM to STD. Unit is the same as input.

    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def uvdist2impix(uvdist, imres):
    """
    Convert UV disance to image pixels.

    Parameters
    ----------
    uvdist: float
        UV distance in wavelength
    imres: float
        Angular resolution of the image pixel in radian

    """
    psf_angle = 1. / uvdist
    return psf_angle / imres


def dist2psf_pix(d, psf_res, frequency):
    wavelength = const.c.value / (frequency * 1e6)
    uvdist = d / wavelength
    psf_angle = 1. / uvdist
    return psf_angle / psf_res


def impix2dist(npix, psf_res, frequency):
    psf_angle = npix * psf_res
    uvdist = 1. / psf_angle
    wavelength = const.c.value / (frequency * 1e6)
    return uvdist * wavelength


def impix2uvdist(npix, psf_res):
    psf_angle = npix * psf_res
    uvdist = 1. / psf_angle
    return uvdist