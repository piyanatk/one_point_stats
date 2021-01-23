import numpy as np
import xarray as xr
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import astropy.constants as const
from astropy.cosmology import Planck15 as Cosmo
from scipy.signal.windows import blackmanharris, hanning, nuttall, gaussian
from astropy.convolution import convolve_fft

from .cosmo import get_cosmo, xyf2k
from ..utils import F21

__all__ = ['wedge', 'kpar', 'fork', 'apply_filter', 'get_rolling_filter_bins',
           'get_wedge_slope', 'rebuild_iter_1mhz']


def _init_filter(x, y, f):
    kx, ky, kz = xyf2k(x, y, f)
    out = np.ones((kz.size, ky.size, kx.size), dtype=float)
    return out, kx, ky, kz


def _smooth_filter_array(array, kernel_shape, kernel_type='nuttall',
                         gaussian_cutoff=3.0):
    kernel_func_dict = {'nuttall': nuttall, 'blackmanharris': blackmanharris,
                        'hanning': hanning, 'gaussian': gaussian}
    func = kernel_func_dict[kernel_type]
    if kernel_type == 'gaussian':
        std = np.array(kernel_shape) / gaussian_cutoff
        k = func(kernel_shape[0], std[0])[:, None, None] * \
            func(kernel_shape[1], std[1])[None, :, None] * \
            func(kernel_shape[2], std[2])[None, None, :]
    else:
        k = func(kernel_shape[0])[:, None, None] * \
            func(kernel_shape[1])[None, :, None] * \
            func(kernel_shape[2])[None, None, :]
    out = convolve_fft(array, k, boundary='wrap', allow_huge=True)
    return out


def apply_filter(data_array, filter_array,
                 fft_multiplier=None, ifft_multiplier=None,
                 output_multiplier=None, apply_window_func=False,
                 window_shift=None, invert_filter=False):
    """FFT image cube, apply mask and iFFT the output back to image domain.

    Parameters
    ----------
    data_array : ndarray
        Image cube.
    filter_array : ndarray
        Filter to multiply to FFT(data_array). Same shape as data_array.
        Assume (min, max) = (0, 1).
    fft_multiplier : float, optional
        Scalar to multiply to FFT output before masking.
    ifft_multiplier : float, optional
        Scalar to multiply to the masked array before iFFT back.
    output_multiplier : float, optional
        Scalar to multiply to output
    apply_window_func : bool, optional
        Apply blackman-nuttall window function to the first dimension of
        the data_array before FFT.
    window_shift : int
        Shift window function by these many pixels.
        Negative will shift to the left.
    invert_filter: bool, optional
        Invert the filter (1 - filter_array) before applying. Default is False.

    Returns
    -------
    out: ndarray
        Masked image cube.

    """
    if apply_window_func:
        k = nuttall(data_array.shape[0])
        if window_shift:
            k = np.roll(k, window_shift)
        w = (np.ones_like(data_array).T * k).T
    else:
        w = np.ones_like(data_array)
    data_fft = fftshift(fftn(ifftshift(data_array * w)))
    if fft_multiplier is not None:
        data_fft *= fft_multiplier
    if invert_filter:
        filter_array = 1 - filter_array
    data_fft *= filter_array
    out = fftshift(ifftn(ifftshift(data_fft))) / w
    if ifft_multiplier is not None:
        out *= ifft_multiplier
    if output_multiplier is not None:
        out *= output_multiplier
    return out.real


def fork(x, y, f, k0_width=0.1, slope_width=0.1,
         taper_edge=False, taper_func='nuttal',
         taper_width_kz=0.1, gaussian_taper_cutoff=3,
         slope_shift_kz=0.0, slope_multiplier=None):
    """Make a fork-shape foreground_filter filter.

    Parameters
    ----------
    x : ndarray
        Angular x coordinate. [rad]
    y : ndarray
        Angular y coordinate. [rad]
    f : ndarray
        Frequency coordinate. [Hz]
    k0_width: float, optional
        Width of k-parallel part filter. [h/Mpc]. Default is 0.1.
    slope_width: float, optional
        Width of wedge-slope part filter. [h/Mpc]. Default is 0.1.
    taper_edge : bool, optional
        Taper edges of the filter. Default is False.
    taper_func : {'nuttall', 'blackmanharris', 'hanning', 'gaussian'}, optional
        Window function to use for edges tapering.
    taper_width_kz : float, optional
        Width of the tapering regions from the edges in kz direction. [h/Mpc].
    gaussian_taper_cutoff : float, optional
        Standard deviation of a Gaussian teperring kernel.
    slope_shift_kz : float, optional
        Shift the wedge slope in kz direction. [h/Mpc].
    slope_multiplier : float, optional
        Multiply the slope of the wedge by this number.

    Returns
    -------
    out: ndarray
        Fork-shape filter with shape=(f.size, y.size, x.size).

    """
    z = (F21 - f) / f
    z0 = z.mean()
    a = get_wedge_slope(z0, multiplier=slope_multiplier)
    out, kx, ky, kz = _init_filter(x, y, f)
    k_perp = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    if taper_edge:
        # Determine the shape of the smoothing kernel
        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        dkz = kz[1] - kz[0]
        slope_shift_kz += taper_width_kz
        kx_shift = slope_shift_kz / a
        taper_kernel_shape = np.ceil([slope_shift_kz / dkz, kx_shift / dky,
                                      kx_shift / dkx]).astype(int) * 2 + 1
        idx1 = np.abs(np.ones_like(out) * kz[:, None, None]) < \
            (k0_width / 2 + taper_width_kz)

        kz_width = np.sqrt(slope_width ** 2 + (a * slope_width) ** 2)
        idx2 = np.logical_and(
            np.abs(kz[:, None, None]) < a * k_perp[None, :, :] +
            (kz_width / 2) + slope_shift_kz,
            np.abs(kz[:, None, None]) > a * k_perp[None, :, :] -
            (kz_width / 2) - slope_shift_kz
        )
        idx = np.logical_or(idx1, idx2)
        out[idx] = 0.0
        out = _smooth_filter_array(out, taper_kernel_shape,
                                   kernel_type=taper_func,
                                   gaussian_cutoff=gaussian_taper_cutoff)
    else:
        idx1 = np.abs(np.ones_like(out) * kz[:, None, None]) < (k0_width / 2)

        kz_width = np.sqrt(slope_width ** 2 + (a * slope_width) ** 2)
        idx2 = np.logical_and(
            np.abs(kz[:, None, None]) < a * k_perp[None, :, :] +
            (kz_width / 2) + slope_shift_kz,
            np.abs(kz[:, None, None]) > a * k_perp[None, :, :] -
            (kz_width / 2) + slope_shift_kz
        )
        idx = np.logical_or(idx1, idx2)
        out[idx] = 0.0
    return out


def get_rolling_filter_bins(filter_bandwidth, channel_width=0.08,
                            nchannels=705):
    """Calculate bin indexes for rolling filter application.

    Parameters
    ----------
    filter_bandwidth : float
        Bandwidth of the filter in MHz
    channel_width : float, optional
        Bandwidth of the frequency channel in MHz. Default 0.08 MHz.
    nchannels : int, optional
        Number of frequency channel in the data. Default 705.

    Returns
    -------
    bins: ndarray of (bin number, frequency channels in each bin).

    The bin #0 is defined as the bin with the highest frequency.

    """
    indexes = np.arange(nchannels)[::-1]
    nch_per_bin = int(np.floor(filter_bandwidth / channel_width))
    bins = np.array([indexes[i:i + nch_per_bin][::-1]
                     for i in range(nchannels - nch_per_bin)])
    return bins


def get_wedge_slope(z0, theta=np.pi/2, multiplier=None):
    """Get slope of the foreground_filter wedge (see Morales et al. (2012), Eq. 10).

    Parameters
    ----------
    z0 : float
        Reference redshift of the data cube
    theta : float, optional
        Maximum angular distance of point sources from the field center.
        This define the slope of the wedge. [rad]
        Default is pi/2 (i.e. horizon)
    multiplier :
        Additional multiplier to the slope value

    Returns
    -------
    out: slope of the wedge.

    """
    cosmo_e, cosmo_d = get_cosmo(z0)
    a = np.sin(theta) * (Cosmo.H0.value * 1e3 * cosmo_e * cosmo_d) / \
        (const.si.c.value * (1 + z0))
    if multiplier is not None:
        a *= multiplier
    return a


def kpar(x, y, f, width=0.1, taper_edge=False, taper_func='nuttall',
         taper_width_kz=0.1, gaussian_taper_cutoff=3):
    """Make a foreground_filter k-parallel filter.

    Parameters
    ----------
    x : ndarray
        Angular x coordinate. [rad]
    y : ndarray
        Angular y coordinate. [rad]
    f : ndarray
        Frequency coordinate. [Hz]
    width: float, optional
        Width of the filter. [h/Mpc]. Default is 0.1.
    taper_edge : bool, optional
        Taper edges of the filter. Default is False.
    taper_func : {'nuttall', 'blackmanharris', 'hanning', 'gaussian'}, optional
        Window function to use for edges tapering.
    taper_width_kz : float, optional
        Width of the tapering regions from the edges in kz direction. [h/Mpc].
    gaussian_taper_cutoff : float, optional
        Standard deviation of a Gaussian teperring kernel.

    Returns
    -------
    out: ndarray
        k-parallel filter with shape=(f.size, y.size, x.size).

    """
    out, kx, ky, kz = _init_filter(x, y, f)
    if taper_edge:
        # Determine the shape of the smoothing kernel
        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        dkz = kz[1] - kz[0]
        taper_kernel_shape = np.ceil(
            [taper_width_kz / dkz, taper_width_kz / dky, taper_width_kz / dkx]
        ).astype(int) * 2 + 1
        idx = np.abs(np.ones_like(out) * kz[:, None, None]) < \
            (width / 2 + taper_width_kz)
        out[idx] = 0.0
        out = _smooth_filter_array(out, taper_kernel_shape,
                                   kernel_type=taper_func,
                                   gaussian_cutoff=gaussian_taper_cutoff)
    else:
        idx = np.abs(np.ones_like(out) * kz[:, None, None]) < (width / 2)
        out[idx] = 0.0
    return out


def wedge(x, y, f, theta=np.pi/2, slope_multiplier=None, slope_shift_pix=None):
    """Make a foreground_filter wedge filter.

    Parameters
    ----------
    x : ndarray
        Angular x coordinate. [rad]
    y : ndarray
        Angular y coordinate. [rad]
    f : ndarray
        Frequency coordinate. [Hz]
    slope_shift_pix : int, optional
        Shift the wedge slope in kz direction by a number of pixels.
        If slope_shift_kz is not None, the amount will be combine.
    slope_multiplier : float, optional
        Multiply the slope of the wedge by this number.
    theta : float, optional
        Maximum angular distance of point sources from the field center.
        This define the slope of the wedge. [rad]
        Default is pi/2 (i.e. horizon)

    Returns
    -------
    out: ndarray
        Wedge filter with shape = (f.size, y.size, x.size).

    """
    # Calculate the slope of the wedge at the reference redshift z0
    z = (F21 - f) / f
    z0 = z.mean()
    a = get_wedge_slope(z0, theta=theta, multiplier=slope_multiplier)

    # Initiate filter array and its k-space coordinates
    out, kx, ky, kz = _init_filter(x, y, f)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    dkz = kz[1] - kz[0]

    # Shift pixel references to the corners outward of x-axis and y-axis,
    # and inward of z-axis. This ensure that the pixel that partially fall
    # under the wedge are masked.
    xshift = np.sign(kx) * (dkx / 2)
    yshift = np.sign(ky) * (dky / 2)
    zshift = np.sign(kz) * (dkz / 2)
    kx += xshift
    ky += yshift
    kz -= zshift

    # Calculate slope shift in kz
    if slope_shift_pix is not None:
        slope_shift_kz = slope_shift_pix * dkz
    else:
        slope_shift_kz = 0

    # Calculate kperp and figure out the voxels to mask
    kp = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    wedge_mask = np.abs(kz[:, None, None]) < a * kp[None, :, :] + slope_shift_kz
    out[wedge_mask] = 0.0

    # Mask the lowest kz mode
    if kz.size % 2 == 0:  # Even
        min_baseline_mask = slice(
            int(np.floor(kz.size / 2 - dkz + 1)),
            int(np.ceil(kz.size / 2 + dkz))
        )
    else:  # Odd
        min_baseline_mask = slice(
            int(np.floor(kz.size / 2 - dkz)),
            int(np.ceil(kz.size / 2 + dkz))
        )
    out[min_baseline_mask] = 0.0
    return out


def rebuild_iter_1mhz(w_in):
    """Iteratively build a wedge filter that matches an equivalent filter at 1 MHz.

    Parameters
    ----------
    w_in : DataArray
        Input filter DataArray

    Returns
    -------
    out : Data Array
        Output filter DataArray

    """
    x = w_in.attrs['x']
    y = w_in.attrs['y']
    f = w_in.attrs['f']
    df = w_in.attrs['channel_bandwidth']
    f1 = np.arange(f.mean() - df / 2 - 5 * df, f.mean() + 6 * df, df)

    s = w_in.attrs['wedge_filter_slope']

    # k-space coordinates
    kx1, ky1, kz1 = xyf2k(x, y, f1)
    kx = w_in.kx.values
    ky = w_in.ky.values
    kz = w_in.kz.values
    dkz1 = kz1[1] - kz1[0]
    dkz = kz[1] - kz[0]

    kz1_bound = kz1 - (np.sign(kz1) * (dkz1 / 2))
    kz1_bound_pos = np.abs(kz1_bound[kz1_bound <= 0])[::-1]
    kp1_bound = kz1_bound_pos / s

    kz_bound = kz - (np.sign(kz) * (dkz / 2))
    kz_bound3d = kz_bound[:, None, None] * np.ones_like(w_in)
    kp_bound3d = np.sqrt((ky ** 2)[None, :, None] + (kx ** 2)[None, None, :])

    out_arr = np.ones_like(w_in)

    for i in range(kz1_bound_pos.size - 1):
        c1 = (np.abs(kz_bound3d) >= kz1_bound_pos[i])
        c2 = (np.abs(kz_bound3d) < kz1_bound_pos[i + 1])
        c3 = (kp_bound3d >= kp1_bound[i])
        mask = np.where(c1 & c2 & c3)
        out_arr[mask] = 0.0

    out = xr.DataArray(out_arr, dims=['kz', 'ky', 'kx'], coords=w_in.coords,
                       attrs=w_in.attrs)
    out.attrs['shape_to_freq'] = 1000000.0

    return out
