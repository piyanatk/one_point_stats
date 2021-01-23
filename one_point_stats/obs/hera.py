import numpy as np
from astropy import constants as const
from astropy.io import fits

from ..stats import get_stats, sky_noise_error, get_hera_eor_spec
from ..utils.settings import MWA_FREQ_EOR_ALL_80KHZ as F


def get_map_size(tel):
    map_size_dict = {
        'hera19': 64, 'hera37': 64, 'hera61': 128, 'hera91': 128,
        'hera127': 128, 'hera169': 128, 'hera217': 256, 'hera271': 256,
        'hera331': 256
    }
    return map_size_dict[tel]


def get_map_pix_area(tel):
    map_pix_dict = {
        'hera19': 0.22903242745449 ** 2, 'hera37': 0.22903242745449 ** 2,
        'hera61': 0.11451621372725 ** 2, 'hera91': 0.11451621372725 ** 2,
        'hera127': 0.11451621372725 ** 2, 'hera169': 0.11451621372725 ** 2,
        'hera217': 0.057258106863623 ** 2, 'hera271': 0.057258106863623 ** 2,
        'hera331': 0.057258106863623 ** 2
    }
    return map_pix_dict[tel]


def get_ant_per_side(tel):
    ant_dict = {'hera19': 3, 'hera37': 4, 'hera61': 5, 'hera91': 6,
                'hera127': 7, 'hera169': 8, 'hera217': 9, 'hera271': 10,
                'hera331': 11}
    return ant_dict[tel]


def get_sb_area(nu_obs, ant_per_side):
    sb_area = (1.22 * (const.si.c.value / (nu_obs * 1e6)) /
               (14 * (2 * ant_per_side - 1))) ** 2
    return sb_area


def get_pb_area(nu_obs):
    pb_area = (1.22 * (const.si.c.value / (nu_obs * 1e6)) / 14) ** 2
    return pb_area


def get_noise_error(nu_obs, ant_per_side, noise_bw, tint=100):
    noise_error = sky_noise_error(
        *get_hera_eor_spec(
            nu_obs=nu_obs, ant_per_side=ant_per_side,
            bw=noise_bw, tint=tint
        )
    )
    return noise_error


def get_nbeam_per_pix(tel, nu_obs):
    ant_per_side = get_ant_per_side(tel)
    maps_pix_area = get_map_pix_area(tel)
    sb_area = get_sb_area(nu_obs, ant_per_side)
    nbeam_per_pix = maps_pix_area / sb_area
    return nbeam_per_pix


def get_list_of_channels(bw, sp=1, return_index=False):
    """Give a list of frequency channels for binning/windowing.

    Parameters
    ----------
    bw : float
        Bandwidth to bin/window [MHz]
    sp : float
        Spacing to between neighboring bin/windows [MHz]
    return_index : bool, option
        Return indexes instead of frequencies

    """
    if return_index:
        ch = np.arange(F.size)[::-1]
    else:
        ch = F[::-1]
    ncb = int(np.ceil(bw / 0.08))
    ncsp = int(np.ceil(sp / 0.08))
    nb = int(np.floor((F.size - ncb) / ncsp + 1))
    chls = [ch[(i * ncsp):ncb+(i * ncsp)][::-1] for i in range(nb)[::-1]]
    return chls


def get_stats(data, noise_err, nbeam_per_pix):
    npix = len(data)
    nsample = nbeam_per_pix * npix
    stats = get_stats(data, noise_err=noise_err, noise_err_npix=nsample)
    return stats


def get_data(fitsfiles, maskfile, average=False, map_scale=1):
    nchannels = len(fitsfiles)
    hdr = fits.getheader(fitsfiles[0])
    map_size = hdr['NAXIS1']
    mask = np.load(maskfile)
    stack = np.empty((nchannels, map_size, map_size))
    for i in range(nchannels):
        stack[i] = fits.getdata(fitsfiles[i]) * map_scale
        stack[i] -= stack[i].mean()
    if average:
        out = stack.mean(axis=0)[mask]
    else:
        mask = (np.ones((nchannels, 1, 1)) * mask).astype(bool)
        out = stack[mask]
    return out
