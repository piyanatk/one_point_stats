"""
Utilities functions.
"""
import numpy as np

__all__ = ['get_mwa_eor_spec', 'get_hera_eor_spec']


def get_mwa_eor_spec(nu_obs=150.0, nu_emit=1420.40575, bw=8.0, tint=1000.0,
                     area_eff=21.5, n_stations=50, bmax=100.0):
    """
    Parameters
    ----------
    nu_obs : float or array-like, optional
        observed frequency [MHz]
    nu_emit : float or array-like, optional
        rest frequency [MHz]
    bw : float or array-like, optional
        frequency bandwidth [MHz]
    tint : float or array-like, optional
        integration time [hour]
    area_eff : float or array-like, optional
        effective area per station [m ** 2]
    n_stations : int or array-like, optional
        number of stations
    bmax : float or array-like, optional
        maximum baseline [wavelength]
    Returns
    -------
    nu_obs, nu_emit, bw, tint, area_eff, n_stations, bmax
    """
    return nu_obs, nu_emit, bw, tint, area_eff, n_stations, bmax


def get_hera_eor_spec(nu_obs=150.0, nu_emit=1420.40575, bw=8.0,
                      tint=1000.0, area_eff=154, ant_per_side=8):
    n_stations = 3 * ant_per_side * (ant_per_side - 1) + 1
    bmax = 14 * (2 * ant_per_side - 1) / (2.99792458e2 / nu_obs)  # wavelength
    return nu_obs, nu_emit, bw, tint, area_eff, n_stations, bmax
