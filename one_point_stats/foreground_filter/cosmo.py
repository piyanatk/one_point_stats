import numpy as np
from numpy.fft import fftshift, fftfreq, ifftshift
from astropy.cosmology import Planck15 as Cosmo
import astropy.constants as const

from ..utils import F21

__all__ = ['get_cosmo', 'xyf2k', 'xyf2uve', 'k2uve', 'k2xyf', 'uve2k']


def get_cosmo(z):
    """Return transverse comoving distance and 3-parameter density term 
     (Omega_m, Omega_k, Lambda) for the given redshifts.

    Parameters
    ----------
    z : float
        Redshift

    Returns
    -------
    out : tuple
        (transverse comoving distance, density term)
    
    """
    cosmo_d = Cosmo.comoving_transverse_distance(z).value
    cosmo_e = np.sqrt(Cosmo.Om0 * (1 + z) ** 3 + Cosmo.Ok0 *
                      (1 + z) ** 2 + Cosmo.Ode0)
    return cosmo_d, cosmo_e


def xyf2uve(x, y, f):
    """Convert (x, y, f) sky coordinate to (u, v, eta) FFT coordinates.
    
    Parameters
    ----------
    x : ndarray
        X coordinates [rad]
    y : ndarray
        Y coordinates [rad]
    f : ndarray
        Frequency coordinate [Hz]

    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    df = f[1] - f[0]
    u = fftshift(fftfreq(x.size, dx))
    v = fftshift(fftfreq(y.size, dy))
    e = fftshift(fftfreq(f.size, df))
    return u, v, e


def uve2k(u, v, e, z0, cosmo_d, cosmo_e):
    kx = u * 2 * np.pi / cosmo_d
    ky = v * 2 * np.pi / cosmo_d
    kz = e * 2 * np.pi * Cosmo.H0.value * 1e3 * F21 * \
        cosmo_e / (const.si.c.value * (1 + z0) ** 2)
    return kx, ky, kz


def k2uve(kx, ky, kz, z0):
    """Convert comoving k-space coordinates to (u, v, eta) coordinates.

    Parameters
    ----------
    kx : ndarray
        Array of kx [h/Mpc].
    ky : ndarray
        Array of ky [h/Mpc].
    kz : ndarray
        Array of kz [h/Mpc].
    z0 : float
        Reference redshift of the data cube.

    Returns
    -------
    out: tuple of ndarray (u, v, eta)

    """
    cosmo_d, cosmo_e = get_cosmo(z0)
    u = kx / (2 * np.pi / cosmo_d)
    v = ky / (2 * np.pi / cosmo_d)
    e = kz / (2 * np.pi * Cosmo.H0.value * 1e3 * F21 * cosmo_e /
              (const.si.c.value * (1 + z0) ** 2))
    return u, v, e


def xyf2k(x, y, f, z0=None):
    """Convert (x, y, f) coordinate to (k_perp, k_par).

    Use formalism from Morales and Hewitt (2004),
    i.e. small field approximation.

    Parameters
    ----------
    x : ndarray
        X coordinates [rad]
    y : ndarray
        Y coordinates [rad]
    f : ndarray
        Frequency coordinate [Hz]
    z0 : float, optional
        Redshift representation of the cube. If None, the parameter will be
        calculated from the mean of f assuming that the emission is 21 cm.

    """
    if z0 is None:
        z0 = ((F21 - f) / f).mean()
    cosmo_d, cosmo_e = get_cosmo(z0)
    u, v, e = xyf2uve(x, y, f)
    kx, ky, kz = uve2k(u, v, e, z0, cosmo_d, cosmo_e)
    return kx, ky, kz


def k2xyf(kx, ky, kz, z0):
    """Convert comoving k-space coordinates to spatial-frequency coordinates.
        
    Parameters
    ----------
    kx : ndarray
        Array of kx [h/Mpc].
    ky : ndarray
        Array of ky [h/Mpc].
    kz : ndarray
        Array of kz [h/Mpc].
    z0 : float
        Reference redshift of the data cube.

    Returns
    -------
    out: tuple of ndarray (x, y, f)
    
    """
    u, v, e = k2uve(kx, ky, kz, z0)
    du = np.abs(u[1] - u[0])
    dv = np.abs(v[1] - v[0])
    de = np.abs(e[1] - e[0])
    f0 = F21 / (1 + z0)
    x = ifftshift(fftfreq(u.size, du))
    y = ifftshift(fftfreq(v.size, dv))
    f = ifftshift(fftfreq(e.size, de)) + f0
    return x, y, f
