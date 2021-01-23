import numpy as np
import healpy as hp
from astropy.wcs import WCS
from astropy.cosmology import Planck15 as Cosmo

__all__ = ['healpix2sine', 'cube2healpix']


def healpix2sine(hpx_array, ra, dec, size, res, hpx_coord='C',
            return_fits_header=False):
    """
    Generate a SIN (orthographic) projected FITS image from a HEALPix image.

    Parameters
    ----------
    hpx_array: numpy.ndarray
        A HEALPix array (single-column format).
    ra: float, range=[0,360]
        Right ascension at the center of the FITS images in degree.
    dec: float, range=[90,-90]
        Declination at the center of the FITS images in degree.
    size: integer
        Size of the output FITS image in number of pixels.
        Only support a square image.
    res: float
        Angular resolution of the FITS image in degree,
        i.e. 'CDELTi' FITS keyword.
    hpx_coord : {'C', 'E' or 'G'}, optional
        Coordinate of the HEALPix map. 'C' for Celestial (default),
        'E' for Ecliptic, and 'G' for Galactic.
    return_fits_header: bool
        Return FITS header for the projected image and flip x and y dimension
        of the output array to match the FITS standard.
        Default is False.

    """
    # Check if healpix array is valid.
    if not hp.isnpixok(len(hpx_array)):
        raise IOError(
            'Number of pixels in a healpix array must be 12 * nside ** 2.'
        )

    # Create a new WCS object and set up a SIN projection.
    w = WCS(naxis=2)
    w.wcs.crpix = [float(size / 2), float(size / 2)]
    w.wcs.cdelt = [-res, res]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.equinox = 2000.0

    # Make pixel array of the projected map.
    x, y = np.mgrid[0:size, 0:size]

    # Convert pixel coordinates to celestial coordinates
    pixra, pixdec = w.wcs_pix2world(x.ravel(), y.ravel(), 0)
    pixra *= np.pi / 180.
    pixdec = np.pi * (90 - pixdec) / 180.  # Healpix dec is 0 to pi.

    # Make a mask out-of-sky values
    valid_pix = np.logical_not(np.isnan(pixra))

    # Remap celestial coordinate to the HELAPix coordinates if needed.
    if hpx_coord != 'C':
        rot = hp.Rotator(coord=['C', hpx_coord])
        pixdec[valid_pix], pixra[valid_pix] = \
            rot(pixdec[valid_pix], pixra[valid_pix])

    # Get the pixel values from a HEALPix map, using Bi-linear interpolation of
    # the 4 nearest neighbors.
    proj_map = np.zeros(size * size)
    proj_map[valid_pix] = hp.get_interp_val(
        hpx_array, pixdec[valid_pix], pixra[valid_pix]
    )

    if return_fits_header:
        header = w.to_header()
        return proj_map.reshape((size, size)).T, header
    else:
        return proj_map.reshape((size, size))


def cube2healpix(cube, cube_res, cube_freq, hpx_nside):
    """
    Tile and grid a 21 cm simulation cube to a full-sky HEALPix map.

    The projection assumes Planck15 cosmology.

    Parameters
    ----------
    cube: numpy.ndarray
        Simulation cube, 3-dimensional
    cube_res: float
        Resolution of the cube in Mpc.
    cube_freq: float
        Observed frequency matching the cube's cosmology in MHz.
    hpx_nside: integer
        NSIDE of the output HEALPix image. Must be a valid NSIDE for HEALPix,
        i.e. 2**i for integer i.

    """
    cube_shape = cube.shape

    # Determine the radial comoving distance dc to the "Universe" shell at the
    # observed frequency.
    f21 = 1420.40575177  # MHz
    z21 = f21 / cube_freq - 1
    dc = Cosmo.comoving_distance(z21).value

    # Get the vector coordinates (vx, vy, vz) of the HEALPix pixels.
    vx, vy, vz = hp.pix2vec(hpx_nside, np.arange(hp.nside2npix(hpx_nside)))

    # Translate vector coordinates to comoving coordinates and determine the
    # corresponding cube indexes (xi, yi, zi). For faster operation, we will
    # use the mod function to determine the nearest neighboring pixels and
    # just grab the data points from those pixels instead of doing linear
    # interpolation. This sets the origin of the projecting shell to pixel
    # (x, y, z) = (0, 0, 0).
    xi = np.mod(np.around(vx * dc / cube_res).astype(int), cube_shape[0])
    yi = np.mod(np.around(vy * dc / cube_res).astype(int), cube_shape[1])
    zi = np.mod(np.around(vz * dc / cube_res).astype(int), cube_shape[2])
    return cube[xi, yi, zi]
