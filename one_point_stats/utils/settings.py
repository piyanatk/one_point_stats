import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord

__all__ = ['MWA_LOC', 'MWA_FIELD_EOR0', 'MWA_FIELD_EOR1', 'MWA_FIELD_EOR2',
           'MWA_FREQ_EOR_ALL_40KHZ', 'MWA_FREQ_EOR_ALL_80KHZ',
           'MWA_FREQ_EOR_HI_40KHZ', 'MWA_FREQ_EOR_HI_80KHZ',
           'MWA_FREQ_EOR_LOW_40KHZ', 'MWA_FREQ_EOR_LOW_80KHZ',
           'HERA_ANT_DICT', 'F21']


F21 = 1420.405751786e6


MWA_LOC = EarthLocation(lat='−26d42m11.94986s', lon='116d40m14.93485s',
                        height=377.827)
MWA_FIELD_EOR0 = SkyCoord(ra='0.0h', dec='-30.0d')
MWA_FIELD_EOR1 = SkyCoord(ra='4.0h', dec='-30.0d')
MWA_FIELD_EOR2 = SkyCoord(ra='10.33h', dec='-10.0d')
MWA_FREQ_EOR_LOW_40KHZ = np.arange(138.895, 167.055, 0.04)
MWA_FREQ_EOR_HI_40KHZ = np.arange(167.055, 195.255, 0.04)
MWA_FREQ_EOR_ALL_40KHZ = np.arange(138.895, 195.255, 0.04)
MWA_FREQ_EOR_LOW_80KHZ = np.arange(138.915, 167.075, 0.08)
MWA_FREQ_EOR_HI_80KHZ = np.arange(167.075, 195.275, 0.08)
MWA_FREQ_EOR_ALL_80KHZ = np.arange(138.915, 195.275, 0.08)
HERA_ANT_DICT = {'hera19': 3, 'hera37': 4, 'hera61': 5, 'hera91': 6,
                 'hera127': 7, 'hera169': 8, 'hera217': 9, 'hera271': 10,
                 'hera331': 11}