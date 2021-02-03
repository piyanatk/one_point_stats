"""Make rolling wedge filters"""
import argparse
import multiprocessing
import os
from datetime import datetime

import numpy as np
import healpy as hp
import xarray as xr

from one_point_stats.utils import MWA_FREQ_EOR_ALL_80KHZ as F
from one_point_stats.utils import F21
from one_point_stats.foreground_filter import xyf2uve, xyf2k, get_cosmo, \
    wedge, get_rolling_filter_bins, get_wedge_slope


def make_wedge(binnum):
    f = F[bins[binnum]]
    f0 = f.mean()
    z0 = (F21 - f0) / f0

    if args.verbose:
        print("Process {:d} is making a filter with fwb={:.2f} MHz, "
              "theta={:.1f} deg, shift={:d} pix, binnum={:03d}, f0={:.3f} MHz."
              .format(multiprocessing.current_process().pid,
                      args.filter_bandwidth, args.theta,
                      args.slope_shift_pix, binnum, f0 / 1e6))

    u, v, e = xyf2uve(x, x, f)
    kx, ky, kz = xyf2k(x, x, f)
    cosmo_d, cosmo_e = get_cosmo(z0)
    filter_array = wedge(x, x, f, theta=args.theta * np.pi / 180,
                         slope_shift_pix=args.slope_shift_pix)
    out = xr.DataArray(
        filter_array, dims=('kz', 'ky', 'kx'),
        coords={'kx': kx, 'ky': ky, 'kz': kz},
        attrs={'filter_bandwidth': args.filter_bandwidth * 1e6,
               'channel_bandwidth': 0.08 * 1e6,
               'bin_number': binnum, 'frequency_channels': bins[binnum],
               'wedge_filter_slope': get_wedge_slope(z0),
               'u': u, 'v': v, 'e': e, 'x': x, 'y': x, 'f': f,
               'f0': f0, 'z0': z0, 'cosmo_d': cosmo_d, 'cosmo_e': cosmo_e,
               'xunit': 'rad', 'yunit': 'rad', 'funit': 'Hz',
               'theta': args.theta, 'theta_unit': 'deg',
               'slope_shift_pix': args.slope_shift_pix},
    )
    outname = '{:s}/wedge_filter_fbw{:.2f}MHz_' \
              'theta{:.1f}deg_shift{:d}_bin{:03d}.nc' \
              .format(args.output_dir, args.filter_bandwidth, args.theta,
                      args.slope_shift_pix, binnum)
    out.to_netcdf(outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filter_bandwidth', type=float,
                        help='bandwidth of the filter in MHz')
    parser.add_argument('--theta', type=float, default=90.0,
                        help='1/2 FoV for wedge boundary in deg')
    parser.add_argument('--slope_shift_pix', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--nprocs', type=int, default=8)
    parser.add_argument('--verbose', action='store_true',
                        help='print diagnostic output')
    args = parser.parse_args()

    start_time = datetime.now()

    # Check output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Data cube dimension is fixed for HERA 350 Core simulation
    nx = 256
    dx = hp.nside2resol(1024)  # rad
    # 0.5 is add so that we refer to a coordinate at the center of each pixel
    x = (np.arange(-128, 128) + 0.5) * dx  # rad
    nf = 705
    df = 80000  # Hz
    F *= 1e6  # Hz

    # Get bins for rolling filter
    bins = get_rolling_filter_bins(args.filter_bandwidth)
    nbins = len(bins)

    pool = multiprocessing.Pool(args.nprocs)
    pool.map(make_wedge, np.arange(nbins))
    pool.close()
    pool.join()

    print('Finish making a filter with fwb={:.2f} MHz, theta={:.1f} deg, '
          'shift={:d} pix, time spent={:.5f} seconds'
          .format(args.filter_bandwidth, args.theta, args.slope_shift_pix,
                  (datetime.now() - start_time).total_seconds()))