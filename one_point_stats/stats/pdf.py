import numpy as np

__all__ = ['cal_pdf']


def cal_pdf(data, bins=60, save_to=None):
    """Calculate probability density function.

    Parameters
    ----------
    data : array-like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (30, by default). If `bins` is a sequence,
        it defines the bin edges, including the rightmost edge, allowing
        for non-uniform bin widths.
    save_to : str
        File to save output to

    Returns
    -------
    pdf: array
        The PDF, normalized so that the integral is 1.
    bin_centers: array
        The center of each PDF bin.

    """
    pdf, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    if save_to is not None:
        np.savetxt(save_to, np.vstack((pdf, bin_centers)).T, delimiter=',',
                   header='pdf, bin_centers')
    return pdf, bin_centers
