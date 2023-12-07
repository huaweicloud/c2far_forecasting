"""Functions just to do the linear clustering (evenly-spaced bins).

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/dataprep/cutoffs/linear_cutoffs.py

import logging
logger = logging.getLogger("sutranets.dataprep.cutoffs.linear_cutoffs")


def get_coarse_cutoffs(nbins, coarse_low, coarse_high):
    """Create the cutoff map for coarses.

    Arguments:

    nbins: Int, the number of bins for which we want cutoffs.

    coarse_low: Float, the lower extreme cutoff for binning.

    coarse_high: Float, the high extreme cutoff for binning.

    Returns:

    cutoffs: List[Int], of length (nbins - 1) - the cutoffs dividing
    the region from coarse_low to coarse_high into nbins evenly-sized
    bins.

    """
    logger.info("Creating simple linear coarse cutoffs")
    if nbins is None or coarse_low is None or coarse_high is None:
        raise RuntimeError("Require nbins and coarse low/high to get linear cutoffs.")
    span = coarse_high - coarse_low
    delta = span / nbins
    cutoffs = []
    for i in range(1, nbins):
        cutoffs.append(coarse_low + i * delta)
    return cutoffs
