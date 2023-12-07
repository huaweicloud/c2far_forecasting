"""Utilities in terms of generic working with our data.  Ideally, this
would be the only function that knows how to parse our meta data, etc.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/data_utils.py
from math import floor, ceil
import torch

# At least this many percent of values must be non-zero:
DEFAULT_TRIVIAL_MIN_NONZERO_FRAC = 0.10


class DataUtils():
    """Static methods of this class let us work with our datasets.

    """
    @staticmethod
    def get_ninclude_pts(period_s, high_period_s, cutoff_s, last_s, span_width_s):
        """Get the number of points from the cutoff to the last_s, INCLUSIVE
        of points that touch the cutoff.  We base the "effective
        cutoff" on cutoff MINUS high_period_s, as described in #148.

        """
        effective_cutoff = cutoff_s - high_period_s
        # How many full periods of this frequency can you fit in?
        npts = floor((last_s - effective_cutoff - span_width_s)/period_s) + 1
        npts = max(npts, 0)
        return npts

    @staticmethod
    def get_nexclude_pts(period_s, high_period_s, cutoff_s, last_s):
        """Get the number of points from the cutoff to the last_s, EXCLUSIVE
        of points that touch the cutoff.  We base the "effective
        cutoff" on cutoff MINUS high_period_s, as described in #148.

        """
        effective_cutoff = cutoff_s - high_period_s
        # How many PARTIAL periods of this frequency can you fit in?
        # Include all partial periods in number of things to exclude.
        npts = ceil((last_s - effective_cutoff)/period_s)
        npts = max(npts, 0)
        return npts

    @classmethod
    def timestamp_to_ninclude_pts(cls, meta, cutoff_s, *, high_period_s=None, span_width_s=None):
        """Given a cutoff in seconds, return the number of points from the end
        of our time series such that their spans (from the beginning
        to the end of each period) fully occur after the cutoff. This
        is the function we will use to find the points to *include* in
        the loss, i.e., from loss_start_pt_s.  The meta data tells us
        our time scale (start/end/sampling_period), and so we make use
        of that.

        Arguments:

        meta: list[String], our normal meta-data, from one line in a
        trace file.

        cutoff_s: Int, the cutoff that we are using to determine which
        points are included.

        high_period_s: Int, the higher-frequency period with which the
        cutoff was originally intended to be used.  If None, use
        period from the time series.

        span_width_s: Int, the span of time that each sample of the
        time series covers.  If None, use period from the time series.

        """
        # Okay, the timestamp we have is the last timestamp:
        # hist_ser.index[-1] in tsg/io_utils.py.
        last_s = int(meta[0])
        period_s = int(meta[9])
        return cls.get_ninclude_pts(period_s, high_period_s, cutoff_s, last_s, span_width_s)

    @classmethod
    def timestamp_to_nexclude_pts(cls, meta, cutoff_s, *, high_period_s=None):
        """Given a cutoff in seconds, return the number of points from the end
        of our time series such that their spans (from the beginning
        to the end of each period) *even partially* occur after the
        cutoff. Counterpart to timestamp_to_ninclude_pts above.

        Arguments:

        meta: list[String], our normal meta-data, from one line in a
        trace file.

        cutoff_s: Int, the cutoff that we are using to determine which
        points are included.

        high_period_s: Int, the higher-frequency period with which the
        cutoff was originally intended to be used.  If None, use
        period from the time series.

        """
        last_s = int(meta[0])
        period_s = int(meta[9])
        return cls.get_nexclude_pts(period_s, high_period_s, cutoff_s, last_s)

    @staticmethod
    def set_last_s(meta, new_last):
        """Adjust the last_s value in the meta: it could be an integer coming
        in.

        """
        # Actually, let's make sure it's a string, since the meta
        # always seems to be composed of strings:
        meta[0] = str(new_last)

    @staticmethod
    def get_last_s(meta):
        """Get the last_s value in the meta.  It's currently a string, so we
        make it an integer going out.

        """
        return int(meta[0])

    @staticmethod
    def get_sampling_period_s(meta):
        """Get the sampling_period_s value in the meta."""
        return int(meta[9])

    @staticmethod
    def set_sampling_period_s(meta, new_val):
        """Set the sampling_period_s value in the meta.  Should be an integer,
        but we convert to string for the meta.

        """
        meta[9] = str(new_val)

    @staticmethod
    def __arr_is_trivial(val_arr, trivial_min_nonzero_frac):
        """Check whether this is a trival series.  Now we have two criteria:

        1. It's constant

        2. Less than 10% of the series is non-zero.

        Should work on torch tensors or numpy arrays, but only ones
        that are one-dimensional.  I.e., it doesn't vectorize the
        triviality detection across multiple series.

        """
        if val_arr.min() == val_arr.max():
            return True
        num_nonzero = (val_arr > 0).sum()
        if num_nonzero < len(val_arr) * trivial_min_nonzero_frac:
            return True
        return False

    @staticmethod
    def torch_batch_is_trivial(val_arr, trivial_min_nonzero_frac):
        """Given a batch of size NSEQ x NBATCH (or just NSEQ if not batched),
        determine which series are trivial.  Note the NSEQ should
        include the first two consumed values, and should only go for
        CSIZE.  This just implements the arr_is_trivial logic, but in
        a vectorized manner.  I.e., check whether for each series:

        1. It's constant

        2. Less than 10% of the series is non-zero.

        Here we skip the first two values, because we never include
        them in the trivial calculation, as noted above.

        """
        val_arr = val_arr[2:]
        # check 1, constant:
        trivials1 = val_arr.min(dim=0).values == val_arr.max(dim=0).values
        # check 2, trivial fraction:
        num_nonzero = (val_arr > 0).sum(dim=0)
        trivials2 = num_nonzero < len(val_arr) * trivial_min_nonzero_frac
        return torch.logical_or(trivials1, trivials2)

    @classmethod
    def get_nmin_nmax(cls, csize, nsize, wser_arr, *, check_trivial=True, trivial_min_nonzero_frac=None):
        """Helper function to compute the cmax and cmin from the wser_arr (all
        the values), given the csize.

        Works for different shapes, assuming sequences are along first
        dimension. E.g., if wser_arr is NSEQ+2 x NBATCH x 1, this
        returns cmax/cmin each of shape NBATCH x 1.

        Arguments:

        csize: Int, length of conditioning window.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        wser_arr: Tensor, with sequences along first dimension, NSEQ x
        NSERIES.

        Optional keyword-args:

        check_trivial: Boolean, if True, also check the input for
        trivialness.  (Not for tensor/batched input)

        trivial_min_nonzero_frac: Float, the proportion of an nwindow
        that must be non-zero in order to not be trivial, or None if
        not checking trivials.

        Return Float, Float if non-trivial, Tensor[Float],
        Tensor[Float] of length NSERIES if vectorized, else None, None
        if any are trivial sequences.

        """
        # cser_arr: the conditioning window (well, technically
        # wser_arr[1] is also in the conditioning window), but we use
        # csize values to avoid confusion - see `notes/on_windows`.
        cser_arr = wser_arr[2:2 + csize]
        # the normalization window:
        norm_arr = cser_arr[-nsize:]
        # Only check triviality if going series-by-series:
        if check_trivial and cls.__arr_is_trivial(norm_arr, trivial_min_nonzero_frac):
            return None, None
        # N.B.: here 'values' turns this 'min' object into a TENSOR,
        # not into numpy (like in pandas)!
        nmin = norm_arr.min(dim=0).values
        nmax = norm_arr.max(dim=0).values
        # Assume (nmin == nmax).any() is False at this stage.
        return nmin, nmax

    @classmethod
    def torch_norm(cls, csize, nsize, wser_arr, *, check_trivial, trivial_min_nonzero_frac):
        """Normalize the wser_arr assuming a normalization window of size
        csize.  Designed for torch tensors: they get nmax/nmin on the
        first dimension (the sequence).

        Arguments:

        csize: Int, length of conditioning window.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        wser_arr: all the values to norm.

        Optional keyword-args:

        check_trivial: Boolean, if True, also check the input for
        trivialness.  For now, don't do if we have batched
        multi-dimensional input.

        trivial_min_nonzero_frac: Float, the proportion of an nwindow
        that must be non-zero in order to not be trivial.

        Return None if it's a trivial sequence.

        """
        nmin, nmax = cls.get_nmin_nmax(csize, nsize, wser_arr,
                                       check_trivial=check_trivial,
                                       trivial_min_nonzero_frac=trivial_min_nonzero_frac)
        if nmin is None:
            return None, None, None
        # Now broadcast these values across each sequence.
        out_values = (wser_arr - nmin) / (nmax - nmin)
        return out_values, nmin, nmax
