"""Utilities for working with data and dates, beyond those for
digitizing.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/dataprep/data_utils.py
import logging

logger = logging.getLogger("c2far.dataprep.data_utils")


class DataUtils():
    """Utilities for working with timestamps, meta, and for normalization.

    """
    @staticmethod
    def get_inclusive_num_pts(t0_s, t1_s, sample_per_s):
        """Get the number of points from t0 to t1, INCLUSIVE of both
        endpoints, with the given sampling period.

        """
        elapsed_dur_s = t1_s - t0_s
        # We include the one at dur=0, so +1:
        npoints = (elapsed_dur_s // sample_per_s) + 1
        return npoints

    @classmethod
    def timestamp_to_num_pts(cls, meta, timestamp_s):
        """Given a duration - either the duration to truncate, or the duration
        to NOT include loss for, or even just a generic
        startpoint_s/endpoint_s.  Turn that duration into the number
        of elements FROM THE END.  The key is that the meta data tells
        us our time scale (start/end/sampling_period), and so we make
        use of that.

        Rather than raising an assertion error if our timestamp is
        beyond the end, we just return None.

        """
        last_s = int(meta[0])
        sample_per_s = int(meta[9])
        if timestamp_s > last_s:
            return None
        npoints = cls.get_inclusive_num_pts(timestamp_s, last_s,
                                            sample_per_s)
        return npoints

    @staticmethod
    def set_last_s(meta, new_last):
        """Adjust the last_s value in the meta.

        """
        # Meta is always composed of strings:
        meta[0] = str(new_last)

    @classmethod
    def get_nmin_nmax(cls, csize, nsize, wser_arr):
        """Helper function to compute the nmax and nmin from the wser_arr (all
        the values), given the csize/nsize.

        Works for different shapes, assuming sequences are along first
        dimension. E.g., if wser_arr is NSEQ+2 x NBATCH x 1, this
        returns cmax/cmin each of shape NBATCH x 1.

        Arguments:

        csize: Int, length of conditioning window.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        wser_arr: Tensor, with sequences along first dimension.

        Return Float, Float if non-trivial, else None, None if any are
        trivial sequences.

        """
        # Conditioning window does not include first two:
        cser_arr = wser_arr[2:2 + csize]
        # the normalization window:
        norm_arr = cser_arr[-nsize:]
        # N.B.: here 'values' turns this 'min' object into a TENSOR,
        # not into numpy (like in pandas).
        nmin = norm_arr.min(dim=0).values
        nmax = norm_arr.max(dim=0).values
        return nmin, nmax

    @classmethod
    def torch_norm(cls, csize, nsize, wser_arr):
        """Normalize the wser_arr assuming a normalization window of size
        nsize.  Designed for torch tensors: they get nmax/nmin on the
        first dimension (the sequence).

        Arguments:

        csize: Int, length of conditioning window.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        wser_arr: all the values to norm.

        Optional keyword-args:

        Return None if it's a trivial sequence.

        """
        nmin, nmax = cls.get_nmin_nmax(csize, nsize, wser_arr)
        # Broadcast these values across each sequence:
        try:
            out_values = (wser_arr - nmin) / (nmax - nmin)
        except ZeroDivisionError as excep:
            logger.error("nmax can't be equal to nmin")
            raise excep
        return out_values
