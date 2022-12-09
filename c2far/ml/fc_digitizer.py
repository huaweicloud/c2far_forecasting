"""A special callable object who has the job of digitizing originals.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/fc_digitizer.py
import logging
import torch
from c2far.dataprep.data_utils import DataUtils
from c2far.dataprep.digitize_utils import DigitizeUtils

logger = logging.getLogger("c2far.ml.fc_digitizer")


class FCDigitizer():
    """Special module to convert from original to binned/digitized
    versions of time series.

    """
    def __init__(self, coarse_cutoffs, coarse_low, coarse_high,
                 nfine_bins, nfine2_bins, *, extremas):
        """Get the cutoffs for use when digitizing examples.

        Arguments:

        coarse_cutoffs: Tensor[NBINS], list of coarse cutoffs.

        coarse_low: Float, the lowest implicit boundary for the coarse
        cutoffs.

        coarse_high: Float, the highest implicit boundary for the
        coarse cutoffs.

        nfine_bins: Int, how many fine-grained bins, if we are doing
        that, or pass None if not doing that.

        nfine2_bins: Int, how many fine2-grained bins, if doing that,
        or pass None if not doing that. fine2 is more fine than fine.

        Keyword-only arguments:

        extremas: Boolean, if True, we also **return the normed
        originals**, so we can model those when they are extrema.

        """
        self.extremas = extremas
        self.coarse_cutoffs = coarse_cutoffs
        self.coarse_cuts_full = None
        self.fine_cutoffs = None
        self.fine2_cutoffs = None
        if nfine_bins is not None:
            # coarse_cuts_full: coarse cutoffs with extreme low/high
            # added. We use these to find our fine pos over coarses:
            self.coarse_cuts_full = self.expand_cuts(self.coarse_cutoffs, coarse_low, coarse_high)
            # And these to find our position over the fines:
            self.fine_cutoffs = self.get_fine_cutoffs(
                nfine_bins, device=coarse_cutoffs.device)
            # And, if doing within-within-bin, get those too:
            if nfine2_bins is not None:
                # Now we need the expanded fines, and these have fixed low/high:
                self.fine_cuts_full = self.expand_cuts(self.fine_cutoffs, 0, 1)
                self.fine2_cutoffs = self.get_fine_cutoffs(
                    nfine2_bins, device=coarse_cutoffs.device)

    @staticmethod
    def expand_cuts(cutoffs, fixed_lower, fixed_upper):
        """Given the cutoffs, expand them to include the very lowest point and
        very highest point.

        Arguments:

        cutoffs: Tensor[NBINS], list of cutoffs.

        fixed_lower/fixed_upper: Float, given lower/upper values for
        the cutoffs.

        Returns:

        expanded_cuts: Tensor[NBINS+2], cutoffs with added points for
        lowest low bin and highest high bin.

        """
        lows, highs = DigitizeUtils.torch_prepare_cutoffs(
            cutoffs, fixed_lower=fixed_lower, fixed_upper=fixed_upper)
        expanded_cuts = torch.cat([lows[0].reshape(-1), cutoffs,
                                   highs[-1].reshape(-1)])
        return expanded_cuts

    @staticmethod
    def get_fine_cutoffs(nfine_bins, *, device="cpu"):
        """Convert the number of bins to the evenly-spaced cutoffs that we can
        use for (A) digitizing and (B) reconstructing the values.

        """
        # Since we count the one before AND after, we need N-1 cutoffs
        # to generate N bins.  This function works because it excludes
        # the last value (which would be 1/1):
        try:
            fine_cutoffs = torch.arange(1, nfine_bins, device=device) / nfine_bins
        except ZeroDivisionError as excep:
            logger.error("The value of fine bins can't be 0")
            raise excep
        return fine_cutoffs

    def __digitize_coarse(self, csize, nsize, origs):
        """Get the 'coarses' from the conditioning window (or windows, if
        dynamically reconditioning), and also return the normed
        originals, which can be used in the finebin work (and note:
        these normed origs really determine both the coarse value bins
        and the fine bins, so the only difference between dynamic
        renorm and non-dynamic renorm is their calculation).

        Arguments:

        csize: Int, length of conditioning window.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        origs: tensor[Float], size NVALUES - the original data.

        Returns:

        coarses: Tensor[Int]: NSEQ, the indices of the coarse bins.

        normed_origs: Tensor[Float] of shape NSEQ, the originals, now
        normalized by the normalization window.

        """
        # Get coarses as before, but retain/return normed origs:
        normed_origs = DataUtils.torch_norm(csize, nsize, origs)
        coarses = DigitizeUtils.torch_digitize_normed(
            normed_origs, self.coarse_cutoffs)
        return coarses, normed_origs

    @staticmethod
    def get_bin_proportions(expanded_cuts, bin_idxs, values):
        """Get the proportion of each bin that we span, via the bin idxs, the
        the expanded_cuts that those idxs correspond to, and the
        values that occur in those bins.

        Arguments:

        expanded_cuts: Tensor[NBINS+2], cutoffs with added points for
        lowest low bin and highest high bin.

        bin_idxs: Tensor[Int] of shape NSEQ, the indices of the bins.
        Also works for multi-dim, e.g. NSEQ x NBATCH x 1

        values: Tensor[Float] of shape NSEQ, e.g. originals,
        normalized by normalization window, also works for multi-dim,
        e.g. NSEQ x NBATCH x 1.

        Returns:

        propors: Tensor[Float] of shape NSEQ, proportion of each
        coarse bin spanned by the normalized originals, or in
        multi-dim, e.g. NSEQ x NBATCH x 1.

        """
        bins_lower = expanded_cuts[bin_idxs]
        bins_higher = expanded_cuts[bin_idxs + 1]
        # Now get proportion of each bin that values are at - for
        # coarse bins, these are essentially the "normalized fines":
        try:
            propors = ((values - bins_lower) / (bins_higher - bins_lower))
        except ZeroDivisionError as excep:
            logger.error("bins_higher can't be equal to bins_lower")
            raise excep
        return propors

    def __call__(self, origs, csize, nsize):
        """Get the coarses, AND the indexes of the fine-grained bins
        (i.e., fine-grained within the coarses), or return None for the
        fine-grained if not using.

        Arguments:

        origs: tensor[Float], size NVALUES - the original data.

        csize: Int, length of conditioning window.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        Returns:

        coarses, fine_bins, fine2_bins, normed: Tensor[NSEQ], as their
        respective bin indices, or None if not doing.

        """
        coarses, normed_origs = self.__digitize_coarse(
            csize, nsize, origs)
        if self.extremas:
            returned_normed = normed_origs
        else:
            returned_normed = None
        if self.fine_cutoffs is None:
            return coarses, None, None, returned_normed
        propors = self.get_bin_proportions(
            self.coarse_cuts_full, coarses, normed_origs)
        fine_bins = DigitizeUtils.torch_encode_values(
            propors, self.fine_cutoffs)
        if self.fine2_cutoffs is None:
            return coarses, fine_bins, None, returned_normed
        propors2 = self.get_bin_proportions(
            self.fine_cuts_full, fine_bins, propors)
        fine_bins2 = DigitizeUtils.torch_encode_values(
            propors2, self.fine2_cutoffs)
        return coarses, fine_bins, fine_bins2, returned_normed
