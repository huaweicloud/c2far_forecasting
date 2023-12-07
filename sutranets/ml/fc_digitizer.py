"""A special callable object who has the job of digitizing originals.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/fc_digitizer.py
import torch
from sutranets.dataprep.data_utils import DataUtils
from sutranets.dataprep.digitize_utils import DigitizeUtils


class DigitizerError(Exception):
    """Base class for exceptions in this module, i.e., for doing
    digitization related to time series.

    """


class TrivialError(DigitizerError):
    """Error related to encountering a trivial series when attempting to
    determine the dynamic range of a normalization window.

    """


class FCDigitizer():
    """Special module to convert from original to binned/digitized
    versions of time series.

    """
    def __init__(self, coarse_cutoffs, coarse_low, coarse_high,
                 nfine_bins, nfine2_bins, *,
                 extremas, check_trivial=True):
        self.extremas = extremas
        self.check_trivial = check_trivial
        self.coarse_cutoffs = coarse_cutoffs
        self.coarse_cuts_full = None
        self.fine_cutoffs = None
        self.fine2_cutoffs = None
        if nfine_bins is not None:
            self.coarse_cuts_full = self.expand_cuts(self.coarse_cutoffs, coarse_low, coarse_high)
            self.fine_cutoffs = self.get_fine_cutoffs(
                nfine_bins, device=coarse_cutoffs.device)
            if nfine2_bins is not None:
                self.fine_cuts_full = self.expand_cuts(self.fine_cutoffs, 0, 1)
                self.fine2_cutoffs = self.get_fine_cutoffs(
                    nfine2_bins, device=coarse_cutoffs.device)

    @staticmethod
    def expand_cuts(cutoffs, fixed_lower, fixed_upper):
        """Given the cutoffs, expand them to include the very lowest point and
        very highest point.

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
        fine_cutoffs = torch.arange(1, nfine_bins, device=device) / nfine_bins
        return fine_cutoffs

    def __get_normed_origs(self, csize, nsize, trivial_min_nonzero_frac, meta, origs):
        """Get the normed_origs via min-max scaling over the conditioning
        window.

        """
        normed_origs, nmin, nmax = DataUtils.torch_norm(
            csize, nsize, origs, check_trivial=self.check_trivial,
            trivial_min_nonzero_frac=trivial_min_nonzero_frac)
        if normed_origs is None:
            msg = f"Unexpected trivial/zero, csize {csize}: {meta},{origs}"
            raise TrivialError(msg)
        return normed_origs, nmin, nmax

    @staticmethod
    def get_bin_proportions(expanded_cuts, bin_idxs, values):
        """Get the proportion of each bin that we span, via the bin idxs, the
        the expanded_cuts that those idxs correspond to, and the
        values that occur in those bins.

        """
        bins_lower = expanded_cuts[bin_idxs]
        bins_higher = expanded_cuts[bin_idxs + 1]
        propors = ((values - bins_lower) / (bins_higher - bins_lower))
        return propors

    def digitize_from_normed(self, normed_origs):
        """Helper and alternative entry point to get all the bin indices from
        the pre-normed originals.

        """
        coarses = DigitizeUtils.torch_digitize_normed(
            normed_origs, self.coarse_cutoffs)
        if self.fine_cutoffs is None:
            return coarses, None, None
        propors = self.get_bin_proportions(
            self.coarse_cuts_full, coarses, normed_origs)
        fine_bins = DigitizeUtils.torch_encode_values(
            propors, self.fine_cutoffs)
        if self.fine2_cutoffs is None:
            return coarses, fine_bins, None
        propors2 = self.get_bin_proportions(
            self.fine_cuts_full, fine_bins, propors)
        fine2_bins = DigitizeUtils.torch_encode_values(
            propors2, self.fine2_cutoffs)
        return coarses, fine_bins, fine2_bins

    def __call__(self, meta, origs, bin_covars, csize, nsize,
                 trivial_min_nonzero_frac, *, covars_only=False):
        """Get the coarses, and the indexes of the fine-grained bins (i.e.,
        fine-grained within the coarses), or return None for the
        fine-grained if not using.  For SutraNets, also digitizes the
        bin_covars.

        """
        normed_origs, nmin, nmax = self.__get_normed_origs(
            csize, nsize, trivial_min_nonzero_frac, meta, origs)
        if bin_covars is not None:
            bin_covars.run_digitize(nmin, nmax, self.digitize_from_normed)
        if covars_only:
            return None, None, None, None, bin_covars
        coarses, fine_bins, fine2_bins = self.digitize_from_normed(
            normed_origs)
        if self.extremas:
            returned_normed = normed_origs
        else:
            returned_normed = None
        return coarses, fine_bins, fine2_bins, returned_normed, bin_covars
