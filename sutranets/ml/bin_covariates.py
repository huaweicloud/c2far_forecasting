"""A class to serve the bin-based covariates that we want to include
when we encode inputs in the BinTensorMaker.  These are for covariates
that may exist at coarse/fine/fine2 levels, such as bins from other
series (in a MultivariateDataset).  Used over multiple phases, and
usage depends on training vs. generation.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/bin_covariates.py
import torch
from sutranets.ml.constants import DUMMY_BIN
from sutranets.ml.bin_tensor_maker import BinTensorMaker


class BinCovariates():
    """A container for coarse/fine/fine2 covariates, to be used in the
    BinTensorMaker.  Basically, every internal list is a List[Tensor
    of size NSEQ] or List[Tensors, size NSEQ x NBATCH], where NSEQ is
    the length of the sequence.  Each of the elements of the tensor is
    an INDEX of a bin.

    """
    def __init__(self, *, raw_lst, include_prevs):
        """Initialize our storage stuff, also setting the raw_lst if it's
        provided.

        """
        self.raw_lst = raw_lst
        self.filtered_raw_lst = None
        self.include_prevs = include_prevs
        self.coarse_lst = None
        self.fine_lst = None
        self.fine2_lst = None
        self.target_level = None

    def set_target_level(self, target_level):
        """Setter for the target_level field."""
        self.target_level = target_level

    def run_digitize(self, nmin, nmax, digitize_fr_normed):
        """Digitize your raw value list into coarse/fine/fine2 lists, first
        norming using nmin/nmax, then using the given
        digitize_fr_normed function.

        """
        if self.target_level is None:
            raise RuntimeError("Must set target level before digitizing.")
        if not self.raw_lst:
            return
        if self.filtered_raw_lst is not None:
            raw_lst = self.filtered_raw_lst
        else:
            raw_lst = self.raw_lst
        self.coarse_lst = []
        self.fine_lst = []
        self.fine2_lst = []
        for series_idx, raws in enumerate(raw_lst):
            if series_idx == self.target_level:
                continue
            if ((not self.include_prevs) and (series_idx > self.target_level)):
                continue
            if raws is None:
                coarses, fines, fines2 = None, None, None
            else:
                normed = (raws - nmin) / (nmax - nmin)
                coarses, fines, fines2 = digitize_fr_normed(normed)
            self.coarse_lst.append(coarses)
            self.fine_lst.append(fines)
            self.fine2_lst.append(fines2)

    @staticmethod
    def __execute_shift(dummy_tens, series_idx, target_level, bins):
        """Helper to shift and return the given bins, according to our rules
        as described below in shift().

        """
        if bins is None:
            return None
        if dummy_tens is None:
            if series_idx < target_level:
                bins = bins[2:]
            else:
                bins = bins[1:-1]
        else:
            if series_idx < target_level:
                bins = bins[2:]
                bins = torch.cat([bins, dummy_tens])
            else:
                bins = bins[1:]
        return bins

    def shift(self, genmode):
        """Shift covariate idxs depending on genmode.

        When genmode==False: get values from `2:` when prior to target
        level (i.e., these are *current*), else values from `1:-1`
        (i.e., these are *previous*) [if any].

        When `genmode==True`, we'd instead use `2:` and subsequent
        values from `1:` (right until the end for subsequent) and have
        dummy values at the end for only the `2:` portion.

        """
        if self.target_level is None:
            raise RuntimeError("Must set target level before shifting.")
        shifted_coarse_lst = []
        shifted_fine_lst = []
        shifted_fine2_lst = []
        for series_idx, (coarses, fines, fines2) in enumerate(
                zip(self.coarse_lst, self.fine_lst, self.fine2_lst)):
            if coarses is not None:
                dummy_tens = None
                if genmode:
                    dummy_tens = torch.tensor([DUMMY_BIN], device=coarses.device)
                    nbatch = BinTensorMaker.get_nbatch(coarses)
                    if nbatch > 0:
                        dummy_tens = dummy_tens.expand(1, nbatch)
                coarses = self.__execute_shift(dummy_tens, series_idx, self.target_level, coarses)
                fines = self.__execute_shift(dummy_tens, series_idx, self.target_level, fines)
                fines2 = self.__execute_shift(dummy_tens, series_idx, self.target_level, fines2)
            shifted_coarse_lst.append(coarses)
            shifted_fine_lst.append(fines)
            shifted_fine2_lst.append(fines2)
        self.coarse_lst = shifted_coarse_lst
        self.fine_lst = shifted_fine_lst
        self.fine2_lst = shifted_fine2_lst

    def get_lsts(self):
        """Return the processed covariate lists"""
        return self.coarse_lst, self.fine_lst, self.fine2_lst

    def get_at_idx(self, idx):
        """Return the covariates at the given idx in the sequence dimension."""
        slice_coarses = []
        slice_fines = []
        slice_fines2 = []
        for coarses, fines, fines2 in zip(self.coarse_lst, self.fine_lst, self.fine2_lst):
            if coarses is None:
                slice_c, slice_f, slice_f2 = None, None, None
            else:
                if len(coarses) == 1:  # we just have a slice
                    idx = 0
                slice_c, slice_f, slice_f2 = coarses[idx], None, None
                if fines is not None:
                    slice_f = fines[idx]
                    if fines2 is not None:
                        slice_f2 = fines2[idx]
            slice_coarses.append(slice_c)
            slice_fines.append(slice_f)
            slice_fines2.append(slice_f2)
        return slice_coarses, slice_fines, slice_fines2

    def replace_slice(self, i, raw_vals):
        """Replace the value in the ith series with the given raw_val (e.g.,
        after generating a new covariate in the multivariate
        interleaved case, #160).

        """
        self.raw_lst[i][0] = raw_vals

    def filter_trivials(self, triv_mask):
        """When elements of a *batch* are trivial, but we still have raw
        values for those series, then we will have dimension
        mismatches later.  So here we remove those trivials from our
        covariates.

        """
        not_trivial = torch.logical_not(triv_mask)
        new_raw_lst = []
        for raws in self.raw_lst:
            if raws is not None:
                raws = raws[:, not_trivial]
            new_raw_lst.append(raws)
        self.filtered_raw_lst = new_raw_lst
