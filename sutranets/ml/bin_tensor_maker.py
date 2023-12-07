"""A class with the job of making tensors for the input and output,
for the binned representations.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/bin_tensor_maker.py
import torch
from sutranets.ml.constants import INPUT_DTYPE, DUMMY_BIN
from sutranets.ml.fc_digitizer import FCDigitizer
from sutranets.ml.loss_functions.constants import IGNORE_INDEX


class BinTensorMaker():
    """Make tensors for inputs and outputs, as requested.

    """
    __DUMMY_VAL = 0.0

    def __init__(self, coarse_cutoffs, coarse_low,
                 coarse_high, nfine_bins, nfine2_bins, genmode, *,
                 extremas, lagfeat_period, bin_dropout=None):
        self.extremas = extremas
        if coarse_cutoffs is None:
            raise RuntimeError("coarse_cutoffs required for bin tensor maker.")
        if nfine2_bins is not None and nfine_bins is None:
            raise RuntimeError("Can only use fine2 if using fine.")
        self.nfine_bins = nfine_bins
        self.nfine2_bins = nfine2_bins
        self.fc_digitizer = FCDigitizer(coarse_cutoffs, coarse_low,
                                        coarse_high,
                                        nfine_bins, nfine2_bins, extremas=extremas)
        self.lagfeat_period = lagfeat_period
        self.ncoarse_bins = len(coarse_cutoffs) + 1
        self.genmode = genmode
        self.bin_dropout = bin_dropout

    @staticmethod
    def get_nbatch(normed):
        """Helper to figure out if there's a batch dimension in an input
        tensor and return the dimensionality.

        """
        if not torch.is_tensor(normed):
            normed = torch.tensor(normed)
        if len(normed.shape) == 1:
            nbatch = -1
        elif len(normed.shape) == 2:
            nbatch = normed.shape[1]
        else:
            raise RuntimeError(f"Weird shape for input normed: {normed.shape}")
        return nbatch

    @staticmethod
    def compute_nsubsets(ninput, ncoarse_bins, nfine_bins,
                         nfine2_bins, extremas):
        """Return how many features in the coarse, fine, and fine2 (input)
        *overlapping* subsets.

        """
        nextrema = 1 if extremas else None
        if nfine_bins is None:
            return ninput, None, None, nextrema
        if nfine2_bins is None:
            nfeats = ninput - ncoarse_bins
            return nfeats, nfeats, None, nextrema
        nfeats = ninput - ncoarse_bins - nfine_bins
        return nfeats, nfeats, nfeats, nextrema

    @staticmethod
    def compute_joint_ninput(ncoarse_subset, ncoarse):
        return ncoarse_subset + ncoarse

    @staticmethod
    def compute_triple_ninput(ncoarse_subset, ncoarse, nfine):
        return ncoarse_subset + ncoarse + nfine

    @classmethod
    def __one_hot_values(cls, values, nbins):
        """1-hot-encode the given values, assuming the given number of bins.

        """
        if not torch.is_tensor(values):
            values = torch.tensor(values)
        tensor = torch.nn.functional.one_hot(values,
                                             num_classes=nbins) \
                                    .to(INPUT_DTYPE)
        nbatch = cls.get_nbatch(values)
        if nbatch < 0:
            tensor = tensor.unsqueeze(dim=1)
        return tensor

    def digitize(self, meta, origs, bin_covars, csize, nsize, trivial_min_nonzero_frac):
        """Return the digitized coarses, deltas, fine-grained bins (if using),
        and fine2 (if using), and normed originals.

        """
        return self.fc_digitizer(meta, origs, bin_covars, csize, nsize, trivial_min_nonzero_frac)

    def __shift_first_three(self, coarses, normed):
        """Helper to shift the coarses, deltas, and normed

        """
        pcoarses = coarses[1:]
        pnormed = normed
        if pnormed is not None:
            pnormed = pnormed[1:]
        if not self.genmode:
            pcoarses = pcoarses[:-1]
            if pnormed is not None:
                pnormed = pnormed[:-1]
        return pcoarses, pnormed

    def __shift_inputs(self, coarses, fines, fines2, normed):
        """The input here are tensors of indexes, representing ranges of the
        time series.  Depending on what mode we're in, and whether
        joint, adjust what range we can see for each given tensor, and
        return the adjusted (shifted) versions.

        """
        nbatch = self.get_nbatch(coarses)
        pcoarses, pnormed = self.__shift_first_three(coarses, normed)
        if self.nfine_bins is None:
            return pcoarses, pnormed, None, None, None, None
        pfines = fines[1:]
        ccoarses = coarses[2:]
        if self.genmode:
            dummy_tens = torch.tensor([DUMMY_BIN], device=ccoarses.device)
            if nbatch > 0:
                dummy_tens = dummy_tens.expand(1, nbatch)
            ccoarses = torch.cat([ccoarses, dummy_tens])
        else:
            pfines = pfines[:-1]
        if self.nfine2_bins is None:
            return pcoarses, pnormed, pfines, ccoarses, None, None
        pfines2 = fines2[1:]
        cfines = fines[2:]
        if self.genmode:
            dummy_tens = torch.tensor([DUMMY_BIN], device=cfines.device)
            if nbatch > 0:
                dummy_tens = dummy_tens.expand(1, nbatch)
            cfines = torch.cat([cfines, dummy_tens])
        else:
            pfines2 = pfines2[:-1]
        return pcoarses, pnormed, pfines, ccoarses, pfines2, cfines

    def _one_hot_lagged(self, pbin_indices, nbins):
        """Get a version of the pbin_indices (previous bin indices) that have
        been lagged by the lag_period.

        """
        if self.lagfeat_period == 1:
            lagged_idxs = pbin_indices
        else:
            lagged_idxs = torch.cat(
                [pbin_indices[1:self.lagfeat_period],
                 pbin_indices[:-self.lagfeat_period+1]], dim=0)
        one_h_lagged = self.__one_hot_values(lagged_idxs, nbins)
        return one_h_lagged

    @staticmethod
    def __check_encode_add(feats_list, idxs_or_vals, nbins, encode_func):
        """Helper to check if a value exists before encoding, and then encode
        with the given one-hot-style function, and then add to the
        features list.

        """
        if idxs_or_vals is not None:
            if nbins is not None:
                encoded = encode_func(idxs_or_vals, nbins)
            else:
                encoded = encode_func(idxs_or_vals)
            feats_list.append(encoded)

    def __add_covariates(self, feats_list, pcoarses, pfines,
                         pfines2, bin_covars):
        """Helper to add features to the featlist for our block of (possible)
        covariates, i.e., normed, and lagged.

        """
        if self.lagfeat_period is not None:
            self.__check_encode_add(feats_list, pcoarses, self.ncoarse_bins, self._one_hot_lagged)
            self.__check_encode_add(feats_list, pfines, self.nfine_bins, self._one_hot_lagged)
            self.__check_encode_add(feats_list, pfines2, self.nfine2_bins, self._one_hot_lagged)
        if bin_covars is None:
            return
        coarse_lst, fine_lst, fine2_lst = bin_covars.get_lsts()
        for coarses, fines, fines2 in zip(coarse_lst, fine_lst, fine2_lst):
            if coarses is None:
                coarses, fines, fines2 = pcoarses, pfines, pfines2
            self.__check_encode_add(feats_list, coarses, self.ncoarse_bins, self.__one_hot_values)
            self.__check_encode_add(feats_list, fines, self.nfine_bins, self.__one_hot_values)
            self.__check_encode_add(feats_list, fines2, self.nfine2_bins, self.__one_hot_values)

    @classmethod
    def __make_normed_input(cls, normed):
        """Helper to make the normed values for features, when doing extrema.

        """
        nan_ones = normed.isnan()
        normed = normed.clone()
        normed[nan_ones] = cls.__DUMMY_VAL
        normed = torch.sigmoid(normed)
        nbatch = cls.get_nbatch(normed)
        if nbatch < 0:
            nbatch = 1
        return normed.reshape(-1, nbatch, 1).to(INPUT_DTYPE)

    def __apply_dropout(self, inputs):
        """Apply dropout to the inputs and return the dropped version."""
        if self.bin_dropout is None or self.genmode:
            return inputs
        nskip = 0
        if self.nfine_bins is not None:
            nskip += self.ncoarse_bins
        if self.nfine2_bins is not None:
            nskip += self.nfine_bins
        targ_inputs = inputs[:, :, :-nskip]
        keep_mask = torch.rand_like(targ_inputs) > self.bin_dropout
        targ_inputs = targ_inputs * keep_mask
        targ_inputs = targ_inputs / (1.0 - self.bin_dropout)
        inputs[:, :, :-nskip] = targ_inputs
        return inputs

    def encode_input(self, coarses, *, fines=None, fines2=None, normed=None,
                     bin_covars=None):
        """Given the key input information, we encode it.

        """
        pcoarses, pnormed, pfines, ccoarses, pfines2, cfines = self.__shift_inputs(
            coarses, fines, fines2, normed)
        if bin_covars is not None:
            bin_covars.shift(self.genmode)
        feats_list = []
        one_h_pcoarse = self.__one_hot_values(pcoarses, self.ncoarse_bins)
        feats_list.append(one_h_pcoarse)  # always
        self.__add_covariates(feats_list, pcoarses, pfines, pfines2, bin_covars)
        self.__check_encode_add(feats_list, pnormed, None, self.__make_normed_input)
        if self.nfine_bins is not None:
            one_h_pfine = self.__one_hot_values(pfines, self.nfine_bins)
            one_h_ccoarse = self.__one_hot_values(ccoarses, self.ncoarse_bins)
            feats_list.insert(1, one_h_pfine)
            feats_list.append(one_h_ccoarse)
            if self.nfine2_bins is not None:
                one_h_pfine2 = self.__one_hot_values(pfines2, self.nfine2_bins)
                one_h_cfine = self.__one_hot_values(cfines, self.nfine_bins)
                feats_list.insert(2, one_h_pfine2)
                feats_list.append(one_h_cfine)
        inputs = torch.cat(feats_list, dim=2)
        inputs = self.__apply_dropout(inputs)
        return inputs

    @staticmethod
    def extract_coarse_extremas(inputs, extremas):
        coarse_inputs = inputs
        if not extremas:
            return coarse_inputs, None
        nextrema_feats = 1
        extrema_inputs = inputs[:, :, -nextrema_feats:]
        return coarse_inputs, extrema_inputs

    @staticmethod
    def extract_coarse_fine(inputs, ncoarse_bins, extremas):
        nfeats = inputs.shape[-1]
        nsubset = nfeats - ncoarse_bins
        coarse_inputs = inputs[:, :, :nsubset]
        fine_inputs = inputs[:, :, -nsubset:]
        if not extremas:
            return coarse_inputs, fine_inputs, None
        extrema_inputs = inputs[:, :, nsubset-1:nsubset]
        return coarse_inputs, fine_inputs, extrema_inputs

    @staticmethod
    def extract_coarse_fine_fine2(inputs, ncoarse_bins, nfine_bins, extremas):
        ntotal = inputs.shape[-1]
        coarse_end = ntotal - ncoarse_bins - nfine_bins
        fine_start = ncoarse_bins
        fine_end = ntotal - nfine_bins
        fine2_start = ncoarse_bins + nfine_bins
        coarse_inputs = inputs[:, :, :coarse_end]
        fine_inputs = inputs[:, :, fine_start:fine_end]
        fine2_inputs = inputs[:, :, fine2_start:]
        if not extremas:
            return coarse_inputs, fine_inputs, fine2_inputs, None
        extrema_inputs = inputs[:, :, coarse_end-1:coarse_end]
        return coarse_inputs, fine_inputs, fine2_inputs, extrema_inputs

    @staticmethod
    def replace_ccoarse(fine_inputs, ccoarse):
        """Background: the inputs can be divided into
        coarse/fine/fine2-specific inputs. Replace the ccoarse
        features in the fine-specific inputs.

        """
        ncoarse = ccoarse.shape[-1]
        fine_inputs[0, :, -ncoarse:] = ccoarse
        return fine_inputs

    @staticmethod
    def replace_cfine(fine2_inputs, cfine):
        """Replace the cfine features in the fine2-specific inputs.

        """
        nfine = cfine.shape[-1]
        fine2_inputs[0, :, -nfine:] = cfine
        return fine2_inputs

    @staticmethod
    def __check_add(replacement_lst, encoded_lst):
        """Simple helper to check if tensors in a list are not None before
        adding to the replacement_lst.

        """
        for feats in encoded_lst:
            if feats is not None:
                replacement_lst.append(feats)

    @staticmethod
    def __sub_replacement_lst(inputs, replacement_lst, start_pt):
        """Helper to concatenate the elements of the replacement_lst in the
        feature dimension, then substitute into the inputs at start_pt
        in the feature dimension.

        """
        if not replacement_lst:
            return
        if len(replacement_lst) == 1:
            replaced_parts = replacement_lst[0]
        else:
            replaced_parts = torch.cat(replacement_lst, dim=1)
        nfeats = replaced_parts.shape[-1]
        inputs[0, :, start_pt:start_pt+nfeats] = replaced_parts
        return

    @classmethod
    def reset_ccoarse_inputs(cls, inputs, pcoarse, *, pfine=None,
                             pfine2=None, normed=None,
                             lcoarse=None, lfine=None, lfine2=None,
                             covar_coarse=None, covar_fine=None, covar_fine2=None):
        """Background: we adjust the inputs incrementally over time.  This
        function is the "reset" to be ready for the next iteration -
        basically, make it ready to generate the next coarse bins
        (ccoarse), which is the first step.  Replace subsets of input
        features in the overall inputs.

        """
        replacement_lst = [pcoarse]
        cls.__check_add(replacement_lst, [pfine, pfine2, lcoarse, lfine, lfine2])
        if covar_coarse is not None:
            for idx, coarses in enumerate(covar_coarse):
                fines, fines2 = None, None
                if covar_fine is not None:
                    fines = covar_fine[idx]
                    if covar_fine2 is not None:
                        fines2 = covar_fine2[idx]
                if coarses is None:
                    coarses, fines, fines2 = pcoarse, pfine, pfine2
                cls.__check_add(replacement_lst, [coarses, fines, fines2])
        if normed is not None:
            normed = torch.sigmoid(normed)
            replacement_lst.append(normed)
        cls.__sub_replacement_lst(inputs, replacement_lst, start_pt=0)
        return inputs

    @classmethod
    def reset_covar_features(cls, inputs, ncoarse, *, covar_coarse, nfine=None,
                             nfine2=None, lagfeat_period=False,
                             covar_fine=None, covar_fine2=None):
        replacement_lst = []
        for idx, coarses in enumerate(covar_coarse):
            fines, fines2 = None, None
            if covar_fine is not None:
                fines = covar_fine[idx]
                if covar_fine2 is not None:
                    fines2 = covar_fine2[idx]
            cls.__check_add(replacement_lst, [coarses, fines, fines2])
        if nfine is None:
            nfine = 0
        if nfine2 is None:
            nfine2 = 0
        if lagfeat_period is not None:
            multiplier = 2
        else:
            multiplier = 1
        start_pt = (ncoarse + nfine + nfine2) * multiplier
        cls.__sub_replacement_lst(inputs, replacement_lst, start_pt)
        return inputs

    @staticmethod
    def encode_originals(originals):
        if originals is None:
            return None
        return originals.unsqueeze(1).unsqueeze(1)

    def __add_masked_extremas(self, normed, stack_list):
        """Add on 'normed' targets for extremas_low and extremas_high, but
        only in those cases where we are in the respective extreme
        bins, otherwise just put IGNORE_INDEX.

        """
        normed = normed[2:]
        coarse_idxs = stack_list[0]
        in_low = coarse_idxs == 0
        in_high = coarse_idxs == self.ncoarse_bins - 1
        nfine12_bins = [self.nfine_bins, self.nfine2_bins]
        for idxs, nbins in zip(stack_list[1:], nfine12_bins):
            if idxs is not None:
                in_low = torch.logical_and(in_low, (idxs == 0))
                in_high = torch.logical_and(in_high, (idxs == (nbins - 1)))
        extremas_low, extremas_high = normed.clone(), normed.clone()
        extremas_low[~in_low] = IGNORE_INDEX
        extremas_high[~in_high] = IGNORE_INDEX
        stack_list.append(extremas_low)
        stack_list.append(extremas_high)

    def encode_target(self, coarses, *, fines=None, fines2=None, normed=None):
        stack_list = []
        coarse_idxs = coarses[2:]
        nseq = len(coarse_idxs)
        if self.nfine_bins is None:
            coarse_idxs = coarse_idxs.clone()
        stack_list = [coarse_idxs]
        if self.nfine_bins is not None:
            stack_list.append(fines[2:])
        if self.nfine2_bins is not None:
            stack_list.append(fines2[2:])
        if self.extremas:
            self.__add_masked_extremas(normed, stack_list)
        targets = torch.stack(stack_list, dim=1).reshape(nseq, 1, -1)
        return targets
