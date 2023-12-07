"""Use a trained LSTM model to generate sequences of the next ngen
values, priming on an existing set of conditioning of data (each csize
in length).

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/generator.py
import copy
import logging
import torch
from torch.distributions.pareto import Pareto
from sutranets.dataprep.data_utils import DataUtils
from sutranets.dataprep.digitize_utils import DigitizeUtils
from sutranets.ml.bin_tensor_maker import BinTensorMaker
from sutranets.ml.constants import \
    ExampleKeys, INPUT_DTYPE, IntraBinDecoder, ProbError
from sutranets.ml.loss_functions.coarse_cross_entropy_loss import \
    SMOOTHING, CoarseCrossEntropyLoss, PARETO_START_CORRECTION
from sutranets.ml.loss_functions.coarse_cross_entropy_loss import TOLERANCE as PARETO_START_TOLERANCE
logger = logging.getLogger("sutranets.ml.generator")
TOLERANCE = 1e-6
MAX_SAMPLE = 1e12


class Generator():
    """Class to generate sequence of values given a trained LSTM model.

    """
    def __init__(self, device, evaluator,
                 coarse_cutoffs, coarse_low, coarse_high, nsize, ngen,
                 bdecoder, *, extremas, lagfeat_period):
        self.device = self.evaluator = self.nsize = None
        self.coarse_cutoffs = None
        self.lagfeat_period = self.ngen = self.extremas = self.bdecoder = self.softplus = None
        self._init_standard_vars(
            device, evaluator, nsize, coarse_cutoffs,
            lagfeat_period, ngen, extremas, bdecoder)
        if self.extremas:
            _, self.pareto_lo, self.pareto_hi = CoarseCrossEntropyLoss.process_coarse_cutoffs(
                self.coarse_cutoffs)
        else:
            self.pareto_lo = self.pareto_hi = None
        low_high_cutoffs = DigitizeUtils.torch_prepare_cutoffs(
            self.coarse_cutoffs, fixed_lower=coarse_low, fixed_upper=coarse_high)
        self.lower_cutoffs, self.upper_cutoffs = low_high_cutoffs
        self.ncoarse = len(self.coarse_cutoffs) + 1

    def _init_standard_vars(self, device, evaluator, nsize, coarse_cutoffs,
                            lagfeat_period, ngen, extremas, bdecoder):
        """Initialize some of the standard variables """
        self.device = device
        self.evaluator = evaluator
        self.nsize = nsize
        self.coarse_cutoffs = coarse_cutoffs.to(device)
        self.lagfeat_period = lagfeat_period
        self.ngen = ngen
        self.extremas = extremas
        if bdecoder not in [IntraBinDecoder.UNIFORM]:
            raise RuntimeError(f"Unsupported bdecoder {bdecoder}")
        self.bdecoder = bdecoder
        self.softplus = torch.nn.Softplus()

    @staticmethod
    def _unnormalize(generated, nmin, nmax):
        """Given the nmin/nmax for the normalization part of each sequence in
        the batch, broadcast the unmapping across every value in
        generated using each batch's corresponding nmin/nmax values.

        """
        unnormed = generated * (nmax - nmin) + nmin
        unnormed[unnormed < 0] = 0
        return unnormed

    def _get_one_extrema_samps(self, mask, extrema_inputs, pareto_start, fwd_func):
        """Get extrema samps for low or high extrema.

        """
        pareto_start = copy.deepcopy(pareto_start)
        if torch.any(mask):
            extrema_inputs = extrema_inputs[mask]
            outs = fwd_func(extrema_inputs)
            outs = outs.reshape(-1)
            alphas = self.softplus(outs) + SMOOTHING
            pareto_correction = None
            if pareto_start <= PARETO_START_TOLERANCE:
                pareto_correction = PARETO_START_CORRECTION - pareto_start
                pareto_start += pareto_correction
            my_pareto = Pareto(pareto_start, alphas)
            samps = my_pareto.sample()
            if pareto_correction is not None:
                samps -= pareto_correction
            samps[samps > MAX_SAMPLE] = MAX_SAMPLE
        else:
            samps = None
        return samps

    def _get_extrema_samps(self, lo_mask, hi_mask, extrema_inputs):
        """Get lo/hi samples for the values that are extremas, also re-usable
        in child classes.

        """
        lo_samps = self._get_one_extrema_samps(
            lo_mask, extrema_inputs, self.pareto_lo, self.evaluator.batch_forward_ex_low)
        if lo_samps is not None:
            lo_samps *= -1.0
        hi_samps = self._get_one_extrema_samps(
            hi_mask, extrema_inputs, self.pareto_hi, self.evaluator.batch_forward_ex_high)
        return lo_samps, hi_samps

    def _sample_and_replace_extremas(self, decoded, extrema_inputs, lo_mask, hi_mask):
        """Generate the Pareto output parameters for the low/high bins as
        needed, and sample actual outputs based on these parameters,
        and replace them at the corresponding locations in decoded.

        """
        nbatch = len(lo_mask)
        extrema_inputs = extrema_inputs.reshape(nbatch, -1)
        lo_samps, hi_samps = self._get_extrema_samps(
            lo_mask, hi_mask, extrema_inputs)
        if lo_samps is not None:
            decoded[lo_mask] = lo_samps
        if hi_samps is not None:
            decoded[hi_mask] = hi_samps
        return decoded

    def __replace_extremas(self, decoded, binned_coarses, extrema_inputs):
        """Find the extrema points, determine samples for them, and then
        replace the parts of decoded corresponding to them.

        """
        if not self.extremas:
            return decoded
        lo_mask = binned_coarses == 0
        hi_mask = binned_coarses == self.ncoarse - 1
        decoded = self._sample_and_replace_extremas(decoded, extrema_inputs, lo_mask, hi_mask)
        return decoded

    def _decode_batch(self, binned_coarses, extrema_inputs, nmin, nmax):
        """Turn the binned coarses into (normalized) output values, using
        whatever bin decoding strategy.

        """
        binned_coarses = binned_coarses.reshape(-1)
        decoded = DigitizeUtils.torch_decode_values(
            binned_coarses, self.lower_cutoffs, self.upper_cutoffs)
        decoded = self.__replace_extremas(decoded, binned_coarses, extrema_inputs)
        decoded = decoded.reshape(-1, 1)
        unnormed = self._unnormalize(decoded, nmin, nmax)
        if self.extremas:
            return decoded, unnormed
        return None, unnormed

    def _get_binned(self, coarse_inputs):
        """Get the coarse binned values.

        """
        coarse_outputs = self.evaluator.batch_forward_coarse(coarse_inputs)
        coarse_logits = coarse_outputs[-1]
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        try:
            binned_coarses = torch.multinomial(coarse_probs, 1)
        except RuntimeError as multi_err:
            raise ProbError("Error with torch.multinomial") from multi_err
        ccoarses = torch.nn.functional.one_hot(
            binned_coarses.reshape(-1), num_classes=self.ncoarse).to(INPUT_DTYPE)
        return binned_coarses, ccoarses

    def _init_lagperiod_arrays(self, cwin_coarses, cwin_fines, cwin_fines2):
        """Initialize the different arrays we use for keeping track of the
        most recent lagfeat_period values.

        """
        if self.lagfeat_period is None:
            return None, None, None
        lagperiod_coarse = cwin_coarses[-self.lagfeat_period:]
        if cwin_fines is not None:
            lagperiod_fine = cwin_fines[-self.lagfeat_period:]
        else:
            lagperiod_fine = None
        if cwin_fines2 is not None:
            lagperiod_fine2 = cwin_fines2[-self.lagfeat_period:]
        else:
            lagperiod_fine2 = None
        return lagperiod_coarse, lagperiod_fine, lagperiod_fine2

    def _update_lagperiod_arrays(self, lagp_coarses, lagp_fines,
                                 lagp_fines2, bcoarses, bfines, bfines2):
        """Update arrays for keeping track of the most recent lagfeat_period
        values.

        """
        if self.lagfeat_period is None:
            return None, None, None
        lag_arrs = [lagp_coarses, lagp_fines, lagp_fines2]
        new_vals = [bcoarses, bfines, bfines2]
        modified_arrs = []
        for lag_arr, new_vals in zip(lag_arrs, new_vals):
            if lag_arr is not None:
                lag_arr = lag_arr.roll(-1, 0)
                lag_arr[-1] = new_vals
            modified_arrs.append(lag_arr)
        return (*modified_arrs,)

    @staticmethod
    def _check_one_hot(mybins, nbins):
        """Encode one set of bins"""
        if mybins is None:
            return None
        return torch.nn.functional.one_hot(mybins, num_classes=nbins).to(INPUT_DTYPE)

    @classmethod
    def _one_hot_bins(cls, coarse, fine, fine2, ncoarse, nfine, nfine2):
        """Encode batches of coarse/fine/fine2 bins as one-hot-features, if
        given.

        """
        one_h_coarse = cls._check_one_hot(coarse, ncoarse)
        one_h_fine = cls._check_one_hot(fine, nfine)
        one_h_fine2 = cls._check_one_hot(fine2, nfine2)
        return one_h_coarse, one_h_fine, one_h_fine2

    def _get_covar_feats(self, bin_covars, targ_idx, ncoarse, nfine, nfine2):
        """Get the covariate features for coarse/fine/fine2, depending on
        whether we are generator or a derived class, by reading from
        bin_covars at the given targ_idx.

        """
        if bin_covars is None:
            return None, None, None
        one_h_coarse_lst = []
        one_h_fine_lst = []
        one_h_fine2_lst = []
        coarse_lst, fine_lst, fine2_lst = bin_covars.get_at_idx(targ_idx)
        for coarse, fine, fine2 in zip(coarse_lst, fine_lst, fine2_lst):
            one_h_coarse, one_h_fine, one_h_fine2 = self._one_hot_bins(
                coarse, fine, fine2, ncoarse, nfine, nfine2)
            one_h_coarse_lst.append(one_h_coarse)
            one_h_fine_lst.append(one_h_fine)
            one_h_fine2_lst.append(one_h_fine2)
        return one_h_coarse_lst, one_h_fine_lst, one_h_fine2_lst

    def _init_covar_inputs(self, last_inputs, bin_covars, ncoarse, nfine, nfine2):
        """If using covariates, initialize them within the last_inputs tensor,
        before first iteration in build_seqs.

        """
        targ_idx = 0
        one_h_coarse_lst, one_h_fine_lst, one_h_fine2_lst = self._get_covar_feats(
            bin_covars, targ_idx, ncoarse, nfine, nfine2)
        if one_h_coarse_lst is None:
            return last_inputs
        last_inputs = BinTensorMaker.reset_covar_features(
            last_inputs, ncoarse, covar_coarse=one_h_coarse_lst, nfine=nfine, nfine2=nfine2,
            lagfeat_period=self.lagfeat_period, covar_fine=one_h_fine_lst, covar_fine2=one_h_fine2_lst)
        return last_inputs

    @staticmethod
    def _get_lagfeats(lagp_coarse, lagp_fine, lagp_fine2, ncoarse, nfine, nfine2):
        """Encode the lagged features as input.

        """
        if lagp_coarse is None:
            return None, None, None
        lag_bin = lagp_coarse[0, :, 0]
        lcoarse = torch.nn.functional.one_hot(lag_bin, num_classes=ncoarse).to(INPUT_DTYPE)
        if lagp_fine is not None:
            lag_bin = lagp_fine[0, :, 0]
            lfine = torch.nn.functional.one_hot(lag_bin, num_classes=nfine).to(INPUT_DTYPE)
        else:
            lfine = None
        if lagp_fine2 is not None:
            lag_bin = lagp_fine2[0, :, 0]
            lfine2 = torch.nn.functional.one_hot(lag_bin, num_classes=nfine2).to(INPUT_DTYPE)
        else:
            lfine2 = None
        return lcoarse, lfine, lfine2

    def _build_outseqs(self, last_inputs, originals, cwindow_coarses, bin_covars):
        """Starting with the last set of inputs from the conditioning window
        (and a network that is already conditioned), auto-regressively
        call the network to build output sequences for all sequences
        in the batch.

        """
        csize = len(originals) - 2
        nmin, nmax = DataUtils.get_nmin_nmax(csize, self.nsize,
                                             originals, check_trivial=False)
        last_inputs = self._init_covar_inputs(last_inputs, bin_covars, self.ncoarse, None, None)
        lagp_coarses, _, _ = self._init_lagperiod_arrays(cwindow_coarses, None, None)
        for idx in range(self.ngen):
            coarse_inputs, extrema_inputs = BinTensorMaker.extract_coarse_extremas(
                last_inputs, extremas=self.extremas)
            binned_coarses, ccoarse = self._get_binned(coarse_inputs)
            lagp_coarses, _, _ = self._update_lagperiod_arrays(
                lagp_coarses, None, None, binned_coarses, None, None)
            normed, generated = self._decode_batch(binned_coarses, extrema_inputs, nmin, nmax)
            yield generated
            if idx == self.ngen - 1:
                break
            lcoarse, _, _ = self._get_lagfeats(lagp_coarses, None, None, self.ncoarse, None, None)
            covar_coarse, _, _ = self._get_covar_feats(bin_covars, idx + 1, self.ncoarse, None, None)
            pcoarse = ccoarse
            last_inputs = BinTensorMaker.reset_ccoarse_inputs(
                last_inputs, pcoarse, normed=normed, lcoarse=lcoarse,
                covar_coarse=covar_coarse)

    def _get_cwindow_seqs(self, batch, do_fines, do_fines2):
        """Get the relevant sequences from the cwindow, depending on the
        situation.

        """
        inputs = batch[ExampleKeys.INPUT]
        originals = batch[ExampleKeys.ORIGINALS]
        cwin_coarses = cwin_fines = cwin_fines2 = None
        if self.lagfeat_period is not None:
            cwin_coarses = batch.get(ExampleKeys.COARSES)
            if cwin_coarses is None:
                raise RuntimeError("Need COARSE bins in batch.")
            if do_fines:
                cwin_fines = batch.get(ExampleKeys.FINES)
                if cwin_fines is None:
                    raise RuntimeError("Need FINE bins in batch.")
            if do_fines2:
                cwin_fines2 = batch.get(ExampleKeys.FINES2)
                if cwin_fines2 is None:
                    raise RuntimeError("Need FINE2 bins in batch.")
        return inputs, originals, cwin_coarses, cwin_fines, cwin_fines2

    def __call__(self, batch, bin_covars=None):
        """Generate self.ngen points forward on the given batch of data.

        """
        if batch is None:
            for _ in range(self.ngen):
                yield None
        inputs, originals, cwindow_coarses, _, _ = self._get_cwindow_seqs(
            batch, do_fines=False, do_fines2=False)
        batch_size = inputs.shape[1]
        self.evaluator.run_init_hidden(batch_size)
        _, _, originals, cwindow_coarses = self.evaluator.batch_forward_outputs(
            inputs[:-1], None, originals, cwindow_coarses)
        last_inputs = inputs[-1:]
        for generated in self._build_outseqs(
                last_inputs, originals, cwindow_coarses, bin_covars):
            yield generated
