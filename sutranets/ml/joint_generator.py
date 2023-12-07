"""Joint generator subclass of generator, for coarse/fine binnings.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD
"""
# sutranets/ml/joint_generator.py
import logging
import torch
from sutranets.dataprep.data_utils import DataUtils
from sutranets.dataprep.digitize_utils import DigitizeUtils
from sutranets.ml.constants import INPUT_DTYPE, ProbError
from sutranets.ml.fc_digitizer import FCDigitizer
from sutranets.ml.bin_tensor_maker import BinTensorMaker
from sutranets.ml.generator import Generator
from sutranets.ml.loss_functions.joint_cross_entropy_loss import JointCrossEntropyLoss
logger = logging.getLogger("sutranets.ml.joint_generator")
TOLERANCE = 1e-6


class JointGenerator(Generator):
    """Generate a sequence of values given a trained JointLSTM model.

    """
    def __init__(self, ncoarse, nfine, device, evaluator,
                 coarse_cutoffs, coarse_low,
                 coarse_high, nsize, ngen, bdecoder, *,
                 extremas, lagfeat_period):
        self.device = self.evaluator = self.nsize = None
        self.coarse_cutoffs = None
        self.lagfeat_period = self.ngen = self.extremas = self.bdecoder = self.softplus = None
        self.ncoarse = self.nfine = self.coarse_cuts_full = self.fcutoffs = None
        self._init_standard_vars(
            device, evaluator, nsize, coarse_cutoffs,
            lagfeat_period, ngen, extremas, bdecoder,
            ncoarse, nfine, coarse_low, coarse_high)
        if self.extremas:
            _, self.pareto_lo, self.pareto_hi = JointCrossEntropyLoss.process_fine_cutoffs(
                self.coarse_cutoffs, nfine)
        else:
            self.pareto_lo = self.pareto_hi = None
        low_high_fcutoffs = DigitizeUtils.torch_prepare_cutoffs(
            self.fcutoffs, fixed_lower=0.0, fixed_upper=1.0)
        self.lower_fcutoffs, self.upper_fcutoffs = low_high_fcutoffs

    def _init_standard_vars(self, device, evaluator, nsize, coarse_cutoffs,
                            lagfeat_period, ngen, extremas, bdecoder,
                            ncoarse, nfine, coarse_low, coarse_high):
        super()._init_standard_vars(device, evaluator, nsize, coarse_cutoffs,
                                    lagfeat_period, ngen, extremas, bdecoder)
        self.ncoarse = ncoarse
        self.nfine = nfine
        self.coarse_cuts_full = FCDigitizer.expand_cuts(self.coarse_cutoffs, coarse_low, coarse_high)
        self.fcutoffs = FCDigitizer.get_fine_cutoffs(self.nfine, device=device)

    def __replace_extremas(self, decoded, binned_coarses, binned_fines, extrema_inputs):
        """Find the extrema points, determine samples for them, and then
        replace the parts of decoded corresponding to them.

        """
        if not self.extremas:
            return decoded
        lo_mask = torch.logical_and(binned_coarses == 0, binned_fines == 0)
        hi_mask = torch.logical_and(binned_coarses == self.ncoarse - 1, binned_fines == self.nfine - 1)
        decoded = self._sample_and_replace_extremas(decoded, extrema_inputs, lo_mask, hi_mask)
        return decoded

    def _decode_batch(self, bcoarses, bfines, extrema_inputs, nmin, nmax):
        """Turn the batches of binned coarse/fine values into real output
        values.

        """
        bcoarses, bfines = bcoarses.reshape(-1), bfines.reshape(-1)
        coarse_bins_lower = self.coarse_cuts_full[bcoarses]
        coarse_bins_higher = self.coarse_cuts_full[bcoarses + 1]
        dfines = DigitizeUtils.torch_decode_values(
            bfines, self.lower_fcutoffs, self.upper_fcutoffs)
        decoded = (dfines * (coarse_bins_higher - coarse_bins_lower) +
                   coarse_bins_lower)
        decoded = self.__replace_extremas(decoded, bcoarses, bfines, extrema_inputs)
        decoded = decoded.reshape(-1, 1)
        unnormed = self._unnormalize(decoded, nmin, nmax)
        if self.extremas:
            return decoded, unnormed
        return None, unnormed

    def _get_binned(self, coarse_inputs, fine_inputs):
        """Get the coarse/fine binned values and ccoarse.

        """
        binned_coarses, ccoarses = super()._get_binned(coarse_inputs)
        fine_inputs = BinTensorMaker.replace_ccoarse(fine_inputs, ccoarses)
        fine_output = self.evaluator.batch_forward_fine(fine_inputs)
        fine_logits = fine_output[-1]
        fine_probs = torch.softmax(fine_logits, dim=1)
        try:
            binned_fines = torch.multinomial(fine_probs, 1)
        except RuntimeError as multi_err:
            raise ProbError("Error with torch.multinomial") from multi_err
        cfines = torch.nn.functional.one_hot(
            binned_fines.reshape(-1), num_classes=self.nfine).to(INPUT_DTYPE)
        return binned_coarses, binned_fines, ccoarses, cfines

    def _build_outseqs(self, last_inputs, originals, cwindow_coarses, cwindow_fines, bin_covars):
        """Starting with the last set of inputs from the conditioning window
        (and a network that is already conditioned), auto-regressively
        call the coarse and fine sub-networks to build output
        sequences for all sequences in the batch.

        """
        csize = len(originals) - 2
        nmin, nmax = DataUtils.get_nmin_nmax(csize, self.nsize,
                                             originals, check_trivial=False)
        last_inputs = self._init_covar_inputs(last_inputs, bin_covars, self.ncoarse, self.nfine, None)
        lagp_coarses, lagp_fines, _ = self._init_lagperiod_arrays(
            cwindow_coarses, cwindow_fines, None)
        for idx in range(self.ngen):
            coarse_inputs, fine_inputs, extrema_inputs = BinTensorMaker.extract_coarse_fine(
                last_inputs, self.ncoarse, extremas=self.extremas)
            binned_coarses, binned_fines, ccoarses, cfines = self._get_binned(
                coarse_inputs, fine_inputs)
            lagp_coarses, lagp_fines, _ = self._update_lagperiod_arrays(
                lagp_coarses, lagp_fines, None, binned_coarses, binned_fines, None)
            normed, generated = self._decode_batch(
                binned_coarses, binned_fines, extrema_inputs, nmin, nmax)
            yield generated
            if idx == self.ngen - 1:
                break
            lcoarse, lfine, _ = self._get_lagfeats(
                lagp_coarses, lagp_fines, None, self.ncoarse, self.nfine, None)
            covar_coarse, covar_fine, _ = self._get_covar_feats(
                bin_covars, idx + 1, self.ncoarse, self.nfine, None)
            pfine, pcoarse = cfines, ccoarses
            last_inputs = BinTensorMaker.reset_ccoarse_inputs(
                last_inputs, pcoarse, pfine=pfine,
                normed=normed, lcoarse=lcoarse, lfine=lfine,
                covar_coarse=covar_coarse, covar_fine=covar_fine)

    def __call__(self, batch, bin_covars=None):
        """Generate self.ngen points forward on the given batch of data.

        """
        if batch is None:
            for _ in range(self.ngen):
                yield None
        inputs, originals, cwindow_coarses, cwindow_fines, _ = self._get_cwindow_seqs(
            batch, do_fines=True, do_fines2=False)
        batch_size = inputs.shape[1]
        self.evaluator.run_init_hidden(batch_size)
        _, _, originals, cwindow_coarses, cwindow_fines = self.evaluator.batch_forward_outputs(
            inputs[:-1], None, originals, cwindow_coarses, cwindow_fines)
        last_inputs = inputs[-1:]
        for generated in self._build_outseqs(
                last_inputs, originals, cwindow_coarses, cwindow_fines, bin_covars):
            yield generated
