"""Use a trained LSTM model to generate sequences of the next ngen
values, priming on an existing set of conditioning of data (each csize
in length).

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/generator.py

import copy
import logging
import torch
from torch.distributions.pareto import Pareto
from c2far.dataprep.data_utils import DataUtils
from c2far.dataprep.digitize_utils import DigitizeUtils
from c2far.ml.bin_tensor_maker import BinTensorMaker
from c2far.ml.constants import \
    ExampleKeys, INPUT_DTYPE, IntraBinDecoder, ProbError
from c2far.ml.loss_functions.coarse_cross_entropy_loss import \
    SMOOTHING, CoarseCrossEntropyLoss, PARETO_START_CORRECTION
from c2far.ml.loss_functions.coarse_cross_entropy_loss import TOLERANCE as PARETO_START_TOLERANCE
logger = logging.getLogger("c2far.ml.generator")
TOLERANCE = 1e-6
# In cases where our Pareto has almost-infinite support (typically
# when we have immature preds), we may sample HUGE values, which cause
# problems, so cap them to be this:
MAX_SAMPLE = 1e12  # Just some huge number


class Generator():
    """Class to generate a sequence of values given a trained LSTM model.

    """
    def __init__(self, device, evaluator,
                 coarse_cutoffs, coarse_low, coarse_high, nsize, ngen,
                 bdecoder, *, extremas):
        """Initialize with the neural net evaluator (or whatever
        binned-output-generating evaluator) and the other things we
        need for generation.

        Arguments:

        device: String, e.g. "cuda:0" or "cpu"

        evaluator: Evaluator, e.g., LSTMEvaluator: what we use to call
        `batch_forward_outputs()`.

        coarse_cutoffs: Tensor[NBINS], for encoding coarse features and
        converting vals to origs

        coarse_low: Float, the lowest implicit boundary for the coarse
        cutoffs.

        coarse_high: Float, the highest implicit boundary for the
        coarse cutoffs.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        ngen: Int, length of generation window

        bdecoder: IntraBinDecoder, the method for decoding the value
        within each coarse bin.

        Keyword-only Arguments:

        extremas: Boolean, if True, use the net's fully-connected
        sub-network to generate outputs when in the extreme bins, else
        False.

        """
        self.device = device
        self.evaluator = evaluator
        self.nsize = nsize
        coarse_cutoffs = coarse_cutoffs.to(device)
        low_high_cutoffs = DigitizeUtils.torch_prepare_cutoffs(
            coarse_cutoffs, fixed_lower=coarse_low, fixed_upper=coarse_high)
        self.lower_cutoffs, self.upper_cutoffs = low_high_cutoffs
        self.ncoarse = len(coarse_cutoffs) + 1
        self.ngen = ngen
        self.extremas = extremas
        if bdecoder not in [IntraBinDecoder.UNIFORM]:
            raise RuntimeError(f"Unsupported bdecoder {bdecoder}")
        self.bdecoder = bdecoder
        if self.extremas:
            _, self.pareto_lo, self.pareto_hi = CoarseCrossEntropyLoss.process_coarse_cutoffs(coarse_cutoffs)
        else:
            self.pareto_lo = self.pareto_hi = None
        self.softplus = torch.nn.Softplus()

    @staticmethod
    def _unnormalize(generated, nmin, nmax):
        """Calculate the nmin/nmax for the normalization part of each sequence
        in the batch, allowing us to broadcast the unmapping across
        every value in generated using each batch's corresponding
        nmin/nmax values.

        Arguments

        generated: NBATCH x 1

        nmin/nmax: each NBATCH x 1

        Returns:

        unnormed: NBATCH x 1

        """
        # generated is NBATCH x 1, so it's ripe for broadcasting:
        unnormed = generated * (nmax - nmin) + nmin
        # Fix any unmapping that is negative (below zero) [IMPORTANT:
        # this should not be used with series that go < 0]:
        unnormed[unnormed < 0] = 0
        return unnormed

    def _get_one_extrema_samps(self, mask, extrema_inputs, pareto_start, fwd_func):
        """Helper that can get extrema samps for low or high extrema.

        Note on pareto_start: as in coarse_cross_entropy_loss, it is
        possible pareto_start will be NEGATIVE, e.g., if the last
        cutoff is negative and we're going upwards OR the first cutoff
        is positive and we're going downwards.  In this case, we
        assume it starts from PARETO_START_CORRECTION, and we shift
        all the targets up accordingly.

        Arguments:

        mask, Tensor[Boolean]: NBATCH, identifying points that are
        extrema (True) or not (False).

        extrema_inputs: NBATCH x NINPUTS=1, the one input per batch
        for the extrema forward function.

        pareto_start: Float, the (positive) starting point for the
        (low or high) Pareto distribution.

        fwd_func: either batch_forward_ex_high or batch_forward_ex_lo.

        Returns:

        samps, Tensor[NEXTREMA]

        """
        # Copy is essential since you may adjust it:
        pareto_start = copy.deepcopy(pareto_start)
        if torch.any(mask):
            extrema_inputs = extrema_inputs[mask]
            outs = fwd_func(extrema_inputs)
            # From this point, work in 1D:
            outs = outs.reshape(-1)
            alphas = self.softplus(outs) + SMOOTHING
            # Do the correction if needed:
            pareto_correction = None
            if pareto_start <= PARETO_START_TOLERANCE:
                pareto_correction = PARETO_START_CORRECTION - pareto_start
                pareto_start += pareto_correction
            my_pareto = Pareto(pareto_start, alphas)
            samps = my_pareto.sample()
            if pareto_correction is not None:
                # We assumed you started from PARETO_START_CORRECTION
                # rather than where you did start, so fix that now:
                samps -= pareto_correction
                # e.g. you started at -0.1, we assume you started at
                # 0.1, so we shift the start up by 0.2, get our
                # samples (e.g. one at 0.15), and then shift that down
                # by 0.2 (e.g. now at -0.05), and so we can have both
                # positive and negative samples.
            # Prevent problems from +-inf in samples:
            samps[samps > MAX_SAMPLE] = MAX_SAMPLE
        else:
            samps = None
        return samps

    def _get_extrema_samps(self, lo_mask, hi_mask, extrema_inputs):
        """Helper to get lo/hi samples for the values that are extremas, also
        re-usable in child classes.

        Arguments:

        lo_mask: Tensor: NBATCH, which values are low extrema

        hi_mask: Tensor: NBATCH, which values are high extrema

        extrema_inputs: Tensor: 1 x NBATCH x NSUBSET=1, the one extrema feature

        Returns:

        lo_samps, hi_samps, each Tensor(Float): N{LO/HI}EXTREMA, the
        lo/hi samples for the points that are extrema (may be
        different sizes for each).

        """
        lo_samps = self._get_one_extrema_samps(
            lo_mask, extrema_inputs, self.pareto_lo, self.evaluator.batch_forward_ex_low)
        # lows must be flipped:
        if lo_samps is not None:
            lo_samps *= -1.0
        hi_samps = self._get_one_extrema_samps(
            hi_mask, extrema_inputs, self.pareto_hi, self.evaluator.batch_forward_ex_high)
        return lo_samps, hi_samps

    def _sample_and_replace_extremas(self, decoded, extrema_inputs, lo_mask, hi_mask):
        """Generate the Pareto output parameters for the low/high bins as
        needed, and sample actual outputs based on these parameters,
        and replace them at the corresponding locations in decoded.

        Arguments:

        decoded: Tensor[Float]: NBATCH, the normalized decoded values.

        extrema_inputs: Tensor: 1 x NBATCH x NSUBSET=1, the one extrema feature

        lo_mask: Tensor: NBATCH, which values are low extrema

        hi_mask: Tensor: NBATCH, which values are high extrema

        Returns:

        decoded: Tensor[Float]: NBATCH x 1, the normalized decoded
        values, but with values in extrema bins replaced by Pareto
        samples.

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

        Arguments:

        decoded: Tensor[Float]: NBATCH, the normalized decoded values.

        binned_coarses: Tensor: NBATCH, the bin idxs to decode

        extrema_inputs: 1 x NBATCH x NINPUTS=1, the one input per batch
        for the extrema forward function.

        Returns:

        decoded: Tensor[Float]: NBATCH x 1, the normalized decoded
        values, but with values in extrema bins replaced by Pareto
        samples.

        """
        if not self.extremas:
            return decoded
        # First, determine which points are extremas
        lo_mask = binned_coarses == 0
        hi_mask = binned_coarses == self.ncoarse - 1
        # Then, get samples for those points and replace points in decoded:
        decoded = self._sample_and_replace_extremas(decoded, extrema_inputs, lo_mask, hi_mask)
        return decoded

    def _decode_batch(self, binned_coarses, extrema_inputs, nmin, nmax):
        """Turn the binned coarses into (normalized) output values, using
        whatever bin decoding strategy.

        Arguments:

        binned_coarses: Tensor: NBATCH x 1, the bin idxs to decode

        extrema_inputs: NBATCH x NINPUTS=1, the one input per batch
        for the extrema forward function.

        nmin/nmax: each NBATCH x 1

        Returns:

        Tensor: NBATCH, the decoded (but still normalized) values, or
        None if not doing extremas.

        Tensor: NBATCH, the decoded+unnormed values.

        """
        binned_coarses = binned_coarses.reshape(-1)
        decoded = DigitizeUtils.torch_decode_values(
            binned_coarses, self.lower_cutoffs, self.upper_cutoffs)
        if self.extremas:
            decoded = self.__replace_extremas(decoded, binned_coarses, extrema_inputs)
        decoded = decoded.reshape(-1, 1)
        unnormed = self._unnormalize(decoded, nmin, nmax)
        if self.extremas:
            return decoded, unnormed
        return None, unnormed

    def _get_binned(self, coarse_inputs):
        """Helper to get the coarse binned values, moved to this separate
        function purely to reduce complexity of build_outseqs (and for
        parallelism with JointGenerator and TripleGenerator).

        Arguments:

        coarse_inputs: Tensor[1 x NBATCH x NSUBSET], the subset of
        coarse features, for each element of the batch, for the latest
        element of the sequence.

        Returns:

        binned_coarses: Tensor: NBATCH x 1, the coarse idxs

        """
        coarse_outputs = self.evaluator.batch_forward_coarse(coarse_inputs)
        coarse_logits = coarse_outputs[-1]
        # Find the prob of each class according to logits:
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        try:
            binned_coarses = torch.multinomial(coarse_probs, 1)
        except RuntimeError as multi_err:
            # Raise our own error, which we can catch as sign
            # these parameters are not good:
            logger.exception("Multinomial: %s -> %s -> %s",
                             coarse_inputs, coarse_outputs, coarse_probs)
            raise ProbError("Error with torch.multinomial") from multi_err
        return binned_coarses

    def _build_outseqs(self, last_inputs, originals):
        """Starting with the last set of inputs from the conditioning window
        (and a network that is already conditioned), auto-regressively
        call the network to build output sequences for all sequences
        in the batch.

        Arguments:

        last_inputs: 1 x NBATCH x NFEATS, final inputs.

        originals: NSEQ+2 x NBATCH x 1

        -- In all cases, NSEQ is length of conditioning window.

        Returns:

        outseqs, torch.tensor(Float), of shape NGEN x NBATCH x 1,
        where NBATCH is perhaps NERIES*NSAMPLES if from batch_predictor

        """
        # Precalculate nmin/nmax, parallelized over all seqs:
        csize = len(originals) - 2
        nmin, nmax = DataUtils.get_nmin_nmax(csize, self.nsize, originals)
        all_gen = []
        for _ in range(self.ngen):
            # Divide the last input into coarse/extrema parts, as needed:
            coarse_inputs, extrema_inputs = BinTensorMaker.extract_coarse_extremas(
                last_inputs, extremas=self.extremas)
            binned_coarses = self._get_binned(coarse_inputs)
            normed, generated = self._decode_batch(binned_coarses, extrema_inputs, nmin, nmax)
            all_gen.append(generated)
            pcoarse = torch.nn.functional.one_hot(
                binned_coarses.reshape(-1), num_classes=self.ncoarse).to(INPUT_DTYPE)
            last_inputs = BinTensorMaker.reset_ccoarse_inputs(
                last_inputs, pcoarse, normed=normed)
        outseqs = torch.cat(all_gen, dim=1)
        return outseqs

    def __call__(self, batch):
        """Generate self.ngen points forward on the given batch of data.

        Arguments:

        batch: Dict[ExampleKeys] -> Tensors[Float], where tensors are
        in shape of Tensor[NSEQ x NSERIES*NSAMPLES x {}] (last dim
        depending on key).

        Returns:

        outseqs, torch.tensor(Float), of shape NBATCH x NGEN (just 2
        dim), all the samples as a batch.

        """
        inputs = batch[ExampleKeys.INPUT]
        originals = batch[ExampleKeys.ORIGINALS]
        coarses = batch[ExampleKeys.COARSES]
        batch_size = inputs.shape[1]
        self.evaluator.run_init_hidden(batch_size)
        # net already in eval mode fr get_lstm_evaluator, but also do this:
        with torch.no_grad():
            # However your evaluator is configured, temporarily
            # disable re-initing hidden states:
            do_hid_state = self.evaluator.get_do_init_hidden()
            self.evaluator.set_do_init_hidden(False)
            # Do fwd pass once - without the targets (this moves
            # outputs and originals to the device) - but not including
            # last element (this is just to update hidden state):
            _, _, originals, coarses = self.evaluator.batch_forward_outputs(
                inputs[:-1], None, originals, coarses)
            last_inputs = inputs[-1:]
            outseqs = self._build_outseqs(last_inputs, originals)
            # -- These are also NGEN x NBATCH x 1
            self.evaluator.set_do_init_hidden(do_hid_state)
            # These are also NSERIESxNSAMPLES x NGEN:
            return outseqs
