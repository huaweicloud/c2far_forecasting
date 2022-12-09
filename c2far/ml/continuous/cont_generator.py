"""Use a trained continuous LSTM model to generate sequences of the
next ngen values, priming on an existing set of conditioning of data
(each csize in length).

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/continuous/cont_generator.py

import torch
from c2far.dataprep.data_utils import DataUtils
from c2far.ml.constants import ExampleKeys, ProbError


class ContGenerator():
    """Class to generate a sequence of values given a trained *continuous*
    LSTM model.  That is, we expect the evaluator to take in encodings
    of single real values, and output mean/variance values for a
    Gaussian.

    """
    def __init__(self, device, evaluator, nsize, ngen):
        """Initialize with the continuous LSTM evaluator and the other things
        we need for generation.

        Arguments:

        device: String, e.g. "cuda:0" or "cpu"

        evaluator: Evaluator, e.g., LSTMEvaluator: what we use to call
        `batch_forward_outputs()`.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        ngen: Int, length of generation window

        """
        self.device = device
        self.evaluator = evaluator
        self.nsize = nsize
        self.ngen = ngen
        # Re-used repeatedly below:
        self.softplus = torch.nn.Softplus()

    def _build_outseqs(self, mean_vars, originals):
        """Starting with the given set of output mean_vars, auto-regressively
        call the LSTM to build output seqs for all sequences in the batch.

        We need 'originals' for repeated computation of nmin/nmax (for
        unnormalizing).

        Arguments:

        mean_vars: NBATCH x 2, starting output mean/vars to regress on

        originals: NSEQ+2 x NBATCH x 1

        -- In all cases, NSEQ is length of conditioning window.

        Returns:

        outseqs, torch.tensor(Float), of shape NGEN x NBATCH x 1,
        where NBATCH is perhaps NERIES*NSAMPLES if from batch_predictor

        """
        # Precalculate nmin/nmax, parallelized over all seqs:
        csize = len(originals) - 2  # DataUtils expects csize given
        nmin, nmax = DataUtils.get_nmin_nmax(csize, self.nsize, originals)
        all_gen = []
        for _ in range(self.ngen):
            # Note use of (singleton) array indexing to maintain shape:
            means = mean_vars[:, [0]]
            variances = self.softplus(mean_vars[:, [1]])
            stddevs = torch.sqrt(variances)
            try:
                samps = torch.normal(means, stddevs)
            except RuntimeError as normal_err:
                # Raise our own error, which we can catch as sign
                # these parameters are not good:
                raise ProbError("Error with torch.normal") from normal_err
            all_gen.append(samps)
            inputs = samps.reshape(1, -1, 1)
            mean_vars, _, _ = self.evaluator.batch_forward_outputs(
                inputs, None, None)
            mean_vars = mean_vars[0]
        outnormed = torch.cat(all_gen, dim=1)
        outseqs = outnormed * (nmax - nmin) + nmin
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
        batch_size = inputs.shape[1]
        self.evaluator.run_init_hidden(batch_size)
        # net already in eval mode fr get_lstm_evaluator, but also do this:
        with torch.no_grad():
            # However your evaluator is configured, temporarily
            # disable re-initing hidden states:
            do_hid_state = self.evaluator.get_do_init_hidden()
            self.evaluator.set_do_init_hidden(False)
            # Do forward pass once - without the targets (this moves
            # outputs and originals to device, primes hidden state):
            outputs, _, originals, _ = self.evaluator.batch_forward_outputs(
                    inputs, None, originals, None)
            # The last element of seq gives mean/var guesses for next
            # value in each batch dim (for batch_predictor, NBATCH =
            # NSERIES x NSAMPLES, and guesses stacked in series order):
            mean_vars = outputs[-1]
            outseqs = self._build_outseqs(mean_vars, originals)
            # -- These are also NGEN x NBATCH x 1
            # Now reset this to whatever was being done before:
            self.evaluator.set_do_init_hidden(do_hid_state)
            # These are also NSERIESxNSAMPLES x NGEN:
            return outseqs
