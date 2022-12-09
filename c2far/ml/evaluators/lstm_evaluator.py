"""Subclasses of Evaluator that use an LSTM to do 1-step-ahead logit
prediction.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/evaluators/lstm_evaluator.py

from c2far.ml.evaluators.evaluator import Evaluator


class LSTMEvaluator(Evaluator):
    """Evaluator that uses an LSTM to do batch_forward (and
    batch_forward_coarse) internally.

    """
    LABEL = "LSTMEvaluator"

    def __init__(self, net, device, do_init_hidden):
        """Pass in the net and the device (torch.device, not the string).
        Note also we can pass do_init_hidden=False if we want to
        preserve the hidden state across calls to make_outputs (which
        we will want in generation mode) - but in testing, we do want
        to re-init it, because we do a whole sequence each time.

        Arguments:

        net: the neural network itself, on which we can call
        'forward()'

        device: torch.device (not the string), onto which we place the
        tensors.

        do_init_hidden: Boolean, if True, we init the hidden state
        every time we make outputs (i.e., run forward pass).

        """
        super().__init__(device)
        self.net = net
        self.do_init_hidden = do_init_hidden

    def set_do_init_hidden(self, do_init_hidden):
        """Setting for whether to init the hidden state"""
        self.do_init_hidden = do_init_hidden

    def get_do_init_hidden(self):
        """Getter for whether do_init_hidden is on/off."""
        return self.do_init_hidden

    def _make_outputs(self, inputs, targets, originals):
        """Override the make_outputs part with our LSTM-specific forward pass.

        """
        if self.do_init_hidden:
            # The default batch dim for LSTMs:
            batch_size = inputs.shape[1]
            self.run_init_hidden(batch_size)
        outputs = self.net(inputs)
        return outputs

    def run_init_hidden(self, batch_size):
        """Public method that lets us init the hidden state when needed - for
        example, from within a Generator.

        """
        self.net.run_init_hidden(device=self.device, batch_size=batch_size)

    def _batch_forward_func(self, inputs, fwd_func):
        """Helper to do batch_forward on the inputs using the given function,
        e.g. we can use either a coarse-specific or fine-specific
        forward function here, but the rest of the logic is the same.
        NOTE: if you call this function with init_hidden on, it raises
        an error - I don't expect this to ever be needed: if we are
        training in parallel, we can use the regular LSTMEvaluator and
        init all the hiddens at once.  If doing generation
        sequentially, we don't init_hidden at all.

        Arguments:

        inputs: Tensor[NSEQ x NBATCH x NSUBSET]

        fwd_func: Function or Callable, how to do the forward pass.

        Returns:

        outputs: Tensor[NSEQ x NBATCH x NSUBSET_TARGS]
        (e.g. NSUBSET_TARGS in {NCOARSE, NFINE})

        """
        inputs = inputs.to(self.device)
        # Note we don't init the hidden state for these - EVER.  They
        # are only for generation.  We used to have a check in here to
        # make sure self.do_init_hidden is not engaged, but that's too
        # expensive to check every time.
        outputs = fwd_func(inputs)
        return outputs

    def batch_forward_coarse(self, inputs):
        """Do batch_forward just on coarse inputs, get coarse-only outputs.

        """
        return self._batch_forward_func(inputs, self.net.coarse_forward)

    def batch_forward_ex_low(self, inputs):
        """Do batch_forward just on the extrema low inputs, get the
        extrema_low outputs.

        """
        return self._batch_forward_func(inputs, self.net.ex_mlp_low)

    def batch_forward_ex_high(self, inputs):
        """Do batch_forward just on the extrema high inputs, get the
        extrema_high outputs.

        """
        return self._batch_forward_func(inputs, self.net.ex_mlp_high)


class JointLSTMEvaluator(LSTMEvaluator):
    """An LSTMEvaluator with special coarse/fine forward methods."""
    LABEL = "JointLSTMEvaluator"

    def batch_forward_fine(self, inputs):
        """Do batch_forward just on *fine* inputs, get fine-only outputs.

        """
        return self._batch_forward_func(inputs, self.net.fine_forward)


class TripleLSTMEvaluator(JointLSTMEvaluator):
    """A JointLSTMEvaluator with special coarse/fine/fine2 forward
    methods.

    """
    LABEL = "TripleLSTMEvaluator"

    def batch_forward_fine2(self, inputs):
        """Do batch_forward just on *fine2* inputs, get fine2-only outputs.

        """
        return self._batch_forward_func(inputs, self.net.fine2_forward)
