"""Subclasses of Evaluator that use an LSTM to do 1-step-ahead logit
prediction.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/evaluators/lstm_evaluator.py
from sutranets.ml.evaluators.evaluator import Evaluator


class LSTMEvaluator(Evaluator):
    LABEL = "LSTMEvaluator"

    def __init__(self, net, device, do_init_hidden):
        super().__init__(device)
        self.net = net
        self.do_init_hidden = do_init_hidden

    def set_do_init_hidden(self, do_init_hidden):
        self.do_init_hidden = do_init_hidden

    def get_do_init_hidden(self):
        return self.do_init_hidden

    def _make_outputs(self, inputs, targets, originals):
        """Override with our LSTM-specific forward pass.

        """
        if self.do_init_hidden:
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
        """Helper to do batch_forward on the inputs using the given function.

        """
        inputs = inputs.to(self.device)
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
