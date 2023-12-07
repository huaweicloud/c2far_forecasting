"""Subclasses of LSTMEvaluator for multi-series data.  Internally can
either call the net on the full set of inputs (via its parent), or
provide a sub-net (sub-evaluator) for each input.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/evaluators/multi_lstm_evaluator.py
from sutranets.ml.evaluators.lstm_evaluator import LSTMEvaluator


class MultiLSTMEvaluator(LSTMEvaluator):
    LABEL = "MultiLSTMEvaluator"

    def __init__(self, net, device, do_init_hidden, lstm_eval_cls):
        super().__init__(net, device, do_init_hidden)
        self.lstm_eval_cls = lstm_eval_cls

    def yield_sub_evaluators(self):
        """Iterator over evaluators corresponding to each individual
        aggregation level.

        """
        for subnet in self.net.yield_models():
            sub_lstm_eval = self.lstm_eval_cls(subnet, self.device, self.do_init_hidden)
            yield sub_lstm_eval
