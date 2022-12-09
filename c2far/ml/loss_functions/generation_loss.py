"""Custom loss functions related to loss on multi-step-ahead
*generated* values.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/generation_loss.py

from c2far.ml.loss_functions.constants import \
    IGNORE_INDEX, PointLossFunc, GenStrategy
from c2far.ml.loss_functions.guesser import Guesser
from c2far.ml.loss_functions.transform_loss import TransformLoss


class GenerationGuesser(Guesser):
    """Guesser used with GenerationLoss.  For GenerationLoss, the values
    are already in the original domain, so we don't need to unmap.  We
    either output the p50s, if computing MSE, MAE, etc., or the
    lows/highs, if we're computing coverage.

    """
    def __init__(self, strategy):
        """Record the strategy, and do some checks now, early on."""
        super().__init__(strategy)
        if self.strategy not in [GenStrategy.GEN_P50, GenStrategy.GEN_COV,
                                 GenStrategy.GEN_WQL]:
            raise RuntimeError(
                f"Unsupported strategy {self.strategy.value} for generation")

    def get_guesses(self, outputs, originals, *, values=None):
        """For GenerationEvaluators, the outputs should be either NSEQ x
        NBATCH x 3, with one for low, p50, high, in that order, or
        NSEQ x NBATCH x 9, with one for each quantile from 0.1 to 0.9.

        originals and values are apparently unused here

        This function uses our GenStrategy to determine what guesses
        to make, but it looks at the shape of outputs in order to
        determine how to unpack those guesses from the outputs object,
        i.e., whether there are 3 percentiles or 9 of them in outputs.

        """
        outputs2 = outputs.reshape(-1, outputs.shape[-1])
        if outputs2.shape[1] == 3:
            p50_dim = 1
            last_dim = 2
        elif outputs2.shape[1] == 9:
            # 0:0.1, 1:0.2, 2:0.3, 3:0.4, 4:0.5, 5:0.6, 6:0.7, 7:0.8, 8:0.9:
            p50_dim = 4
            last_dim = 8
        else:
            raise RuntimeError(f"Outputs {outputs} not in low/p50s/highs format")
        if self.strategy == GenStrategy.GEN_P50:
            guesses2 = outputs2[:, p50_dim]
        elif self.strategy == GenStrategy.GEN_COV:
            guesses2 = outputs2[:, 0], outputs2[:, last_dim]
        elif self.strategy == GenStrategy.GEN_WQL:
            # Get each qtile separately - if an IndexError here, you
            # should probably be using confidence=-1 in the predictor:
            guesses2 = [outputs2[:, i] for i in range(9)]
        else:
            raise RuntimeError(
                f"Unsupported strategy {self.strategy.value} for generation")
        return guesses2


class GenerationLoss(TransformLoss):
    """Loss function that operates on the output of a generation-based
    evaluator (e.g., GenerationEvaluator): pulling out the right
    values as "guesses" depending on the loss function.

    """
    def __init__(self, pt_loss_func, strategy, *,
                 ignore_index=IGNORE_INDEX):
        """Arguments:

        pt_loss_func: PointLossFunc (Enum), the "point" loss: whether
        we should use MSE, MASE, MAPE, etc.

        strategy: GenStrategy (Enum), whether we should pick the most
        likely output (MLE), or the P50, etc.

        ignore_index: Int, which index should be ignored in the loss
        functions.

        """
        guesser = GenerationGuesser(strategy)
        super().__init__(pt_loss_func, guesser,
                         ignore_index=ignore_index)
        self.__validate(strategy, pt_loss_func)

    @staticmethod
    def __validate(strategy, pt_loss_func):
        """Make sure the loss function and strategy combo are valid."""
        do_cov_strategy = strategy in [GenStrategy.GEN_COV]
        do_cov_loss = pt_loss_func in [PointLossFunc.COVERAGE,
                                       PointLossFunc.COV_WIDTH]
        if do_cov_strategy != do_cov_loss:
            raise RuntimeError("Mismatching strategy/loss for coverage")

    def __str__(self):
        """Be more explicit when we're outputting the name."""
        label = f"<GenerationLoss.{self._get_info()}>"
        return label
