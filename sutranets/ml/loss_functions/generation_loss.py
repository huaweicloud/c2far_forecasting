"""Custom loss functions related to loss on multi-step-ahead
*generated* values.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/generation_loss.py
import torch
from sutranets.ml.loss_functions.constants import \
    IGNORE_INDEX, PointLossFunc, GenStrategy
from sutranets.ml.loss_functions.guesser import Guesser
from sutranets.ml.loss_functions.transform_loss import TransformLoss
COMBO_LOSS_FLAG = -100


class GenerationGuesser(Guesser):
    """Guesser used with GenerationLoss.  For GenerationLoss, the values
    are already in the original domain, so we don't need to unmap.  We
    either output the p50s, if computing MSE, MAE, etc., or the
    lows/highs, if we're computing coverage.

    """
    def __init__(self, strategy):
        super().__init__(strategy)
        if self.strategy not in [GenStrategy.GEN_P50, GenStrategy.GEN_COV,
                                 GenStrategy.GEN_WQL]:
            raise RuntimeError(
                f"Unsupported strategy {self.strategy.value} for generation")

    def get_guesses(self, outputs, originals, *, values=None):
        """For GenerationEvaluators, the outputs should be either NSEQ x
        NBATCH x 3, with one for low, p50, high, in that order, or
        NSEQ x NBATCH x 9, with one for each quantile from 0.1 to 0.9.

        """
        outputs2 = outputs.reshape(-1, outputs.shape[-1])
        if outputs2.shape[1] == 3:
            p50_dim = 1
            last_dim = 2
        elif outputs2.shape[1] == 9:
            p50_dim = 4
            last_dim = 8
        else:
            raise RuntimeError(f"Outputs {outputs} not in low/p50s/highs format")
        if self.strategy == GenStrategy.GEN_P50:
            guesses2 = outputs2[:, p50_dim]
        elif self.strategy == GenStrategy.GEN_COV:
            guesses2 = outputs2[:, 0], outputs2[:, last_dim]
        elif self.strategy == GenStrategy.GEN_WQL:
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
    def __init__(self, pt_loss_func, strategy, target_horizon,
                 *, targ_level=None, nsub_series=None, ignore_index=IGNORE_INDEX):
        guesser = GenerationGuesser(strategy)
        super().__init__(pt_loss_func, guesser,
                         target_horizon=target_horizon,
                         ignore_index=ignore_index)
        self.nsub_series = nsub_series
        self.targ_level = targ_level
        self.orig_start = self.orig_end = self.targ_start = self.targ_end = self.gsize = None
        self.__validate(strategy, pt_loss_func)

    @staticmethod
    def __validate(strategy, pt_loss_func):
        do_cov_strategy = strategy in [GenStrategy.GEN_COV]
        do_cov_loss = pt_loss_func in [PointLossFunc.COVERAGE,
                                       PointLossFunc.COV_WIDTH]
        if do_cov_strategy != do_cov_loss:
            raise RuntimeError("Mismatching strategy/loss for coverage")

    def __str__(self):
        if self.targ_level is not None:
            extra_info = f".agg_targ={self.targ_level},g={self.gsize}"
        elif self.nsub_series is not None:
            extra_info = f".nseries={self.nsub_series}.targ={self.targ_level}"
        else:
            extra_info = ""
        label = f"<GenerationLoss.{self._get_info()}{extra_info}>"
        return label

    def __repr__(self):
        return self.__str__()

    def __call__(self, outputs, targets, originals, coarses=None):
        """Converts our outputs to guesses, then computes generation loss.

        """
        if self.orig_start is not None:
            if self.targ_level is not None:
                outputs = outputs[self.targ_level]
            originals = originals[self.orig_start:self.orig_end]
            targets = targets[self.targ_start:self.targ_end]
            if coarses is not None:
                coarses = coarses[self.targ_start:self.targ_end]
        elif self.nsub_series is not None:
            if self.targ_level != COMBO_LOSS_FLAG:
                outputs = outputs[self.targ_level]
                originals = originals[self.targ_level]
                targets = targets[self.targ_level]
                if coarses is not None:
                    coarses = coarses[self.targ_level]
            else:
                outputs = torch.cat(outputs, dim=0)
                targets = torch.cat(targets.get_lst_of_tensors(), dim=0)
                if coarses is not None:
                    coarses = torch.cat(coarses.get_lst_of_tensors(), dim=0)
                all_origs = originals.get_lst_of_tensors()
                originals = torch.cat([all_origs[0]] + [o[2:] for o in all_origs[1:]], dim=0)
        return super().__call__(outputs, targets, originals, coarses)
