"""Utilities related to making criterions for training and testing.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/make_criterions.py

from c2far.ml.constants import ConfidenceFlags
from c2far.ml.loss_functions.coarse_cross_entropy_loss import CoarseCrossEntropyLoss
from c2far.ml.loss_functions.constants import IGNORE_INDEX, PointLossFunc, GenStrategy
from c2far.ml.loss_functions.joint_cross_entropy_loss import JointCrossEntropyLoss
from c2far.ml.loss_functions.generation_loss import GenerationLoss
from c2far.ml.loss_functions.gauss_loss import GaussLoss
from c2far.ml.loss_functions.triple_cross_entropy_loss import TripleCrossEntropyLoss


class MakeCriterions():
    """Class with static utility methods for making test criterions."""

    @classmethod
    def make_next_step_train(cls, args, coarse_cutoffs):
        """Given the arguments, make the losses of interest for training.
        Note there's a lot of overlap between this function and
        make_next_step_test, but since sometimes we turn off some
        losses in either one or the other, we keep them separate.

        """
        if coarse_cutoffs is not None:
            test_criterions = []
            if args.nfine_bins is None:
                ce_loss = CoarseCrossEntropyLoss(
                    coarse_cutoffs, extremas=args.extremas, ignore_index=IGNORE_INDEX)
            elif args.nfine2_bins is None:
                ce_loss = JointCrossEntropyLoss(
                    coarse_cutoffs, args.nfine_bins, extremas=args.extremas, ignore_index=IGNORE_INDEX)
            else:
                ce_loss = TripleCrossEntropyLoss(
                    coarse_cutoffs, args.nfine_bins, args.nfine2_bins,
                    extremas=args.extremas, ignore_index=IGNORE_INDEX)
            train_criterions = [ce_loss]
            if not test_criterions:
                test_criterions = [ce_loss]
        else:
            train_criterions = [GaussLoss()]
            test_criterions = [GaussLoss()]
        return train_criterions, test_criterions

    @classmethod
    def make_next_step_test(cls, args, coarse_cutoffs):
        """Helper function to encapsulate making all the criterions we desire
        for *testing*.

        """
        if coarse_cutoffs is not None:
            if args.nfine_bins is None:
                criterions = [CoarseCrossEntropyLoss(
                    coarse_cutoffs, extremas=args.extremas, ignore_index=IGNORE_INDEX)]
            elif args.nfine2_bins is None:
                criterions = [JointCrossEntropyLoss(
                    coarse_cutoffs, args.nfine_bins, extremas=args.extremas, ignore_index=IGNORE_INDEX)]
            else:
                criterions = [TripleCrossEntropyLoss(
                    coarse_cutoffs, args.nfine_bins, args.nfine2_bins,
                    extremas=args.extremas, ignore_index=IGNORE_INDEX)]
        else:
            criterions = [GaussLoss()]
        return criterions

    @classmethod
    def create_gen_mae(cls):
        """Helper to create just the multi-step MAE

        """
        mae_loss = GenerationLoss(PointLossFunc.MAE, GenStrategy.GEN_P50)
        return mae_loss

    @classmethod
    def create_gen_criterions(cls, confidence):
        """Helper to create all the multi-step criterions of interest.

        Arguments:

        confidence, Int, % for confidence bands.  If -1, we do all
        quantiles between 0.1 and 0.9, and also add WQL as a loss.

        """
        mae_loss = cls.create_gen_mae()
        ndnorm_loss = GenerationLoss(PointLossFunc.ND_NORMALIZER, GenStrategy.GEN_P50)
        cov_loss = GenerationLoss(PointLossFunc.COVERAGE, GenStrategy.GEN_COV)
        wid_loss = GenerationLoss(PointLossFunc.COV_WIDTH, GenStrategy.GEN_COV)
        criterions = [mae_loss, ndnorm_loss, cov_loss, wid_loss]
        if confidence == ConfidenceFlags.WQL_QTILES.value:
            wql_loss = GenerationLoss(PointLossFunc.WQL, GenStrategy.GEN_WQL)
            criterions.append(wql_loss)
        return criterions
