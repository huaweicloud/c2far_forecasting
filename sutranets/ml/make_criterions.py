"""Utilities related to making criterions for training and testing.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/make_criterions.py
from sutranets.ml.constants import ConfidenceFlags
from sutranets.ml.loss_functions.coarse_cross_entropy_loss import CoarseCrossEntropyLoss
from sutranets.ml.loss_functions.constants import IGNORE_INDEX, PointLossFunc, GenStrategy
from sutranets.ml.loss_functions.joint_cross_entropy_loss import JointCrossEntropyLoss
from sutranets.ml.loss_functions.generation_loss import GenerationLoss, COMBO_LOSS_FLAG
from sutranets.ml.loss_functions.multivariate_loss import MultivariateLoss
from sutranets.ml.loss_functions.triple_cross_entropy_loss import TripleCrossEntropyLoss
HORIZ_SPACING_THRESH = 500
COARSE_HORIZ_SPACING = 5


class MakeCriterions():
    """Class with static utility methods for making test criterions."""

    @staticmethod
    def __make_one_ce_loss(args, coarse_cutoffs):
        """Make a single ce_loss based on the given arguments."""
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
        return ce_loss

    @classmethod
    def make_next_step_train(cls, args, coarse_cutoffs):
        """Make the losses of interest for training.

        """
        if coarse_cutoffs is not None:
            test_criterions = []
            if args.nsub_series is not None:
                losses = [cls.__make_one_ce_loss(args, coarse_cutoffs) for _ in range(args.nsub_series)]
                ce_loss = MultivariateLoss(losses)
            else:
                ce_loss = cls.__make_one_ce_loss(args, coarse_cutoffs)
            train_criterions = [ce_loss]
            if not test_criterions:
                test_criterions = [ce_loss]
        return train_criterions, test_criterions

    @classmethod
    def make_all_losses(cls, args, coarse_cutoffs):
        """Make all of the losses, in one shot (used in plotting, etc.)

        """
        gauss_loss = coarse_loss = fine_loss = fine2_loss = None
        if coarse_cutoffs is not None:
            coarse_loss = CoarseCrossEntropyLoss(
                coarse_cutoffs, extremas=args.extremas, ignore_index=IGNORE_INDEX)
        if args.nfine_bins is not None:
            fine_loss = JointCrossEntropyLoss(
                coarse_cutoffs, args.nfine_bins, extremas=args.extremas, ignore_index=IGNORE_INDEX)
        if args.nfine2_bins is not None:
            fine2_loss = TripleCrossEntropyLoss(
                coarse_cutoffs, args.nfine_bins, args.nfine2_bins,
                extremas=args.extremas, ignore_index=IGNORE_INDEX)
        return gauss_loss, coarse_loss, fine_loss, fine2_loss

    @classmethod
    def make_next_step_test(cls, args, coarse_cutoffs):
        """Make all the criterions we desire for *testing*.

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
        return criterions

    @classmethod
    def create_gen_mae(cls, targ_level, nsub_series):
        """Create just the multi-step MAE.

        """
        mae_loss = GenerationLoss(PointLossFunc.MAE, GenStrategy.GEN_P50, None, targ_level=targ_level,
                                  nsub_series=nsub_series)
        return mae_loss

    @staticmethod
    def __append_horiz_criterions(criterions, gsize, kw_parms):
        """Append loss functions for specific horizons.

        """
        if gsize is None:
            raise RuntimeError("Need gsize to do horizon criterions.")
        if gsize > HORIZ_SPACING_THRESH:
            hspacing = COARSE_HORIZ_SPACING
        else:
            hspacing = 1
        for i in range(0, gsize, hspacing):
            mae_h = GenerationLoss(
                PointLossFunc.MAE, GenStrategy.GEN_P50, target_horizon=(i, gsize), **kw_parms)
            criterions.append(mae_h)

    @classmethod
    def __gen_criter_helper(cls, confidence, *, do_horizs, gsize,
                            targ_level=None, nsub_series=None):
        """Create all the multi-step criterions of interest, at a particular
        targ_level (or at the only level, if targ_level is None).

        """
        kw_parms = {"targ_level": targ_level, "nsub_series": nsub_series}
        mae_loss = cls.create_gen_mae(**kw_parms)
        ndnorm_loss = GenerationLoss(PointLossFunc.ND_NORMALIZER, GenStrategy.GEN_P50, None, **kw_parms)
        smape_loss = GenerationLoss(PointLossFunc.SMAPE, GenStrategy.GEN_P50, None, **kw_parms)
        me_loss = GenerationLoss(PointLossFunc.MPE, GenStrategy.GEN_P50, None, **kw_parms)
        mse_loss = GenerationLoss(PointLossFunc.MSE, GenStrategy.GEN_P50, None, **kw_parms)
        cov_loss = GenerationLoss(PointLossFunc.COVERAGE, GenStrategy.GEN_COV, None, **kw_parms)
        wid_loss = GenerationLoss(PointLossFunc.COV_WIDTH, GenStrategy.GEN_COV, None, **kw_parms)
        criterions = [mae_loss, ndnorm_loss, smape_loss, me_loss, mse_loss, cov_loss, wid_loss]
        if confidence == ConfidenceFlags.WQL_QTILES.value:
            wql_loss = GenerationLoss(PointLossFunc.WQL, GenStrategy.GEN_WQL, None, **kw_parms)
            criterions.append(wql_loss)
        if do_horizs:
            cls.__append_horiz_criterions(criterions, gsize, kw_parms)
        return criterions

    @classmethod
    def create_gen_criterions(cls, confidence, *, do_horizs, gsize,
                              nsub_series):
        """Create all the multi-step criterions of interest.  The first one is
        usually the "target" one in tuning, e.g., it's the aggregate
        loss for multi-freq, or the HF loss in multi-freq.

        """
        crits = []
        if nsub_series is not None:
            sub_gsize = gsize // nsub_series
            crits += cls.__gen_criter_helper(confidence, do_horizs=do_horizs, gsize=sub_gsize,
                                             targ_level=COMBO_LOSS_FLAG, nsub_series=nsub_series)
            for lev in range(nsub_series):
                crits += cls.__gen_criter_helper(confidence, do_horizs=do_horizs, gsize=sub_gsize,
                                                 targ_level=lev, nsub_series=nsub_series)
        else:
            crits += cls.__gen_criter_helper(confidence, do_horizs=do_horizs, gsize=gsize)
        return crits
