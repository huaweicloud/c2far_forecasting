"""Utilities that can be shared by different ML code for creating the datasets.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/dataset_utils.py
import logging
from torch.utils.data import DataLoader
from sutranets.ml.bin_dataset import BinDataset
from sutranets.ml.collate_utils import CollateUtils
from sutranets.ml.fc_dataset import DatasetArgs
from sutranets.ml.multivariate.hf_multivariate_dataset import HFMultivariateDataset
logger = logging.getLogger("sutranets.ml.dataset_utils")


class DatasetUtils():
    """Static utility methods to help create the dataloaders"""

    @staticmethod
    def __make_dataset(trace_paths, offs, bin_fargs, fargs, dargs,
                       nsub_series, sub_csize, mv_no_prevs,
                       mv_backfill):
        path_args = [trace_paths, offs]
        if nsub_series is not None:
            dset = HFMultivariateDataset(*path_args, **bin_fargs, **fargs, dataset_args=dargs,
                                         sub_csize=sub_csize, nsub_series=nsub_series, mv_no_prevs=mv_no_prevs,
                                         mv_backfill=mv_backfill)
        else:
            dset = BinDataset(*path_args, **bin_fargs, **fargs, dataset_args=dargs)
        return dset

    @classmethod
    def make_dataloaders(cls, args, coarse_cutoffs, *,
                         include_meta=False, make_train_set=True,
                         make_test_set=True, include_train_origs=False,
                         disable_randomize_train_start=False,
                         disable_checkpoints_train=False):
        if not (make_train_set or make_test_set):
            raise RuntimeError("Must make either the train or test set")
        shared_dargs = {"nfine_bins": args.nfine_bins, "nfine2_bins": args.nfine2_bins, "include_meta": include_meta,
                        "extremas": args.extremas, "high_period_s": args.sample_period_s}
        shared_fargs = {"csize": args.csize, "gsize": args.gsize,
                        "trivial_min_nonzero_frac": args.trivial_min_nonzero_frac}
        if args.nsub_series is None:
            shared_fargs["nsize"] = args.nsize
        shared_bin_fargs = {"coarse_cutoffs": coarse_cutoffs,
                            "coarse_low": args.coarse_low, "coarse_high": args.coarse_high,
                            "lagfeat_period": args.lagfeat_period}
        shared_largs = {"collate_fn": CollateUtils.batching_collator,
                        "num_workers": args.num_workers, "pin_memory": False}
        multivar_params = {"nsub_series": args.nsub_series, "sub_csize": args.sub_csize,
                           "mv_no_prevs": args.mv_no_prevs, "mv_backfill": args.mv_backfill}
        if make_test_set:
            do_include_coarses = not args.dont_run_1step_eval
            do_targets_only = args.dont_run_1step_eval
            test_dargs = DatasetArgs(ncheckpoint=args.ntest_checkpoint, truncate_pt_s=args.test_truncate_pt_s,
                                     loss_start_pt_s=args.test_loss_start_pt_s, loss_end_pt_s=args.test_loss_end_pt_s,
                                     cache_set=args.cache_testset, include_coarses=do_include_coarses,
                                     targets_only=do_targets_only, include_originals=True, **shared_dargs)
            testset = cls.__make_dataset(args.test_trace_paths, args.test_offs,
                                         shared_bin_fargs, shared_fargs, test_dargs, **multivar_params)
            testloader = DataLoader(testset, shuffle=False, batch_size=args.test_batch_size, **shared_largs)
        else:
            testloader = None
        if make_train_set:
            if disable_randomize_train_start:
                do_randomize_start = False
            else:
                do_randomize_start = args.randomize_start
            if disable_checkpoints_train:
                ncheckpoint = -1
            else:
                ncheckpoint = args.ntrain_checkpoint
            train_dargs = DatasetArgs(ncheckpoint=ncheckpoint, randomize_start=do_randomize_start,
                                      truncate_pt_s=args.train_truncate_pt_s,
                                      loss_start_pt_s=args.train_loss_start_pt_s,
                                      loss_end_pt_s=args.train_loss_end_pt_s,
                                      include_originals=include_train_origs,
                                      bin_dropout=args.bin_dropout,
                                      **shared_dargs)
            trainset = cls.__make_dataset(args.train_trace_paths, args.train_offs,
                                          shared_bin_fargs, shared_fargs, train_dargs, **multivar_params)
            trainloader = DataLoader(trainset, shuffle=True, batch_size=args.train_batch_size, **shared_largs)
        else:
            trainloader = None
        return trainloader, testloader
