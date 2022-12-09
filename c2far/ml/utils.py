"""Utilities that can be shared by different ML code for c2faring.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/utils.py

from glob import glob
import logging
import torch
from torch.utils.data import DataLoader
from c2far.ml.batch_predictor import BatchPredictor
from c2far.ml.collate_utils import CollateUtils
from c2far.ml.evaluators.lstm_evaluator import \
    LSTMEvaluator, JointLSTMEvaluator, TripleLSTMEvaluator
from c2far.ml.bin_dataset import BinDataset
from c2far.ml.continuous.cont_dataset import ContDataset
from c2far.ml.fc_dataset import DatasetArgs
from c2far.ml.nets.fc_lstm import FCLSTM
from c2far.ml.nets.joint_lstm import JointLSTM
from c2far.ml.nets.triple_lstm import TripleLSTM


logger = logging.getLogger("c2far.ml.utils")


def load_lstm_model(nfine_bins, nfine2_bins, lstm_path, mydevice):
    """Helper to take care of creating the appropriate PyTorch model given
    the args.

    """
    if nfine_bins is None:  # this is also the one for continuous:
        return FCLSTM.create_from_path(lstm_path, device=mydevice)
    if nfine2_bins is None:
        return JointLSTM.create_from_path(lstm_path, device=mydevice)
    return TripleLSTM.create_from_path(lstm_path, device=mydevice)


def get_lstm_evaluator(device_str, lstm_model, *other_cls_args,
                       nfine_bins=None, nfine2_bins=None, **other_cls_kwargs):
    """Basically, this is a generic function that will set up your net
    (read from disk, configure the device, etc.) from some generic
    argparser args, as well initialize the LSTMEvaluator that will be
    used to get 1-step-ahead predictions from the net.

    """
    device = torch.device(device_str)
    try:
        lstm_path = glob(lstm_model)[0]
    except IndexError as excep:
        logger.error("Can't find lstm path: %s", lstm_model)
        raise excep
    net = load_lstm_model(nfine_bins, nfine2_bins, lstm_path, device_str)
    net = net.to(device)
    logger.info("Testing net: %s", str(net))
    net.eval()
    if nfine_bins is None:
        test_lstm = LSTMEvaluator(net, device, *other_cls_args,
                                  **other_cls_kwargs)
    elif nfine2_bins is None:
        test_lstm = JointLSTMEvaluator(net, device, *other_cls_args,
                                       **other_cls_kwargs)
    else:
        test_lstm = TripleLSTMEvaluator(net, device, *other_cls_args,
                                        **other_cls_kwargs)
    return test_lstm


def create_batch_predictor(lstm_eval, coarse_cutoffs, args, bdecoder):
    """Helper to take care of creating the BatchPredictor.

    Arguments:

    lstm_eval: LSTMEvaluator: system passed to generator to generate
    the outputs auto-regressively.

    coarse_cutoffs: List[Float], the bin boundaries for coarse input, output.

    args, NamedTuples from argparser with all the many arguments to
    c2far.

    bdecoder: IntraBinDecoder, method for decoding values within each
    coarse bin.

    Returns:

    BatchPredictor

    """
    bpred = BatchPredictor(lstm_eval, coarse_cutoffs, args.coarse_low,
                           args.coarse_high, csize=args.csize,
                           nsize=args.nsize, nstride=args.nstride,
                           gsize=args.gsize, nsamples=args.nsamples,
                           device=args.device,
                           confidence=args.confidence_pct,
                           bdecoder=bdecoder,
                           nfine_bins=args.nfine_bins,
                           nfine2_bins=args.nfine2_bins,
                           extremas=args.extremas)
    return bpred


def __make_dataset(do_cont, vcpus, mem, offs, coarse_cutoffs,
                   coarse_low, coarse_high, shared_fargs, dargs):
    """Helper to avoid repeating the do_cont check logic."""
    if not do_cont:
        dset = BinDataset(vcpus, mem, offs,
                          coarse_cutoffs=coarse_cutoffs,
                          coarse_low=coarse_low,
                          coarse_high=coarse_high, **shared_fargs,
                          dataset_args=dargs)
    else:
        dset = ContDataset(vcpus, mem, offs, **shared_fargs, dataset_args=dargs)
    return dset


def make_dataloaders(args, coarse_cutoffs, *, make_train_set=True,
                     make_test_set=True, include_train_origs=False,
                     disable_randomize_train_start=False):
    """Return the training and test dataloaders, by making the respective
    datasets.

    Arguments:

    args, NamedTuples from argparser with all the many arguments to
    c2far.

    coarse_cutoffs: List[Float], the bin boundaries for coarse input, output.

    Optional arguments:

    make_train_set: Boolean, whether to make the training dataloader.

    make_test_set: Boolean, whether to make the test dataloader.

    include_train_origs: Boolean, note by default, we always include
    the test originals, but this argument controls whether we include
    the train ones (rarely necessary).  TODO: something similar for
    include_coarses?

    disable_randomize_train_start: Boolean, if given, we ignore
    whatever is in args.randomize_start and simply choose False for
    this train arg.

    """
    if not (make_train_set or make_test_set):
        raise RuntimeError("Must make either the train or test set")
    shared_dargs = {"nfine_bins": args.nfine_bins, "nfine2_bins":
                    args.nfine2_bins, "extremas": args.extremas}
    shared_fargs = {"csize": args.csize, "nsize": args.nsize, "gsize": args.gsize}
    shared_largs = {"collate_fn": CollateUtils.batching_collator,
                    "num_workers": args.num_workers}
    if make_test_set:
        do_include_coarses = True
        test_dargs = DatasetArgs(ncheckpoint=args.ntest_checkpoint,
                                 loss_start_pt_s=args.test_loss_start_pt_s,
                                 loss_end_pt_s=args.test_loss_end_pt_s,
                                 cache_set=args.cache_testset,
                                 include_coarses=do_include_coarses,
                                 include_originals=True, **shared_dargs)
        testset = __make_dataset(args.continuous, args.test_vcpus, args.test_mem, args.test_offs,
                                 coarse_cutoffs, args.coarse_low, args.coarse_high,
                                 shared_fargs, test_dargs)
        testloader = DataLoader(testset, shuffle=False,
                                batch_size=args.test_batch_size, **shared_largs)
    else:
        testloader = None
    if make_train_set:
        if disable_randomize_train_start:
            do_randomize_start = False
        else:
            do_randomize_start = args.randomize_start
        ncheckpoint = args.ntrain_checkpoint
        train_dargs = DatasetArgs(ncheckpoint=ncheckpoint, randomize_start=do_randomize_start,
                                  loss_start_pt_s=args.train_loss_start_pt_s,
                                  loss_end_pt_s=args.train_loss_end_pt_s,
                                  include_originals=include_train_origs, **shared_dargs)
        trainset = __make_dataset(args.continuous, args.train_vcpus, args.train_mem, args.train_offs,
                                  coarse_cutoffs, args.coarse_low, args.coarse_high,
                                  shared_fargs, train_dargs)
        trainloader = DataLoader(trainset, shuffle=True,
                                 batch_size=args.train_batch_size, **shared_largs)
    else:
        trainloader = None
    return trainloader, testloader
