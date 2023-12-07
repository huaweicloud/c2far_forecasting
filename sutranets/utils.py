"""Generic file, log, and CLI-argument utilities.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/utils.py

import logging
import os
import random
import numpy as np
import torch
from sutranets.dataprep.data_utils import DEFAULT_TRIVIAL_MIN_NONZERO_FRAC
from sutranets.ml.constants import BinStrategy

logger = logging.getLogger(
    "sutranets." + os.path.basename(__file__))
LOG_FORMAT = "%(asctime)s [%(name)-10.35s] [%(levelname)-5.5s]  %(message)s"


def __filter_handlers(logger_levels, filt_cls):
    """Possibly we are already logging to another file (e.g. if we are
    doing a loop in tuning) so stop outputting to that file first, or
    we are in UTs and attaching console loggers repeatedly, so remove
    those first too.

    """
    for loggername, _ in logger_levels:
        mylogger = logging.getLogger(loggername)
        keep_handlers = [h for h in mylogger.handlers if not
                         isinstance(h, filt_cls)]
        mylogger.handlers = keep_handlers


def init_log_helper(handler, logger_levels):
    """Helper function with the common logic of initing file and console
    loggers.

    """
    log_formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(log_formatter)
    for loggername, loglevel in logger_levels:
        mylogger = logging.getLogger(loggername)
        mylogger.setLevel(loglevel)
        mylogger.addHandler(handler)
        mylogger.propagate = False


def init_file_logger(logger_levels, filename):
    """Configure the loggers to write to stdout AND to disk.  Pick their
    format, levels, etc.

    """
    __filter_handlers(logger_levels, logging.FileHandler)
    file_handler = logging.FileHandler(filename)
    init_log_helper(file_handler, logger_levels)


def init_console_logger(logger_levels):
    """Configure the loggers to write to stdout only.  Pick their format,
    levels, etc.

    """
    __filter_handlers(logger_levels, logging.StreamHandler)
    console_handler = logging.StreamHandler()
    init_log_helper(console_handler, logger_levels)


def set_all_seeds(seed_value):
    """For experimental reproducibility, set seeds in Python, numpy,
    torch.

    """
    if seed_value is not None:
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)


def add_binning_args(parser):
    """Args specifically related to our output binning."""
    parser.add_argument('--ncoarse_bins', type=int, required=False,
                        help="The cutoffs for our coarse bins")
    parser.add_argument('--coarse_low', type=float, required=False,
                        help="The extreme low cutoff for (normed) binning (e.g. -5.0)")
    parser.add_argument('--coarse_high', type=float, required=False,
                        help="The extreme high cutoff for (normed) binning (e.g. +5.0)")
    parser.add_argument('--nfine_bins', type=int, required=False,
                        help="If given, encode fine-grained intra-bin idxs as input and output.")
    parser.add_argument('--nfine2_bins', type=int, required=False,
                        help="If given, encode finer fine2-grained intra-bin idxs as well.")


def add_map_args(parser):
    """Helper to add args controlling whether we bin, how we bin, for
    features or output.

    """
    parser.add_argument('--extremas', action="store_true", default=False,
                        help="If given, use special features/outputs for extremas.")
    parser.add_argument('--lagfeat_period', type=int, required=False,
                        help="If given, use lag features with the given period.")
    parser.add_argument('--nsub_series', type=int, required=False,
                        help="Use a multivariate high-frequency dataset with the given number of sub-series.")
    parser.add_argument('--sub_csize', type=int, required=False,
                        help="Use the given csize in the multivariate high-frequency dataset.")
    parser.add_argument('--mv_no_prevs', action="store_true", default=False,
                        help="If given, assume wall2wall generation, only use 'current' AR values.")
    parser.add_argument('--mv_backfill', action="store_true", default=False,
                        help="If given, we generate multivariate series from furthest to closest.")
    parser.add_argument('--sample_period_s', type=int, required=False,
                        help="Need to explicitly provide sample period for some evaluators, datasets.")
    add_binning_args(parser)
    parser.add_argument('--bin_strategy', type=BinStrategy, required=False, default=BinStrategy.LINEAR,
                        help="What type of binning to use: linear (default) - actually that's only one supported.")
    parser.add_argument('--cutoff_dir', type=str, required=False,
                        help="DIR with cutoffs - get from here instead of making.")


def add_arg_triple(parser, *, label=""):
    """Helper to add the trace files and offset file arguments, with a
    label.

    """
    if label:
        traces_arg = f'--{label}_trace_paths'
        offs_arg = f'--{label}_offs'
        help_label = f"{label}ing "
    else:
        traces_arg = '--trace_paths'
        offs_arg = '--offs'
        help_label = ""
    kwargs = {"type": str, "required": True}
    parser.add_argument(traces_arg, **kwargs, nargs="+",
                        help=f"The paths for the trace {help_label}data.")
    parser.add_argument(offs_arg, **kwargs,
                        help=f"The offsets {help_label}data.")


def add_window_args(parser):
    """Helper to add args for the conditioning, normalization, and
    generated window sizes.

    """
    parser.add_argument('--csize', type=int, required=True, help="How much to condition on.")
    parser.add_argument('--nsize', type=int, required=True, help="How much to normalize on.")
    parser.add_argument('--gsize', type=int, required=True, help="How much to generate.")
    parser.add_argument('--trivial_min_nonzero_frac', type=float, required=False,
                        default=DEFAULT_TRIVIAL_MIN_NONZERO_FRAC,
                        help="Consider series trivial if not >= this frac of nwindow is non-zero")


def cross_arg_checker(args, parser, *, check_nsize=True,
                      check_stride=False, check_multiv=False):
    """Check relationships between standard arguments are valid
    (cross-argument validation).  E.g., the normalization window size
    (nsize) should be <= the conditioning window size (csize).

    """
    if check_nsize and args.nsize > args.csize:
        parser.error(f"--nsize=={args.nsize} must be <= --csize=={args.csize}")
    if check_stride and args.nstride is not None and args.nstride > args.gsize:
        parser.error(f"--nstride=={args.nstride} must be <= --gsize=={args.gsize}")
    if check_multiv and ((args.nsub_series is None) != (args.sub_csize is None)):
        raise RuntimeError("nsub_series and sub_csize must be used together.")
