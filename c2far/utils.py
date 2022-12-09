"""Generic utilities related to logging, CLI args, etc.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/utils.py

import logging
import os
import random
import numpy as np
import torch
from c2far.ml.constants import BinStrategy

logger = logging.getLogger("c2far.utils")
LOG_FORMAT = "%(asctime)s [%(name)-10.35s] [%(levelname)-5.5s]  %(message)s"


def init_log_helper(handler, logger_levels):
    """Helper function with the common logic of initing both loggers."""
    log_formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(log_formatter)
    for loggername, loglevel in logger_levels:
        mylogger = logging.getLogger(loggername)
        mylogger.setLevel(loglevel)
        mylogger.addHandler(handler)
        # See stackoverflow,
        # python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        mylogger.propagate = False


def init_file_logger(logger_levels, filename):
    """Configure the loggers to write to stdout AND to disk.  Pick their
    format, levels, etc.

    """
    # Possibly we are already logging to another file (e.g. if we are
    # doing a loop in tuning) so stop outputting to that file first.
    for loggername, _ in logger_levels:
        mylogger = logging.getLogger(loggername)
        keep_handlers = [h for h in mylogger.handlers if not
                         isinstance(h, logging.FileHandler)]
        mylogger.handlers = keep_handlers
    file_handler = logging.FileHandler(filename)
    init_log_helper(file_handler, logger_levels)


def init_console_logger(logger_levels):
    """Configure the loggers to write to stdout only.  Pick their format,
    levels, etc.

    """
    console_handler = logging.StreamHandler()
    init_log_helper(console_handler, logger_levels)


def set_all_seeds(seed_value):
    """To enable reproducibility of experiments."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def add_binning_args(parser):
    """Args specifically related to our output binning."""
    parser.add_argument('--ncoarse_bins', type=int, required=False,
                        help="The cutoffs for our coarse bins - unless doing continuous.")
    parser.add_argument('--coarse_low', type=float, required=False,
                        help="The extreme low cutoff for (normed) binning (e.g. -5.0), "
                        "required unless doing continuous")
    parser.add_argument('--coarse_high', type=float, required=False,
                        help="The extreme high cutoff for (normed) binning (e.g. +5.0), "
                        "required unless doing continuous")
    parser.add_argument('--nfine_bins', type=int, required=False,
                        help="If given, encode fine-grained intra-bin idxs as input and output.")
    parser.add_argument('--nfine2_bins', type=int, required=False,
                        help="If given, encode finer fine2-grained intra-bin idxs as well.")


def add_map_args(parser):
    """Helper to add args basically controlling whether we bin, how we
    bin, for features or output.

    """
    parser.add_argument('--extremas', action="store_true",
                        help="If given, use special features/outputs for extremas.")
    add_binning_args(parser)
    parser.add_argument('--bin_strategy', type=BinStrategy, required=False, default=BinStrategy.LINEAR,
                        help="What type of binning to use: linear (default) - actually that's only one supported.")
    parser.add_argument('--continuous', action="store_true",
                        help="If given, don't use binning, use real values.")


def add_arg_triple(parser, label):
    """Helper to add the vcpus, memory, and offset file arguments, with a
    label.

    Arguments:

    parser: ArgumentParser

    label: String, if given, use the label as a suffix for
    vcpus/memory/offs

    """
    vcpus_arg = f'--{label}_vcpus'
    mem_arg = f'--{label}_mem'
    offs_arg = f'--{label}_offs'
    help_label = f"{label}ing "
    kwargs = {"type": str, "required": True}
    parser.add_argument(vcpus_arg, **kwargs,
                        help=f"The vcpus {help_label}data.")
    parser.add_argument(mem_arg, **kwargs,
                        help=f"The {help_label}mem, digitized.")
    parser.add_argument(offs_arg, **kwargs,
                        help=f"The {help_label}offs, digitized.")


def add_window_args(parser):
    """Helper to add args for the conditioning, normalization, and
    generated window sizes.

    """
    parser.add_argument(
        '--csize', type=int, required=True,
        help="How much to condition on.")
    parser.add_argument(
        '--nsize', type=int, required=True,
        help="How much to normalize on.")
    parser.add_argument(
        '--gsize', type=int, required=True,
        help="How much to generate.")


def cross_arg_checker(args, parser, *, check_stride=False):
    """Check relationships between standard arguments are valid
    (cross-argument validation).  E.g., the normalization window size
    (nsize) should be <= the conditioning window size (csize).

    Arguments:

    args: the arguments to check

    parser: the ArgumentParser object, which we pass in here to give
    standard parser error messages.

    check_stride: Boolean, if True, we also check the stride vs. gsize
    (needed when we do generation stuff).

    """
    if args.nsize > args.csize:
        parser.error(
            f"--nsize=={args.nsize} must be <= --csize=={args.csize}")
    if check_stride and args.nstride is not None and args.nstride > args.gsize:
        parser.error(
            f"--nstride=={args.nstride} must be <= --gsize=={args.gsize}")
