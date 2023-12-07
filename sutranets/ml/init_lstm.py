"""Create an initial LSTM neural network model with the given
characteristics.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD
"""
# sutranets/ml/init_lstm.py
import argparse
import logging
from sutranets.dataprep.cutoffs.create_cutoffs import CreateCutoffs
from sutranets.ml.nets.fc_lstm import FCLSTM
from sutranets.ml.nets.joint_lstm import JointLSTM
from sutranets.ml.nets.multivariate_lstm import MultivariateLSTM
from sutranets.ml.nets.triple_lstm import TripleLSTM
from sutranets.ml.bin_tensor_maker_dims import BinTensorMakerDims
from sutranets.utils import add_map_args, init_console_logger, \
    cross_arg_checker
logger = logging.getLogger("sutranets.ml.init_lstm")


def analyze_model(model):
    """Analyze the different parts of the model and how many parameters
    they have.

    """
    pairs = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        pairs.append((name, params))
        total_params += params
    return pairs, total_params


def __create_model_from_args(args, ninput, noutput):
    """Create the actual model based on the args."""
    if args.nfine_bins is None:
        model = FCLSTM(ninput, args.nhidden, noutput, args.nlayers, extremas=args.extremas)
    elif args.nfine2_bins is None:
        model = JointLSTM(ninput, args.ncoarse_bins, args.nfine_bins,
                          args.nhidden, noutput, args.nlayers, extremas=args.extremas)
    else:
        model = TripleLSTM(ninput, args.ncoarse_bins, args.nfine_bins,
                           args.nfine2_bins, args.nhidden, noutput,
                           args.nlayers, extremas=args.extremas)
    return model


def __get_tmaker_dims(args, cutoffs, nmultiv_siblings):
    """Get a tmaker_dims object from the given arguments.

    """
    if cutoffs is None:
        if args.ncoarse_bins is None:
            raise RuntimeError("Need args.ncoarse_bins.")
        coarse_cutoffs = CreateCutoffs.do_create_cutoffs(
            args, force_linear=True)
    else:
        coarse_cutoffs = cutoffs
    tmaker_dims = BinTensorMakerDims(coarse_cutoffs,
                                     nfine_bins=args.nfine_bins,
                                     nfine2_bins=args.nfine2_bins,
                                     extremas=args.extremas,
                                     lagfeat_period=args.lagfeat_period,
                                     nmultiv_siblings=nmultiv_siblings)
    return tmaker_dims


def __init_multiv_model(args, cutoffs):
    """Make the model specifically for the multivariate case."""
    models = []
    nmultiv_siblings = args.nsub_series - 1
    for i in range(args.nsub_series):
        if args.mv_no_prevs:
            nmultiv_siblings = i
        tmaker_dims = __get_tmaker_dims(args, cutoffs, nmultiv_siblings=nmultiv_siblings)
        ninput = tmaker_dims.get_ninput()
        noutput = tmaker_dims.get_noutput()
        model = __create_model_from_args(args, ninput, noutput)
        models.append(model)
    model = MultivariateLSTM(models)
    return model


def create_lstm(args, *, cutoffs=None):
    """Helper to create the LSTM, but not write it.

    """
    if args.nsub_series:
        model = __init_multiv_model(args, cutoffs)
        return model
    tmaker_dims = __get_tmaker_dims(args, cutoffs, nmultiv_siblings=0)
    ninput = tmaker_dims.get_ninput()
    noutput = tmaker_dims.get_noutput()
    logger.debug("Model with %d input, %d output, %d nhidden", ninput, noutput, args.nhidden)
    model = __create_model_from_args(args, ninput, noutput)
    return model


def do_init_lstm(args):
    """Reads the argument info on the NN topology and builds a model for
    an LSTM NN.

    """
    model = create_lstm(args)
    pairs, total_params = analyze_model(model)
    logger.info("Total Trainable Params: %d", total_params)
    for name, numelem in pairs:
        logger.info("%s: %d", name, numelem)
    model.save_python(args.out_fn)


def main(args):
    logger_levels = [("sutranets", logging.DEBUG)]
    init_console_logger(logger_levels)
    do_init_lstm(args)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="Forecasting LSTM model creator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_map_args(parser)
    parser.add_argument(
        '--gsize', type=int, required=False,
        help="How much to generate (high-freq).")
    parser.add_argument(
        '--nhidden', type=int, required=True,
        help="The number of hidden units in each hidden layer - "
        "provide as a list.")
    parser.add_argument(
        '--nlayers', type=int, required=True,
        help="The number of layers in an LSTM - only for LSTMs.")
    parser.add_argument(
        '--out_fn', type=str, required=True,
        help="Where to save the output.")
    args = parser.parse_args(args)
    cross_arg_checker(args, parser, check_nsize=False, check_stride=False, check_multiv=True)
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
