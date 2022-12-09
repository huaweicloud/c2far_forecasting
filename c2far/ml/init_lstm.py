"""Create an LSTM neural network model with the given characteristics.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/init_lstm.py

import argparse
import logging
from c2far.dataprep.cutoffs.create_cutoffs import CreateCutoffs
from c2far.ml.nets.fc_lstm import FCLSTM
from c2far.ml.nets.joint_lstm import JointLSTM
from c2far.ml.nets.triple_lstm import TripleLSTM
from c2far.ml.bin_tensor_maker import BinTensorMaker
from c2far.ml.continuous.cont_tensor_maker import ContTensorMaker
from c2far.utils import add_map_args, init_console_logger
logger = logging.getLogger("c2far.ml.init_lstm")


def analyze_model(model):
    """Analyze the different parts of the model and how many parameters
    they have.

    Arguments:

    model: nn.Module, e.g. FCLSTM, JointLSTM, TripleLSTM

    Returns:

    pairs: List[(String, Int)] with param_name, num. elements pairs

    total_params: Int, the total number of parameters for this model.

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


def create_lstm(args):
    """Helper to create the LSTM, but not write it.

    Arguments:

    args: CLI-arg named-tuple.

    Returns:

    model: nn.Module, e.g. FCLSTM, JointLSTM, TripleLSTM

    """
    if not args.continuous:
        if args.ncoarse_bins is None:
            raise RuntimeError("Need args.ncoarse_bins if not doing continuous.")
        coarse_cutoffs = CreateCutoffs.get_coarse_cutoffs(args)
        # Let's ask the tensor_maker the ndims corresponding to these
        # maps (genmode is irrelevant here):
        tmaker = BinTensorMaker(coarse_cutoffs, args.coarse_low,
                                args.coarse_high,
                                nfine_bins=args.nfine_bins,
                                nfine2_bins=args.nfine2_bins,
                                genmode=False, extremas=args.extremas)
    else:
        tmaker = ContTensorMaker(genmode=False)
    ninput = tmaker.get_ninput()
    noutput = tmaker.get_noutput()
    logger.debug("Model with %d input, %d output, %d nhidden", ninput, noutput, args.nhidden)
    if args.continuous and args.extremas:
        raise RuntimeError("Cannot use continuous with extremas")
    if args.continuous or args.nfine_bins is None:
        model = FCLSTM(ninput, args.nhidden, noutput, args.nlayers, extremas=args.extremas)
    elif args.nfine2_bins is None:
        model = JointLSTM(ninput, args.ncoarse_bins, args.nfine_bins,
                          args.nhidden, noutput, args.nlayers, extremas=args.extremas)
    else:
        model = TripleLSTM(ninput, args.ncoarse_bins, args.nfine_bins,
                           args.nfine2_bins, args.nhidden, noutput,
                           args.nlayers, extremas=args.extremas)
    return model


def do_init_lstm(args):
    """Helper that actually reads the argument info on the NN topology and
    builds a model for an LSTM NN, and saves to disk.

    """

    model = create_lstm(args)
    pairs, total_params = analyze_model(model)
    logger.info("Total Trainable Params: %d", total_params)
    for name, numelem in pairs:
        logger.info("%s: %d", name, numelem)
    model.save_python(args.out_fn)


def main(args):
    """Read the info on the NN topology and builds a model for an LSTM NN.

    """
    logger_levels = [("c2far", logging.DEBUG)]
    init_console_logger(logger_levels)
    do_init_lstm(args)


def parse_arguments(args=None):
    """Helper function to parse the command-line arguments, return as an
    'args' object.  If args == None, will get args from sys.argv[1:].

    """
    parser = argparse.ArgumentParser(
        description="Forecasting LSTM model creator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_map_args(parser)
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
    return parser.parse_args(args)


if __name__ == "__main__":  # pragma: no cover
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
