"""Read in the training data and test data and do an evaluation.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/lstm_evaluation.py

import argparse
import logging
import os
import torch
from c2far.dataprep.cutoffs.create_cutoffs import CreateCutoffs
from c2far.file_utils import setup_outdir
from c2far.ml.add_args import add_standard_args, add_gen_eval_args
from c2far.ml.constants import IntraBinDecoder
from c2far.ml.evaluators.generation_evaluator import GenerationEvaluator
from c2far.ml.make_criterions import MakeCriterions
from c2far.ml.utils import get_lstm_evaluator, make_dataloaders, create_batch_predictor
from c2far.utils import init_console_logger, init_file_logger, cross_arg_checker, set_all_seeds

logger = logging.getLogger("c2far.ml.lstm_evaluation")
LOG_SHORT_FN = "log.txt"


def run_evals(testloaders, systems, criterions):
    """Helper to run through all the test sets, systems, and criterions
    and get all the results.

    """
    all_losses = []
    for testloader in testloaders:
        logger.info("")
        logger.info("")
        logger.info("Testing on next dataset")
        for system in systems:
            logger.info("")
            logger.info(system)
            with torch.no_grad():
                loss = system.get_test_score(criterions, testloader)
                all_losses.append(loss)
    return all_losses


def make_log_dir_plus_fn(args):
    """Helper to create the logging (output) dir, create the path to the
    logging file, and return this path.

    """
    subdir_parts = []
    subdir_parts.append(os.path.basename(args.test_offs))
    outdir = os.path.join(args.logging_dir, *subdir_parts)
    target_dir = setup_outdir(outdir)
    log_fn = os.path.join(target_dir, LOG_SHORT_FN)
    return log_fn


def setup_logging(args):
    """Helper function to set up all the logging."""
    logger_levels = [("c2far", logging.INFO)]
    init_console_logger(logger_levels)
    if args.logging_dir is not None:
        log_fn = make_log_dir_plus_fn(args)
        init_file_logger(logger_levels, log_fn)
    logger.info("ARGS: %s", str(vars(args)))


def run_next_step_eval(testloaders, lstm_eval, coarse_cutoffs, args):
    """Run the standard evaluation of 1-next-step predictions from logits.

    """
    if lstm_eval is None:
        raise RuntimeError("Need an lstm_eval to evaluate.")
    systems = [lstm_eval]
    # Here's our losses - currently this is just the CE loss:
    criterions = MakeCriterions.make_next_step_test(args, coarse_cutoffs)
    return run_evals(testloaders, systems, criterions)


def create_gen_eval(lstm_eval, coarse_cutoffs, args):
    """Helper to return an LSTM-based multi-step generation evaluator.

    Arguments:

    lstm_eval: LSTMEvaluator: system passed to generator to generate
    the outputs auto-regressively.

    coarse_cutoffs: Tensor[Float], the bin boundaries for coarse input, output.

    args, NamedTuples from argparser with all the many arguments to c2far.

    Returns:

    GenerationEvaluator

    """
    key_arg_names = ["nsamples", "confidence", "nstride", "lstm_evaluator"]
    key_arg_values = [args.nsamples, args.confidence_pct, args.nstride, lstm_eval]
    for arg_nm, arg in zip(key_arg_names, key_arg_values):
        if arg is None:
            raise RuntimeError(f"Required argument '{arg_nm}' is unexpectedly None")
    if not args.continuous:
        key_arg_names = ["gen_bdecoder", "coarse_cutoffs"]
        key_arg_values = [args.gen_bdecoder, coarse_cutoffs]
        for arg_nm, arg in zip(key_arg_names, key_arg_values):
            if arg is None:
                raise RuntimeError(f"'{arg_nm}' required for non-continuous eval")
        if not isinstance(args.gen_bdecoder, IntraBinDecoder):
            raise RuntimeError(f"gen_bdecoder {args.gen_bdecoder} is not an IntraBinDecoder enum")
    predictor = create_batch_predictor(lstm_eval, coarse_cutoffs,
                                       args, args.gen_bdecoder)
    gen_eval = GenerationEvaluator(predictor, args.device)
    return gen_eval


def create_multi_evaluators(lstm_eval, coarse_cutoffs, args):
    """Create the multi-step-ahead evaluators (LSTM).

    Arguments:

    lstm_eval: Evaluator, system that can generate LSTM predictions on
    test data as an evaluator.

    coarse_cutoffs: Tensor[Float], the bin boundaries for coarse input, output.

    args: Dict, all command-line arguments.

    """
    systems = []
    if lstm_eval is not None:
        gen_eval = create_gen_eval(lstm_eval, coarse_cutoffs, args)
        systems.append(gen_eval)
    return systems


def run_multi_step_eval(testloaders, lstm_eval, coarse_cutoffs, args):
    """Run the generator-based multi-step evaluation.

    """
    if not args.run_gen_eval:
        return None
    systems = create_multi_evaluators(lstm_eval, coarse_cutoffs, args)
    criterions = MakeCriterions.create_gen_criterions(args.confidence_pct)
    return run_evals(testloaders, systems, criterions)


def setup_eval(coarse_cutoffs, args):
    """Helper to setup logging, as well as the lstm_evaluator and the
    testloaders.

    """
    logger.info("Reading in test data for evaluation.")
    _, testloader = make_dataloaders(args, coarse_cutoffs,
                                     make_train_set=False)
    testloaders = [testloader]
    # Here we do init hidden state, by default, so that it works fine
    # in 1-step eval, but we disable internally when doing the
    # generation eval (where we go 1-step at a time).
    lstm_eval = get_lstm_evaluator(
        args.device, args.lstm_model, do_init_hidden=True,
        nfine_bins=args.nfine_bins, nfine2_bins=args.nfine2_bins)
    return lstm_eval, testloaders


def do_lstm_evaluation(args):
    """Create the cutoffs, read in the datasets, and run the specified
    evaluations on these data.

    """
    set_all_seeds(args.seed)
    coarse_cutoffs = CreateCutoffs.get_coarse_cutoffs(args)
    lstm_eval, testloaders = setup_eval(coarse_cutoffs, args)
    # Run the standard next-step-eval:
    # run_next_step_eval(testloaders, lstm_eval, coarse_cutoffs, args)
    # Next, run the generation multi-step-eval:
    run_multi_step_eval(testloaders, lstm_eval, coarse_cutoffs, args)


def main(args):
    """Setup logging and then launch the main eval process."""
    if not args.quiet:
        setup_logging(args)
    do_lstm_evaluation(args)


def add_lstm_evaluation_args(parser):
    """Helper to add the standard evaluation arguments to the argument
    parser.  By default we do require the training set.

    """
    add_standard_args(parser, add_train_args=False)
    add_gen_eval_args(parser)
    parser.add_argument('--quiet', action="store_true",
                        help="If given, don't have any logging.")
    parser.add_argument('--lstm_model', type=str, required=True,
                        help="The model for the LSTM that we wish to test.")
    parser.add_argument('--logging_dir', type=str, required=False,
                        help="If provided, also log to subdir of this dir.")


def parse_arguments(args=None):
    """Helper function to parse the command-line arguments, return as an
    'args' object.  If args == None, will get args from sys.argv[1:].

    """
    parser = argparse.ArgumentParser(
        description="Eval of LSTM and Baselines Predictor/Generators")
    add_lstm_evaluation_args(parser)
    args = parser.parse_args(args)
    cross_arg_checker(args, parser, check_stride=True)
    return args


if __name__ == "__main__":  # pragma: no cover
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
