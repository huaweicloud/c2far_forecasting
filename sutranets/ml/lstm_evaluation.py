"""Read in the training data and test data and do a proper evaluation,
comparing to baselines.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/lstm_evaluation.py
import argparse
import logging
import torch
from sutranets.dataprep.cutoffs.create_cutoffs import setup_cutoffs
from sutranets.ml.add_args import add_standard_args, add_gen_eval_args
from sutranets.ml.eval_utils import EvalUtils
from sutranets.ml.make_criterions import MakeCriterions
from sutranets.ml.utils import load_lstm_evaluator
from sutranets.ml.dataset_utils import DatasetUtils
from sutranets.utils import cross_arg_checker, set_all_seeds
logger = logging.getLogger("sutranets.ml.lstm_evaluation")


def __run_evals(testloaders, systems, criterions):
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


def run_next_step_eval(testloaders, lstm_eval, coarse_cutoffs, args):
    """Run the standard evaluation of 1-next-step predictions from logits.

    """
    if args.dont_run_1step_eval:
        return None
    if lstm_eval is None:
        raise RuntimeError("Need an lstm_eval to evaluate.")
    systems = [lstm_eval]
    criterions = MakeCriterions.make_next_step_test(args, coarse_cutoffs)
    return __run_evals(testloaders, systems, criterions)


def run_multi_step_eval(testloaders, lstm_eval, coarse_cutoffs, args):
    """Run the generator-based multi-step evaluation.

    """
    if not args.run_gen_eval:
        return None
    systems = EvalUtils.create_multi_evaluators(lstm_eval,
                                                coarse_cutoffs, args)
    criterions = MakeCriterions.create_gen_criterions(
        args.confidence_pct, do_horizs=args.do_horizs, gsize=args.gsize,
        nsub_series=args.nsub_series)
    return __run_evals(testloaders, systems, criterions)


def setup_eval(coarse_cutoffs, args, *, include_meta=False):
    """Helper to setup logging, as well as the lstm_evaluator and the
    testloaders.  We re-use this in the eval event generation.

    """
    assert args.lstm_model, "Need model to run the LSTM"
    logger.info("Reading in test data for evaluation.")
    _, testloader = DatasetUtils.make_dataloaders(args, coarse_cutoffs,
                                                  include_meta=include_meta, make_train_set=False)
    testloaders = [testloader]
    if args.lstm_model is not None:
        EvalUtils.validate_cutoffs(args.lstm_model, coarse_cutoffs)
        lstm_eval = load_lstm_evaluator(args.device, args.lstm_model,
                                        do_init_hidden=True, nfine_bins=args.nfine_bins,
                                        nfine2_bins=args.nfine2_bins,
                                        nsub_series=args.nsub_series)
    else:
        lstm_eval = None
    return lstm_eval, testloaders


def do_lstm_evaluation(args):
    """Create the cutoffs, read in the datasets, and run the specified
    evaluations on these data.

    """
    set_all_seeds(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    coarse_cutoffs = setup_cutoffs(None, args)
    lstm_eval, testloaders = setup_eval(coarse_cutoffs, args)
    run_next_step_eval(testloaders, lstm_eval, coarse_cutoffs, args)
    run_multi_step_eval(testloaders, lstm_eval, coarse_cutoffs, args)


def main(args):
    """Setup logging and then launch the main eval process."""
    if not args.quiet:
        EvalUtils.setup_logging(args)
    do_lstm_evaluation(args)


def add_lstm_evaluation_args(parser, *, add_train_args=True):
    """Helper to add the standard evaluation arguments to the argument
    parser.  By default we do require the training set.

    """
    add_standard_args(parser, add_train_args=add_train_args)
    add_gen_eval_args(parser)
    parser.add_argument('--quiet', action="store_true",
                        help="If given, don't have any logging.")
    parser.add_argument('--lstm_model', type=str, required=False,
                        help="The model for the LSTM that we wish to test.")
    parser.add_argument('--logging_dir', type=str, required=False,
                        help="If provided, also log to subdir of this dir.")
    parser.add_argument('--dont_run_baselines', action="store_true",
                        help="Use this to disable the multi-step-ahead baselines.")
    parser.add_argument('--do_horizs', action="store_true",
                        help="Run the multi-step MAE evaluation at all the horizons.")


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="Eval of LSTM and Baselines Predictor/Generators")
    add_lstm_evaluation_args(parser)
    args = parser.parse_args(args)
    cross_arg_checker(args, parser, check_stride=True, check_multiv=True)
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
