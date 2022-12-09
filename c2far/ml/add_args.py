"""Common arguments that are used for training and/or testing.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/add_args.py

from c2far.file_utils import MyArgumentTypes
from c2far.ml.constants import IntraBinDecoder
from c2far.utils import add_map_args, add_arg_triple, add_window_args

DEFAULT_MAX_NUM_ITERS = 100000
DEFAULT_NUM_WORKERS = 8
DEFAULT_EARLY_STOP_PATIENCE = 37


def add_gen_eval_args(parser):
    """Arguments specific to multi-step-generation evaluation."""
    parser.add_argument('--run_gen_eval', action="store_true",
                        help="Run multi-step generation evaluation.")
    parser.add_argument('--gen_bdecoder', type=IntraBinDecoder, required=False,
                        help="Which intra-bin-decoder to use in generation.")
    parser.add_argument('--nsamples', type=int, required=False,
                        help="Num. samples to use in multi-step generation eval.")
    parser.add_argument('--confidence_pct', type=int, required=False,
                        help="Confidence value (in %) indicating desired width of intervals "
                        "in multi-step generation eval.")
    parser.add_argument('--sample_period_s', type=int, required=False,
                        help="Need to explicitly provide sample period for some evaluators.")
    parser.add_argument('--nstride', type=int, required=False,
                        help="How much to stride each time when generating.")


def add_option_args(parser, *, add_train_args=True, add_test_args=True):
    """Helper to add args that provide common options that we use in
    different scenarios (training, testing, generation).

    """
    add_map_args(parser)
    add_window_args(parser)
    parser.add_argument('--device', type=str, required=True,
                        help="Run the optimization on a GPU (\"cuda:[01]\") "
                        "or on the CPU (\"cpu\").")
    if add_train_args:
        parser.add_argument('--train_batch_size', type=int, required=True,
                            help="How big to make the *TRAINING* batches after collation.")
        parser.add_argument('--ntrain_checkpoint', type=int, required=False, default=-1,
                            help="How many examples to process before checkpoint (~epoch size). "
                            "Pass -1 to use all (default).")
    if add_test_args:
        parser.add_argument('--test_batch_size', type=int, required=True,
                            help="How big to make the *TESTING* batches after collation.")
        parser.add_argument('--ntest_checkpoint', type=int, required=False, default=-1,
                            help="Num. test examples to use in total. -1 means use all.")
        parser.add_argument('--dont_run_1step_eval', action="store_true",
                            help="Don't run the standard 1-step evaluation.")
    parser.add_argument('--randomize_start', action="store_true",
                        required=False, default=False,
                        help="If given, randomize the starting point within the offsets.")


def add_set_args_by_label(parser, label, require_loss_start=True,
                          require_loss_end=True):
    """Helper so we can share the code for label=train, label=test, or
    whatever else you need.

    """
    add_arg_triple(parser, label=label)
    parser.add_argument(f'--{label}_loss_start_pt_s', type=int, required=require_loss_start,
                        help=f"If given, only compute loss on points from here on in {label}.")
    parser.add_argument(f'--{label}_loss_end_pt_s', type=int, required=require_loss_end,
                        help=f"If given, only compute loss on points from here on in {label}.")


def add_standard_args(parser, *, add_train_args=True, add_test_args=True):
    """Helper to add arguments that we use, whether training or testing,
    whether flavs or durs.

    """
    # Train should have loss endpoint, test should have startpoint (can
    # set to small/HUGE if needed for, e.g. test on train experiment):
    if add_train_args:
        add_set_args_by_label(parser, label="train", require_loss_start=False)
    if add_test_args:
        add_set_args_by_label(parser, label="test", require_loss_end=False)
        parser.add_argument('--cache_testset', action="store_true",
                            help="If given, store testset in memory after creating examples once.")
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help="If given, set a random seed for this run.")
    add_option_args(parser, add_train_args=add_train_args, add_test_args=add_test_args)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help="The number of workers to use in the test_lstm dataloader.")


def add_standard_train_args(parser):
    """Helper to add the standard arguments used for training.

    """
    add_standard_args(parser)
    parser.add_argument('--lr', type=float, required=True,
                        help="The learning rate of the optimizer.")
    parser.add_argument('--lstm_dropout', type=float, required=False, default=0.0,
                        help="Probability of dropout in all-but-last-layer of LSTM.")
    parser.add_argument('--lstm_model0', type=MyArgumentTypes.filenametype, required=False,
                        help="An initial model for a LSTM, basically defines our NN topology. "
                        "Required unless doing tuning, where we create the LSTM dynamically.")
    parser.add_argument('--weight_decay', type=float, required=True,
                        help="Weight decay in the Adam optimizer (L2 penalty).")
    parser.add_argument('--early_stop_patience', type=int, required=False, default=DEFAULT_EARLY_STOP_PATIENCE,
                        help="Max num. iterations to go without improvement before stopping.")
    parser.add_argument('--max_num_iters', type=int, required=False,
                        default=DEFAULT_MAX_NUM_ITERS,
                        help="Maximum number of iterations to train for.")
    parser.add_argument('--dont_delete_old_models', action="store_true",
                        help="If given, keep old models, else delete models from previous iters.")
    parser.add_argument('--save_all_models', action="store_true",
                        help="If given, save all models, else only save improved ones.")
    parser.add_argument('--model_save_period', type=int, required=False, default=1,
                        help="How often to save the model to disk.")
    parser.add_argument('--test_warmup_period', type=int, required=False, default=0,
                        help="How many initial epochs to wait before running testing.")
    parser.add_argument('--test_eval_period', type=int, required=False, default=1,
                        help="How often to run testing, by iter (only applies if testing).")
    parser.add_argument('--plot_period', type=int, required=False, default=1,
                        help="How often to plot what you have to disk.")
    parser.add_argument('--out_dir', type=MyArgumentTypes.outdirtype, required=True,
                        help="Store plots, and other output in this DIR if set.")
