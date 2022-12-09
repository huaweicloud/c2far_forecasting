"""Read in a set of training and testing data, an initial model, and
an outdir.  Turn the input data into 'Datasets', run training and
evaluate as you go.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/train_lstm.py

import argparse
from collections import defaultdict
import logging
import os
import socket
import time
from typing import NamedTuple
import torch
from c2far.dataprep.cutoffs.create_cutoffs import CreateCutoffs
from c2far.file_utils import setup_outdir
from c2far.ml.add_args import add_standard_train_args, add_gen_eval_args
from c2far.ml.evaluators.lstm_evaluator import TripleLSTMEvaluator
from c2far.ml.loss_stats import LossStats
from c2far.ml.lstm_evaluation import create_gen_eval
from c2far.ml.make_criterions import MakeCriterions
from c2far.ml.train_plotter import TrainPlotter
from c2far.ml.utils import make_dataloaders, load_lstm_model
from c2far.utils import (init_console_logger, init_file_logger,
                         cross_arg_checker, set_all_seeds)
logger = logging.getLogger("c2far.ml.train_lstm")
LOG_SHORT_FN = "log.txt"
REPLACE_CHAR = "_"


class TrainArgs(NamedTuple):
    """Arguments to be used in training."""
    learn_rate: float
    weight_decay: float
    lstm_dropout: float
    model_save_period: int
    test_warmup_period: int
    test_eval_period: int
    plot_period: int
    early_stop_patience: int
    max_num_iters: int
    dont_delete_old_models: bool
    save_all_models: bool
    dont_run_1step_eval: bool
    run_gen_eval: bool


class TrainLSTM(TripleLSTMEvaluator):
    """Overall class to handle the training.  It's an LSTMEvaluator in two
    senses: (1) it runs batch_forward on the training set (so we can
    re-use the batch_forward method from the Evaluator grandparent
    class), and (2) it calls 'get_test_score' on the TEST set, so we
    can re-use that method too, and (3) it leverages the make_outputs
    logic in LSTMEvaluator that we need for an LSTM (in the
    LSTMEvaluator class itself).  It's a "TripleLSTMEvaluator" so that
    it has some additional methods for when we're GENERATING
    coarse/fine or coarse/fine/fine2 jointly. But it still has all the
    methods that we need if we're doing non-joint or non-triple
    evaluation.

    When do we reset or init the hidden state (run_init_hidden)?
    Well, this class calls 'batch_forward' in the evaluator,
    batch_forward in the evaluator calls 'batch_forward_outputs',
    'batch_forward_outputs' calls _make_outputs inside LSTMEvaluator,
    and it's LSTMEvaluator that initializes the hidden state, if it's
    configured to do so.

    """
    __PT_NN_FN = "model"
    __PLOT_FN = "plots.pdf"
    __TRAIN_SCORES_STR = "train_scores"
    __TEST_SCORES_STR = "test_scores"

    def __init__(self, net, device, train_args, *, trainloader,
                 testloader, target_dir, coarse_cutoffs, args):
        """You need to provide the net, device, train_args, trainloader.
        Optionally the testloader and target_dir can be None.  The
        command-line args also come in here, to let us set up the
        generation evaluator - we must do that in here, since it
        actually uses `self` as an argument.

        Positional arguments:

        net, torch.nn.Module, that you can call batch_forward on, etc.

        device, where to run training/eval, e.g. "cpu" or "cuda:1"

        train_args: TrainArgs (named tuple), various arguments needed
        for training, see above.

        Keyword-only arguments:

        trainloader: torch.DataLoader, for drawing training batches.

        testloader: torch.DataLoader (or None), for drawing validation
        batches.

        target_dir: string (or None), where to store the models/plots.

        coarse_cutoffs: Tensor[Float], the bin boundaries for coarse input, output.

        args, CLI named-tuple args, which we need in order to create
        the generation evaluator, if we are doing that.

        """
        super().__init__(net, device, do_init_hidden=True)
        self.train_args = train_args
        self.trainloader = trainloader
        self.testloader = testloader
        self.target_dir = target_dir
        self.__validate()
        if train_args.lstm_dropout > 0:
            self.net.set_dropout(train_args.lstm_dropout)
        if self.train_args.run_gen_eval:
            # Must pass `self`: train_lstm *is* the lstm_evaluator:
            self.gen_eval = create_gen_eval(
                self, coarse_cutoffs, args)
            # Only use the first criterion, mae_loss:
            self.gen_loss = MakeCriterions.create_gen_criterions(
                args.confidence_pct)[0]

    def __validate(self):
        """Checks to make sure passed parameters make sense."""
        if self.train_args.lstm_dropout > 0:
            if self.net.nlayers == 1:
                raise RuntimeError("Dropout expects nlayers > 1")
        if self.train_args.max_num_iters < self.train_args.test_eval_period:
            raise RuntimeError("Cannot report best result if no testing.")

    def __run_train_batch(self, data, optimizer, criterions):
        """Given the net, device, a single set of input data/targets from the
        training set, an optimizer, and some loss criterions, run a
        single training step and return the number of inputs
        processed, and the loss.

        Compute loss.backward() on the loss from the first criterion.

        """
        optimizer.zero_grad()
        num, losses = self.batch_forward(data, criterions)
        loss = losses[0]
        loss.backward()
        optimizer.step()
        return num, losses

    def __iterate_models(self, optimizer, criterions):
        """Run a single training epoch and yield the model"""
        for epoch_num in range(1, self.train_args.max_num_iters + 1):
            self.net.train()
            logger.info("Training epoch %d", epoch_num)
            loss_stat_lst = [LossStats(str(c)) for c in criterions]
            batch_num = 0  # Avoid pylint errors
            for batch_num, batch in enumerate(self.trainloader, 1):
                num, losses = self.__run_train_batch(
                    batch, optimizer, criterions)
                self.update_loss_stats(loss_stat_lst, num, losses,
                                       "Train", batch_num, epoch_num)
            self.print_loss_stats(epoch_num, batch_num, loss_stat_lst,
                                  "Train")
            overall_loss = loss_stat_lst[0].overall_loss()
            self.trainloader.dataset.advance_next_epoch()
            yield epoch_num, overall_loss

    def __replace_nn_model(self, model_fn_base, epoch_num, prev_epoch):
        """Let's replace the model on disk, if it exists, with the latest
        model - if we have a target_dir.

        Arguments:

        model_fn_base: the desired basename for the model in the
        target_dir.

        epoch_num: Int, the current epoch number, which becomes part
        of the model filename.

        prev_epoch: Int, the epoch number for the last epoch where we
        saved the model.

        Returns:

        new_nn_fn, String: the path on disk to the model for this
        epoch, or None if no target_dir.

        """
        new_nn_fn = None
        if self.target_dir is not None:
            pt_nn_stem = os.path.join(self.target_dir, model_fn_base)
            new_nn_fn = pt_nn_stem + "." + str(epoch_num) + ".pt"
            self.net.save_python(new_nn_fn)
            if (not self.train_args.dont_delete_old_models) and prev_epoch is not None:
                old_nn_fn = pt_nn_stem + "." + str(prev_epoch) + ".pt"
                os.remove(old_nn_fn)
        return new_nn_fn

    def __test_go_time(self, epoch_num):
        """Helper to include all the logic for whether we test or not on the
        given epoch.

        Arguments:

        epoch_num: Int, the current epoch number, indexed starting at 1.

        Returns:

        Boolean, True to go, False to not go.

        """
        if self.testloader is None:
            return False
        if epoch_num % self.train_args.test_eval_period != 0:
            return False
        if epoch_num <= self.train_args.test_warmup_period:
            return False
        return True

    def __update_scores(self, epoch_num, train_loss, eval_scores,
                        criterions):
        """Helper function to update the training eval scores, and also run
        the model on the test set and update that score.

        Returns the score on the validation set that we are TRACKING,
        e.g., we compute many scores, but we only track one (most
        likely the gen_eval MAE score).

        """
        eval_scores[self.__TRAIN_SCORES_STR].append(train_loss)
        tracked_score = None
        if self.__test_go_time(epoch_num):
            logger.info("Testing epoch %d", epoch_num)
            self.net.eval()
            with torch.no_grad():
                if not self.train_args.dont_run_1step_eval:
                    tracked_score = self.get_test_score(
                        criterions, self.testloader, epoch_num=epoch_num)
                if self.train_args.run_gen_eval:
                    tracked_score = self.gen_eval.get_test_score(
                        [self.gen_loss], self.testloader, epoch_num=epoch_num)
        eval_scores[self.__TEST_SCORES_STR].append(tracked_score)
        return tracked_score

    def __setup_net_device(self):
        """Configure the net to run on the GPU or CPU, depending on what user
        asked.

        """
        self.net = self.net.to(self.device)

    def __setup_optimizer(self):
        """Quick helper to set up the optimizer."""
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.train_args.learn_rate,
                                     weight_decay=self.train_args.weight_decay)
        logger.info("Optimizer: %s", optimizer)
        return optimizer

    @staticmethod
    def __update_time(prevtime, epoch_num):
        """Little helper to do the time tracking and status messaging."""
        currtime = time.time()
        elapsed = currtime - prevtime
        status_msg = f"Finished epoch {epoch_num} in {elapsed:.1f} seconds"
        prevtime = currtime
        logger.info(status_msg)
        return prevtime

    def __update_models(self, epoch_num, prev_model_save_epoch,
                        test_score, best_test):
        """Write (or replace) the model on disk.

        """
        set_new_best = False
        if (test_score is not None) and (
                best_test is None or test_score < best_test):
            best_test = test_score
            set_new_best = True
        # Now, write latest model if it makes sense:
        if (self.train_args.save_all_models or set_new_best) and \
           (epoch_num % self.train_args.model_save_period == 0):
            self.__replace_nn_model(self.__PT_NN_FN, epoch_num,
                                    prev_model_save_epoch)
            # Next time, we'll remove this one (if removing):
            prev_model_save_epoch = epoch_num
        return best_test, prev_model_save_epoch

    def __do_early_stop(self, eval_scores):
        """Check whether the current trial should be stopped early due to lack
        of progress.  Progress means reaching LOWER scores.

        """
        # Get all OBSERVED/COMPUTED test scores as list of floats:
        scores = [t.item() for t in eval_scores[self.__TEST_SCORES_STR] if t is not None]
        patience = self.train_args.early_stop_patience
        if len(scores) < (patience + 1):
            return False
        # Otherwise, see if val from patience+1 ago not improved afterwards:
        test_val = scores[-(patience + 1)]
        if any(s < test_val for s in scores[-patience:]):
            return False
        return True

    def __call__(self, train_criterions, test_criterions):
        """Run training on the given neural network.

        """
        self.__setup_net_device()
        optimizer = self.__setup_optimizer()
        eval_scores = defaultdict(list)
        if self.target_dir is not None:
            plot_path = os.path.join(self.target_dir, self.__PLOT_FN)
        logger.info("Starting training")
        starttime = time.time()
        prevtime = starttime
        best_test = None
        # For keeping track of models we've written to disk, but
        # initially there are none:
        prev_model_save_epoch = None
        for epoch_num, train_loss in self.__iterate_models(
                optimizer, train_criterions):
            test_score = self.__update_scores(epoch_num, train_loss,
                                              eval_scores, test_criterions)
            best_test, prev_model_save_epoch = self.__update_models(
                epoch_num, prev_model_save_epoch, test_score, best_test)
            if (self.target_dir is not None and epoch_num %
                    self.train_args.plot_period == 0):
                TrainPlotter.do_plots(plot_path, self.net, optimizer,
                                      eval_scores, starttime)
            prevtime = self.__update_time(prevtime, epoch_num)
            if self.__do_early_stop(eval_scores):
                logger.info("Stopping early due to lack of progress.")
                break
        logger.info("Finished training")
        # Return the best validation score, for use in UTs and Optuna:
        return best_test


def create_sanitized_str(my_str):
    """Turn the given string into a string appropriate for a filename or a
    dirname by replacing any problematic characters with
    REPLACE_CHAR.

    """
    return "".join([x if x.isalnum() else REPLACE_CHAR for x in
                    str(my_str)])


def create_outdir(args, criterion):
    """Make the output directory (output_dir). To prevent us forgetting to
    mark each outdir with important info about what we were training,
    we use this helper to make the full outdir name automatically.

    """
    criterion_str = create_sanitized_str(criterion)
    # Make subdirs depending on some stuff:
    subdir_parts = []
    subdir_parts.append(criterion_str)
    model_str = create_sanitized_str(args.lstm_model0)
    subdir_parts.append(model_str)
    subdir_parts.append(os.path.basename(args.train_offs))
    outdir = os.path.join(args.out_dir, *subdir_parts)
    # And then this one adds a datestamp to it too:
    target_dir = setup_outdir(outdir)
    return target_dir


def setup_training(args, logger_levels, train_criterion):
    """Common parts to running different kinds of training.  Args are all
    the command-line args, logger_levels can be None if you don't want
    the file_logger, and the train_criterion is just for purposes of
    organizing our output DIRs.

    """
    target_dir = create_outdir(args, train_criterion)
    if logger_levels is not None:
        log_fn = os.path.join(target_dir, LOG_SHORT_FN)
        init_file_logger(logger_levels, log_fn)
        logger.info("Logging to %s", log_fn)
    logger.info("Outputting to: %s", target_dir)
    logger.info("ARGS: %s", args)
    logger.info("Host: %s", socket.gethostname())
    if args.device.find("cuda") >= 0:
        logger.info("Training on GPU: %s.", args.device)
    else:
        logger.info("Training on cpu.")
    return target_dir


def do_train_lstm(args, net, logger_levels):
    """Helper to do the actual training.

    Required Arguments:

    args: Argparser or other NamedTuple, with all the arguments you
    need for setting up criterions, the network, the datasets, and
    running the training.

    net: FCLSTM (or other network), the parameters that we are
    actually training.

    logger_levels: List(tuple) to setup a file logger to the output
    DIR, or None to not do logging.

    """
    set_all_seeds(args.seed)
    device = torch.device(args.device)
    if args.continuous:
        coarse_cutoffs = None
    else:
        coarse_cutoffs = CreateCutoffs.get_coarse_cutoffs(args)
    # Here's our NEXT-STEP losses, if doing (GENERATION losses done in TrainLSTM constructor):
    train_criterions, test_criterions = MakeCriterions.make_next_step_train(args, coarse_cutoffs)
    logger.info("Reading in train/test data for evaluation.")
    trainloader, testloader = make_dataloaders(args, coarse_cutoffs)
    target_dir = setup_training(args, logger_levels, train_criterions[0])
    train_args = TrainArgs(args.lr, args.weight_decay, args.lstm_dropout, args.model_save_period,
                           args.test_warmup_period, args.test_eval_period, args.plot_period,
                           args.early_stop_patience, args.max_num_iters, args.dont_delete_old_models,
                           args.save_all_models, args.dont_run_1step_eval, args.run_gen_eval)
    # FYI: Adding the multi-step evaluator is done in the constructor:
    train_run = TrainLSTM(net, device, train_args, trainloader=trainloader, testloader=testloader,
                          target_dir=target_dir, coarse_cutoffs=coarse_cutoffs, args=args)
    return train_run(train_criterions, test_criterions)


def main(args):
    """Given the command-line arguments, setup console logging, get the
    neural network, and call do_train_lstm to run the training.

    """
    logger_levels = [("c2far", logging.INFO)]
    init_console_logger(logger_levels)
    if args.lstm_model0 is None:
        msg = "lstm_model0 required when using train_lstm directly."
        raise RuntimeError(msg)
    net = load_lstm_model(args.nfine_bins, args.nfine2_bins, args.lstm_model0, args.device)
    logger.info("Initial net: %s", str(net))
    if args.device.find("cuda") >= 0:
        with torch.cuda.device(args.device):
            do_train_lstm(args, net, logger_levels)
    else:
        do_train_lstm(args, net, logger_levels)


def parse_arguments(args=None):
    """Helper function to parse the command-line arguments, return as an
    'args' object.  If args == None, will get args from sys.argv[1:].

    """
    parser = argparse.ArgumentParser(
        description="Training the PyTorch-Based Forecasting Model")
    add_standard_train_args(parser)
    add_gen_eval_args(parser)
    args = parser.parse_args(args)
    cross_arg_checker(args, parser, check_stride=True)
    return args


if __name__ == "__main__":  # pragma: no cover
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
