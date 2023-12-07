"""Read in a set of training and testing data, an initial model, and
an outdir.  Turn the input data into 'Datasets', run training and
evaluate as you go.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/train_lstm.py
import argparse
from collections import defaultdict
import logging
import os
import socket
import time
from typing import NamedTuple
import torch
from sutranets.dataprep.cutoffs.create_cutoffs import setup_cutoffs
from sutranets.dataprep.cutoffs.utils import save_cutoffs
from sutranets.file_utils import setup_outdir
from sutranets.ml.add_args import add_standard_train_args, add_gen_eval_args
from sutranets.ml.dataset_utils import DatasetUtils
from sutranets.ml.eval_utils import EvalUtils
from sutranets.ml.loss_stats import LossStats
from sutranets.ml.make_criterions import MakeCriterions
from sutranets.ml.train_plotter import TrainPlotter
from sutranets.ml.utils import load_lstm_model, make_lstm_evaluator
from sutranets.utils import (init_console_logger, init_file_logger,
                             cross_arg_checker, set_all_seeds)
logger = logging.getLogger("sutranets.ml.train_lstm")
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
    batch_subset_frac: float
    dont_run_1step_eval: bool
    run_gen_eval: bool


class TrainLSTM():
    """Overall class to handle the training.

    """
    __PT_NN_FN = "model"
    __PLOT_FN = "plots.pdf"
    __TRAIN_SCORES_STR = "train_scores"
    __TEST_SCORES_STR = "test_scores"
    __DEFAULT_LR_SCHED_GAMMA = 0.99

    def __init__(self, net, device, train_args, *, trainloader,
                 testloader, target_dir, coarse_cutoffs, args):
        self.net = net
        self.device = device
        self.train_args = train_args
        self.trainloader = trainloader
        self.testloader = testloader
        self.target_dir = target_dir
        self.__validate()
        self.lstm_eval = make_lstm_evaluator(
            net, device, args.nfine_bins, args.nfine2_bins,
            args.nsub_series, do_init_hidden=True)
        if train_args.lstm_dropout > 0:
            self.net.set_dropout(train_args.lstm_dropout)
        if self.train_args.run_gen_eval:
            self.gen_eval = EvalUtils.create_gen_eval(
                self.lstm_eval, coarse_cutoffs, args)
            self.gen_loss = MakeCriterions.create_gen_criterions(
                args.confidence_pct, do_horizs=False, gsize=args.gsize,
                nsub_series=args.nsub_series)[0]

    def __validate(self):
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

        """
        optimizer.zero_grad()
        num, losses = self.lstm_eval.batch_forward(data, criterions)
        loss = losses[0]
        loss.backward()
        optimizer.step()
        return num, losses

    def __iterate_models(self, optimizer, scheduler, criterions):
        """Run a single training epoch and yield the model"""
        for epoch_num in range(1, self.train_args.max_num_iters + 1):
            self.net.train()
            logger.info("Training epoch %d", epoch_num)
            loss_stat_lst = [LossStats(str(c)) for c in criterions]
            batch_num = 0
            ntarget_batches = (len(self.trainloader) *
                               self.train_args.batch_subset_frac)
            for batch_num, batch in enumerate(self.trainloader, 1):
                if batch_num > ntarget_batches:
                    break
                if batch is None:
                    continue
                num, losses = self.__run_train_batch(
                    batch, optimizer, criterions)
                self.lstm_eval.update_loss_stats(loss_stat_lst, num, losses,
                                                 "Train", batch_num, epoch_num)
            self.lstm_eval.print_loss_stats(epoch_num, batch_num, loss_stat_lst,
                                            "Train")
            overall_loss = loss_stat_lst[0].overall_loss()
            self.trainloader.dataset.advance_next_epoch()
            scheduler.step()
            yield epoch_num, overall_loss

    def __replace_nn_model(self, model_fn_base, epoch_num, prev_epoch):
        """Replace the model on disk, if it exists, with the latest model - if
        we have a target_dir.

        """
        new_nn_fn = None
        if self.target_dir is not None:
            pt_nn_stem = os.path.join(self.target_dir, model_fn_base)
            new_nn_fn = pt_nn_stem + "." + str(epoch_num) + ".pt"
            new_nn_pdf = pt_nn_stem + "." + str(epoch_num) + ".pdf"
            self.net.save_python(new_nn_fn)
            self.net.save_viz(new_nn_pdf)
            if (not self.train_args.dont_delete_old_models) and prev_epoch is not None:
                old_nn_fn = pt_nn_stem + "." + str(prev_epoch) + ".pt"
                old_nn_pdf = pt_nn_stem + "." + str(prev_epoch) + ".pdf"
                self.net.delete_model(old_nn_fn)
                self.net.delete_pdf(old_nn_pdf)
        return new_nn_fn

    def __test_go_time(self, epoch_num):
        """All the logic for whether we test or not on the given epoch.

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
        """Update the training eval scores, and also run the model on the test
        set and update that score.

        """
        eval_scores[self.__TRAIN_SCORES_STR].append(train_loss)
        tracked_score = None
        if self.__test_go_time(epoch_num):
            logger.info("Testing epoch %d", epoch_num)
            self.net.eval()
            with torch.no_grad():
                if not self.train_args.dont_run_1step_eval:
                    tracked_score = self.lstm_eval.get_test_score(
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
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.train_args.learn_rate,
                                     weight_decay=self.train_args.weight_decay)
        logger.info("Optimizer: %s", optimizer)
        return optimizer

    @classmethod
    def __setup_lr_scheduler(cls, optimizer):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cls.__DEFAULT_LR_SCHED_GAMMA)
        logger.info("LR Scheduler: %s with gamma %f", scheduler,
                    cls.__DEFAULT_LR_SCHED_GAMMA)
        return scheduler

    @staticmethod
    def __update_time(prevtime, epoch_num):
        """Do the time tracking and status messaging."""
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
        if (self.train_args.save_all_models or set_new_best) and \
           (epoch_num % self.train_args.model_save_period == 0):
            self.__replace_nn_model(
                self.__PT_NN_FN, epoch_num, prev_model_save_epoch)
            prev_model_save_epoch = epoch_num
        return best_test, prev_model_save_epoch

    def __do_early_stop(self, eval_scores):
        """Check whether the current trial should be stopped early due to lack
        of progress.  Progress means reaching LOWER scores.

        """
        scores = [t.item() for t in eval_scores[self.__TEST_SCORES_STR] if t is not None]
        patience = self.train_args.early_stop_patience
        if len(scores) < (patience + 1):
            return False
        test_val = scores[-(patience + 1)]
        if any(s < test_val for s in scores[-patience:]):
            return False
        return True

    def __call__(self, train_criterions, test_criterions):
        """Run training on the given neural network.  Evaluate using the given
        train and test criterions (extra generation-based criterions
        may also be evaluated inside update_scores, if doing).

        """
        self.__setup_net_device()
        optimizer = self.__setup_optimizer()
        scheduler = self.__setup_lr_scheduler(optimizer)
        eval_scores = defaultdict(list)
        if self.target_dir is not None:
            plot_path = os.path.join(self.target_dir, self.__PLOT_FN)
        logger.info("Starting training")
        starttime = time.time()
        prevtime = starttime
        best_test = None
        prev_model_save_epoch = None
        for epoch_num, train_loss in self.__iterate_models(
                optimizer, scheduler, train_criterions):
            test_score = self.__update_scores(epoch_num, train_loss,
                                              eval_scores, test_criterions)
            best_test, prev_model_save_epoch = self.__update_models(
                epoch_num, prev_model_save_epoch, test_score, best_test)
            if (self.target_dir is not None and epoch_num %
                    self.train_args.plot_period == 0):
                TrainPlotter.do_plots(plot_path, self.net, optimizer,
                                      scheduler, eval_scores, starttime)
            prevtime = self.__update_time(prevtime, epoch_num)
            if self.__do_early_stop(eval_scores):
                logger.info("Stopping early due to lack of progress.")
                break
        logger.info("Finished training")
        return best_test


def create_sanitized_str(my_str):
    """Turn the given string into a string appropriate for a filename or a
    dirname by replacing any problematic characters with
    REPLACE_CHAR.

    """
    return "".join([x if x.isalnum() else REPLACE_CHAR for x in
                    str(my_str)])


def create_outdir(args, criterion):
    """Make the output directory (output_dir).

    """
    criterion_str = create_sanitized_str(criterion)
    subdir_parts = []
    subdir_parts.append("values")
    subdir_parts.append(criterion_str)
    if "lstm_model0" in args and args.lstm_model0 is not None:
        model_str = create_sanitized_str(args.lstm_model0)
    else:
        model_str = f"nlay={args.nlayers}.nhid={args.nhidden}"
    subdir_parts.append(model_str)
    subdir_parts.append(os.path.basename(args.train_offs))
    outdir = os.path.join(args.out_dir, *subdir_parts)
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


def do_train_lstm(args, net, logger_levels, *, cutoffs=None):
    """Do the actual training.

    """
    set_all_seeds(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(args.device)
    coarse_cutoffs = setup_cutoffs(cutoffs, args)
    train_criterions, test_criterions = MakeCriterions.make_next_step_train(args, coarse_cutoffs)
    logger.info("Reading in train/test data for evaluation.")
    trainloader, testloader = DatasetUtils.make_dataloaders(args, coarse_cutoffs)
    target_dir = setup_training(args, logger_levels, train_criterions[0])
    save_cutoffs(target_dir, coarse_cutoffs)
    train_args = TrainArgs(args.lr, args.weight_decay, args.lstm_dropout, args.model_save_period,
                           args.test_warmup_period, args.test_eval_period, args.plot_period,
                           args.early_stop_patience, args.max_num_iters, args.dont_delete_old_models,
                           args.save_all_models, args.batch_subset_frac, args.dont_run_1step_eval,
                           args.run_gen_eval)
    train_run = TrainLSTM(net, device, train_args, trainloader=trainloader, testloader=testloader,
                          target_dir=target_dir, coarse_cutoffs=coarse_cutoffs,
                          args=args)
    return train_run(train_criterions, test_criterions)


def main(args):
    logger_levels = [("sutranets", logging.INFO)]
    init_console_logger(logger_levels)
    if args.lstm_model0 is None:
        msg = "lstm_model0 required when using train_lstm directly."
        raise RuntimeError(msg)
    net = load_lstm_model(args.nfine_bins, args.nfine2_bins, args.lstm_model0,
                          args.nsub_series, args.device)
    logger.info("Initial net: %s", str(net))
    if args.device.find("cuda") >= 0:
        with torch.cuda.device(args.device):
            final_score = do_train_lstm(args, net, logger_levels)
    else:
        final_score = do_train_lstm(args, net, logger_levels)
    return final_score


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="Training the PyTorch-Based Forecasting Model")
    add_standard_train_args(parser)
    add_gen_eval_args(parser)
    args = parser.parse_args(args)
    cross_arg_checker(args, parser, check_stride=True, check_multiv=True)
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
