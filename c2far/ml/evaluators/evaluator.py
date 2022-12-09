"""ABC for any kind of evaluation class (baselines, etc.).

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/evaluators/evaluator.py

from abc import ABC, abstractmethod
import logging
import torch
from c2far.ml.constants import ExampleKeys
from c2far.ml.loss_functions.constants import IGNORE_INDEX
from c2far.ml.loss_stats import LossStats

logger = logging.getLogger("c2far.ml.evaluator")
REPORT = 50


class Evaluator(ABC):
    """Class that evaluates by calling batch forward on a testloader.

    """
    LABEL = "BaseEvaluator (overwrite this)"

    def __init__(self, device):
        """All our evaluators require a device."""
        self.device = device

    def _move_to_device(self, *tensor_lst):
        """Helper to move the tensors to the GPU.  Helpfully checks if the
        tensor is None first.

        """
        out_lst = []
        for tensor in tensor_lst:
            if tensor is not None:
                tensor = tensor.to(self.device)
            out_lst.append(tensor)
        return out_lst

    @abstractmethod
    def _make_outputs(self, inputs, targets, originals):
        """This will depend on the specific strategy you are using."""

    def batch_forward_outputs(self, inputs, targets, originals,
                              *varargs):
        """Entry point to doing the batch-forward pass but without computing
        the loss yet.  We can use this directly in cases where we
        don't have targets (e.g. in generation mode).  The inputs,
        targets, and originals are used to make the outputs, while any
        varargs passed will just be moved to the device.

        Arguments:

        inputs: Tensor[NSEQ x NBATCH x NINPUT]

        targets: Tensor[NSEQ x NBATCH x 1]

        originals: Tensor[NSEQ+2 x NBATCH x 1]

        *varargs - might include:

        coarses: Tensor[NSEQ+2 x NBATCH x 1]

        fines: Tensor[NSEQ+2 x NBATCH x 1]

        fines2: Tensor[NSEQ+2 x NBATCH x 1]

        Returns:

        outputs: Tensor[NSEQ x NBATCH x NOUTPUT]

        targets: Tensor[NSEQ x NBATCH x 1]

        originals: Tensor[NSEQ+2 x NBATCH x 1]

        a tensor for each input in *varargs

        """
        inputs, targets, originals, *others = self._move_to_device(
            inputs, targets, originals, *varargs)
        # Now, make the outputs according to whatever strategy has
        # been implemented in the subclass:
        outputs = self._make_outputs(inputs, targets, originals)
        retvals = outputs, targets, originals, *others
        return retvals

    def batch_forward(self, batch, criterions):
        """Get the losses on the given batch, based on comparing the outputs
        to the targets, using the given criterions.

        Arguments:

        batch: Dict[ExampleKeys -> inputs/targets/etc.]

        In particular: inputs of NSEQ x NBATCH x NFEATS, and targets
        of NSEQ x NBATCH x 1.

        criterions: List[criterion]: functions (or callables) that can
        be used to calculate the loss.

        Returns:

        nums, losses: List[Int], List[Float], the losses for each
        criterion, and how many examples processed for each.

        """
        inputs = batch[ExampleKeys.INPUT]
        targets = batch[ExampleKeys.TARGET]
        originals = batch.get(ExampleKeys.ORIGINALS)
        coarses = batch.get(ExampleKeys.COARSES)
        outputs, targets, originals, coarses = self.batch_forward_outputs(
            inputs, targets, originals, coarses)
        losses = []
        nums = []
        for crit_num, criterion in enumerate(criterions):
            new_outputs = outputs.reshape(-1, outputs.shape[-1])
            if targets.shape[-1] == 1:  # coarse targets
                new_targets = targets.reshape(-1)
            else:  # coarse, fine, and maybe fine2 bin targets
                new_targets = targets.reshape(-1, targets.shape[-1])
            try:
                if crit_num != 0:
                    with torch.no_grad():  # only backprop on first
                        loss = criterion(new_outputs, new_targets)
                else:
                    loss = criterion(new_outputs, new_targets)
                # How many do we process - all those we didn't ignore
                # (of coarse bins):
                num = (targets[:, :, 0] != IGNORE_INDEX).sum()
            except TypeError:
                # It's one of our special losses, which may compute
                # 'num' dynamically (e.g. only for certain horizons):
                if crit_num != 0:
                    with torch.no_grad():  # only backprop on first
                        loss, num = criterion(
                            outputs, targets, originals, coarses)
                else:
                    loss, num = criterion(outputs, targets, originals, coarses)
            losses.append(loss)
            nums.append(num)
        return nums, losses

    @staticmethod
    def __log_print_loss(batch_num, loss_stats, label):
        """Helper to print the full loss line."""
        # Let's do it in two statements, just so it looks nicer:
        logger.info(label)
        logger.info(
            '[%5d, %7d, %7d]: overall loss %.5f, running loss: %.5f',
            batch_num, loss_stats.get_tot_examples(),
            loss_stats.get_run_tot_examples(),
            loss_stats.overall_loss(), loss_stats.running_loss())

    @classmethod
    def print_loss_stats(cls, epoch_num, batch_num, loss_stat_lst,
                         dataset_name):
        """Helper functon to print the loss stats for all the loss_stats
        objects in the list.

        dataset_name: String, which dataset we're printing loss for.

        """
        for loss_stats in loss_stat_lst:
            loss_criterion = loss_stats.get_name()
            if epoch_num is None:
                label = f"{dataset_name}-{loss_criterion}"
            else:
                label = f"{dataset_name}-{epoch_num}-{loss_criterion}"
            cls.__log_print_loss(batch_num, loss_stats, label)
            loss_stats.reset_running()

    @classmethod
    def update_loss_stats(cls, loss_stat_lst, nums, losses, dataset,
                          batch_num, epoch_num):
        """Little helper to take care of updating and reporting the losses,
        whether training or testing.

        """
        for loss_stats, num, loss in zip(loss_stat_lst, nums, losses):
            if num > 0:
                loss_stats.update(num, loss)
        if batch_num % REPORT == 0:
            cls.print_loss_stats(epoch_num, batch_num, loss_stat_lst,
                                 dataset)

    def get_test_score(self, criterions, testloader, *, epoch_num=None):
        """Get the scores of the given evaluator on the test set.

        Arguments:

        criterions: which loss functions to compute

        testloaders: where to draw our test batches

        epoch_num: Int, in training: which epoch we are on (i.e., even
        when reporting test results), but in pure testing/evaluation,
        just leave as None.

        """
        loss_stat_lst = [LossStats(c) for c in criterions]
        batch_num = 0  # Avoid pylint errors below
        for batch_num, batch in enumerate(testloader, 1):
            nums, losses = self.batch_forward(batch, criterions)
            self.update_loss_stats(loss_stat_lst, nums, losses, "Test",
                                   batch_num, epoch_num)
        # Print final one - which may be empty, if you divided evenly:
        self.print_loss_stats(epoch_num, batch_num, loss_stat_lst,
                              "Test")
        # An extra message just to emphasize the overall loss on the
        # first loss criterion (which is the main one):
        overall_loss = loss_stat_lst[0].overall_loss()
        if epoch_num is None:
            logger.info('Final test loss: %.9f', overall_loss)
        else:
            logger.info('Test-loss-epoch-%d: %.9f', epoch_num,
                        overall_loss)
        return overall_loss
