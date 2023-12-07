"""ABC for any kind of evaluation class (baselines, etc.).

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/evaluators/evaluator.py
from abc import ABC, abstractmethod
import logging
import torch
from sutranets.ml.constants import ExampleKeys
from sutranets.ml.loss_functions.constants import IGNORE_INDEX
from sutranets.ml.loss_stats import LossStats
logger = logging.getLogger("sutranets.ml.evaluator")
REPORT = 50


class Evaluator(ABC):
    """Class that evaluates by calling batch forward on a testloader.

    """
    LABEL = "BaseEvaluator (overwrite this)"

    def __init__(self, device):
        self.device = device

    def _move_to_device(self, *tensor_lst):
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
        the loss yet.

        """
        inputs, targets, originals, *others = self._move_to_device(
            inputs, targets, originals, *varargs)
        outputs = self._make_outputs(inputs, targets, originals)
        retvals = outputs, targets, originals, *others
        return retvals

    @staticmethod
    def _is_flat(criterion):
        """Helper to quickly check if we should use the flattened
        outputs/targets or the ones with a batch dimension.

        """
        flat = True
        try:
            flat = not criterion.NONFLAT
        except AttributeError:
            pass
        return flat

    @staticmethod
    def flatten_tensors(outputs, targets):
        """Utility to shape the batches for loss calculations for losses that
        expect flat tensors.

        """
        flat_outputs = outputs.reshape(-1, outputs.shape[-1])
        if targets.shape[-1] == 1:
            flat_targets = targets.reshape(-1)
        else:
            flat_targets = targets.reshape(-1, targets.shape[-1])
        return flat_outputs, flat_targets

    def batch_forward(self, batch, criterions):
        """Get the losses on the given batch, based on comparing the outputs
        to the targets, using the given criterions.

        """
        inputs = batch.get(ExampleKeys.INPUT)
        targets = batch[ExampleKeys.TARGET]
        originals = batch.get(ExampleKeys.ORIGINALS)
        coarses = batch.get(ExampleKeys.COARSES)
        outputs, targets, originals, coarses = self.batch_forward_outputs(
            inputs, targets, originals, coarses)
        losses = []
        nums = []
        flat_outputs, flat_targets = None, None
        for crit_num, criterion in enumerate(criterions):
            if self._is_flat(criterion):
                if flat_outputs is None:
                    flat_outputs, flat_targets = self.flatten_tensors(outputs, targets)
                if crit_num != 0:
                    with torch.no_grad():
                        loss = criterion(flat_outputs, flat_targets)
                else:
                    loss = criterion(flat_outputs, flat_targets)
                num = (targets[:, :, 0] != IGNORE_INDEX).sum()
            else:
                if crit_num != 0:
                    with torch.no_grad():
                        loss, num = criterion(outputs, targets, originals, coarses)
                else:
                    loss, num = criterion(outputs, targets, originals, coarses)
            losses.append(loss)
            nums.append(num)
        return nums, losses

    @staticmethod
    def __log_print_loss(batch_num, loss_stats, label):
        """Helper to print the full loss line."""
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
        """Helper to take care of updating and reporting the losses, whether
        training or testing.

        """
        for loss_stats, num, loss in zip(loss_stat_lst, nums, losses):
            num_updated = loss_stats.update(num, loss)
            if num and not num_updated:
                logger.error("Could not update %s, num=%s, loss=%s",
                             loss_stats.get_name(), num, loss)
        if batch_num % REPORT == 0:
            cls.print_loss_stats(epoch_num, batch_num, loss_stat_lst,
                                 dataset)

    def get_test_score(self, criterions, testloader, *, epoch_num=None):
        """Get the scores of the given evaluator on the test set.

        """
        loss_stat_lst = [LossStats(c) for c in criterions]
        batch_num = 0
        for batch_num, batch in enumerate(testloader, 1):
            if batch is None:
                continue
            nums, losses = self.batch_forward(batch, criterions)
            self.update_loss_stats(loss_stat_lst, nums, losses, "Test",
                                   batch_num, epoch_num)
        self.print_loss_stats(epoch_num, batch_num, loss_stat_lst, "Test")
        overall_loss = loss_stat_lst[0].overall_loss()
        if epoch_num is None:
            logger.info('Final test loss: %.9f', overall_loss)
        else:
            logger.info('Test-loss-epoch-%d: %.9f', epoch_num,
                        overall_loss)
        return overall_loss

    def __str__(self):
        """Unless we override this, each evaluator will print itself as its label."""
        return self.LABEL
