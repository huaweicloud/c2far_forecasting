"""Helper class to track the loss statistics as we go through
training.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_stats.py
import logging
import re
import torch
logger = logging.getLogger("sutranets.ml.loss_stats")
NO_LOSS_FLAG = torch.tensor(-1.0)


class LossStats():
    """Hold, and reset as needed, the loss stats, during training or
    testing.

    """
    def __init__(self, criterion):
        self.name = str(criterion)
        self.name = re.sub(r"<function ", "", self.name)
        self.name = re.sub(r" at 0x.*", "", self.name)
        self.name = re.sub(r"\(\)", "", self.name)
        self.tot_loss = 0
        self.tot_examples = 0
        self.run_tot_loss = 0
        self.run_tot_examples = 0

    def get_name(self):
        return self.name

    def update(self, num, loss):
        """Given we've processed [num] examples, and observed a loss of
        [loss], update our totals.

        """
        if num == 0:
            return 0
        if loss.isnan():
            logger.warning("nan encountered, skipping batch in loss update")
            return 0
        self.tot_loss += loss * num
        self.tot_examples += num
        self.run_tot_loss += loss * num
        self.run_tot_examples += num
        return num

    def get_tot_examples(self):
        return self.tot_examples

    def get_run_tot_examples(self):
        return self.run_tot_examples

    def overall_loss(self):
        """Calculate and return the overall loss, i.e., the loss accumulated
        across every time we've ever called `update`.

        """
        if not self.tot_examples:
            logger.warning(
                "No examples seen for metric, returning %s", NO_LOSS_FLAG)
            return NO_LOSS_FLAG
        return self.tot_loss / self.tot_examples

    def running_loss(self):
        """Calculate and return the running loss, i.e., the loss since the
        last time we called `reset_running`, i.e., the loss since the
        last time we printed the results (e.g., REPORT=50, so after 50
        batches).

        """
        if self.run_tot_examples == 0:
            logger.debug("No values since last report, returning %s", NO_LOSS_FLAG)
            return NO_LOSS_FLAG
        return self.run_tot_loss / self.run_tot_examples

    def reset_running(self):
        """Set the running totals back to zero."""
        self.run_tot_loss = 0
        self.run_tot_examples = 0
