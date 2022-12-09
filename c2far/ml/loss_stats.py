"""Helper class to track the loss statistics as we go through
training.  These stats include the loss that we've calculated since
the last time we reported (e.g., on the current 50 batches, the
"running total") and the overall loss across all batches ("overall
loss").

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_stats.py

import logging
import re
import torch
logger = logging.getLogger("c2far.ml.loss_stats")
NO_LOSS_FLAG = torch.tensor(-1.0)


class LossStats():
    """A class to hold, and reset as needed, the loss stats, during
    training or testing.

    """
    def __init__(self, criterion):
        """Initialize all our running totals to zero.

        Arguments:

        criterion: function or callable object, that we convert to a
        name.

        """
        self.name = str(criterion)
        # Remove the decoration, if we have a function name here:
        self.name = re.sub(r"<function ", "", self.name)
        self.name = re.sub(r" at 0x.*", "", self.name)
        # Or an object name:
        self.name = re.sub(r"\(\)", "", self.name)
        # Init running totals to zero:
        self.tot_loss = 0
        self.tot_examples = 0
        self.run_tot_loss = 0
        self.run_tot_examples = 0

    def get_name(self):
        """Simple getter for name"""
        return self.name

    def update(self, num, loss):
        """Given we've processed [num] examples, and observed a loss of
        [loss], update our totals.  Losses functions are averaged over
        their inputs, so we 'de-average' here by multiplying by 'num',
        and then we add that into the total, but we keep track of the
        running tot_examples so we can convert back to an average
        later.

        """
        if num == 0:
            return
        if loss.isnan():
            logger.warning("nan encountered, skipping batch in loss update")
            return
        self.tot_loss += loss * num
        self.tot_examples += num
        self.run_tot_loss += loss * num
        self.run_tot_examples += num

    def get_tot_examples(self):
        """Return the total number of examples processed since the beginning.

        """
        return self.tot_examples

    def get_run_tot_examples(self):
        """Return the total number of examples processed this run."""
        return self.run_tot_examples

    def overall_loss(self):
        """Calculate and return the overall loss, i.e., the loss accumulated
        across every time we've ever called `update`.

        """
        if not self.tot_examples:
            logger.warning(
                "No examples seen for metric, returning %s", NO_LOSS_FLAG)
            return NO_LOSS_FLAG
        overall_loss = None
        try:
            overall_loss = self.tot_loss / self.tot_examples
        except ZeroDivisionError as excep:
            logger.error("tot_examples can't be zero.")
            raise excep
        return overall_loss

    def running_loss(self):
        """Calculate and return the running loss, i.e., the loss since the
        last time we called `reset_running`, i.e., the loss since the
        last time we printed the results (e.g., REPORT=50, so after 50
        batches).

        """
        if self.run_tot_examples == 0:
            logger.debug("No values since last report, returning %s", NO_LOSS_FLAG)
            return NO_LOSS_FLAG
        running_loss = None
        try:
            running_loss = self.run_tot_loss / self.run_tot_examples
        except ZeroDivisionError as excep:
            logger.error("run_tot_examples can't be zero.")
            raise excep
        return running_loss

    def reset_running(self):
        """Set the running totals back to zero."""
        self.run_tot_loss = 0
        self.run_tot_examples = 0
