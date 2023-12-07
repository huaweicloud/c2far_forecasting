"""Helper class to plot the progress of the training.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/train_plotter.py
import time
import logging
import numpy as np
from numpy import convolve, ones
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from sutranets.plot_utils import PlotUtils
FIGSIZE = (16, 9)
CRV_MARKER = '.'
CRV_LW = 1
logger = logging.getLogger("sutranets.ml.train_plotter")


def smooth_curve(smoothing, curves_dict):
    """Replace the curve with a moving-average, of period 'smoothing', for
    every series in the curves dict, and return this new dict.

    """
    if smoothing is None:
        return curves_dict
    smoothed_dict = {}
    for name, scores in curves_dict.items():
        smoothed_scores = convolve(scores, ones((smoothing,)) /
                                   smoothing, mode='valid')
        smoothed_dict[name] = smoothed_scores
    return smoothed_dict


class TrainPlotter():
    """Utility functions for plotting the progress of training."""

    @staticmethod
    def __detach_scores(scores):
        """Convert cuda/cpu tensors to numpy.

        """
        new_scores = []
        for score in scores:
            if score is not None and isinstance(score, torch.Tensor):
                score = score.detach()
                if score.is_cuda:
                    score = score.cpu()
            new_scores.append(score)
        return new_scores

    @classmethod
    def _plot_trend(cls, scores, label, pdf):
        """Plot an individual learning curve to our running PDF.

        """
        def finalize_trend_plot(ax, fig, pdf):
            """Add commonalities to trend plots"""
            plt.minorticks_on()
            ax.tick_params(right=True, labelright=True, which='minor')
            ax.tick_params(right=True, labelright=True, which='major')
            ax.grid(True, axis='y', alpha=0.4, zorder=5)
            pdf.savefig(fig, dpi=80)
            plt.close()
        scores = cls.__detach_scores(scores)
        np_scores = np.array(scores).astype(np.double)
        score_mask = np.isfinite(np_scores)
        if not np.any(score_mask):
            return
        fig, ax = plt.subplots(figsize=FIGSIZE)
        xvals = np.arange(1, len(scores) + 1)
        plt.plot(xvals[score_mask], np_scores[score_mask], lw=CRV_LW,
                 marker=CRV_MARKER)
        plt.suptitle(f'{label} progress by iteration')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        finalize_trend_plot(ax, fig, pdf)
        if np.all(score_mask):
            for smoothing in [10, 50]:
                if len(scores) <= smoothing:
                    continue
                fig, ax = plt.subplots(figsize=FIGSIZE)
                curve_dict = {'CRV': scores}
                smoothed_dict = smooth_curve(smoothing, curve_dict)
                smoothed_scores = smoothed_dict['CRV']
                xvals = np.arange(1, len(smoothed_scores) + 1)
                plt.plot(xvals, smoothed_scores, lw=CRV_LW)
                plt.suptitle(
                    f'{label} progress with smoothing={smoothing} iterations')
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Smoothed score")
                finalize_trend_plot(ax, fig, pdf)

    @staticmethod
    def do_plots(plot_path, net, optimizer, scheduler, eval_scores, starttime):
        """Make a PDF in order to get insights on the output.  The net and
        optimizer are just for printing as strings.

        """
        pdf = PdfPages(plot_path)
        title_txt = "pytorch_train_lstm plots\n"
        title_txt += f"Net: {net}\n"
        title_txt += f"Optimizer: {optimizer}\n"
        title_txt += f"Scheduler: {scheduler}\n"
        for part in plot_path.split("/"):
            title_txt += part + "\n"
        elapsed = time.time() - starttime
        title_txt += f"Running for {elapsed:.1f} seconds"
        PlotUtils.make_pdf_page(title_txt, pdf)
        logger.info("Plotting training curve.")
        for series_name, scores in eval_scores.items():
            TrainPlotter._plot_trend(scores, series_name, pdf)
        pdf.close()
