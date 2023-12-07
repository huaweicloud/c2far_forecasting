"""Utility functions that would be used by pytorch neural networks,
e.g., visualizing the network.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/nets/utils.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn
from sutranets.plot_utils import PlotUtils
FIGSIZE = (16, 9)


def plot_nn_model(net, outfn):
    """Save a vizualization of the network at the given fn."""
    pdf = PdfPages(outfn)
    tot_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    title_txt = f"{str(net)}\nNum params: {tot_params}"
    PlotUtils.make_pdf_page(title_txt, pdf)
    for name, param in net.named_parameters():
        fig, _ = plt.subplots(figsize=FIGSIZE)
        plt.title(name)
        if param.data.is_cuda:
            data = param.data.cpu().numpy()
        else:
            data = param.data.numpy()
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        try:
            plt.imshow(data, interpolation='none', aspect='auto')
        except TypeError:
            print(f"Can't output {name} for some reason.")
            print(data.shape)
        plt.colorbar()
        pdf.savefig(fig, dpi=80)
        plt.close()
    pdf.close()


def make_extrema_mlps(nextrema_subset, nhidden, extremas):
    """Create the two internal fully-connected neural networks that return
    the extreme low and extreme high parameters.

    """
    if extremas:
        net_low = nn.Sequential(nn.Linear(nextrema_subset, nhidden), nn.ReLU(), nn.Linear(nhidden, 1))
        net_high = nn.Sequential(nn.Linear(nextrema_subset, nhidden), nn.ReLU(), nn.Linear(nhidden, 1))
        return net_low, net_high
    return None, None
