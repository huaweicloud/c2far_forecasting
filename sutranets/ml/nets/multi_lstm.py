"""Base class for a long-short-term memory neural network that has
multiple internal LSTM models.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/nets/multi_lstm.py
from abc import abstractmethod
from shutil import rmtree
from torch import nn
from sutranets.ml.nets.utils import plot_nn_model


class MultiLSTM(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.nlayers = models[0].nlayers

    def yield_models(self):
        """Iterator over our internal models, from highest frequency to lowest
        frequency.

        """
        for model in self.models:
            yield model

    def run_init_hidden(self, device, batch_size):
        """Initialize the component LSTM's hidden states. I.e., before doing
        each new example, we typically re-init the hidden state to
        zero.

        """
        for model in self.models:
            model.run_init_hidden(device=device, batch_size=batch_size)

    def set_dropout(self, dropout_val):
        """Helper to set the dropout on both the coarse and the fine LSTMs."""
        for model in self.models:
            model.set_dropout(dropout_val)

    @abstractmethod
    def forward(self, minibatch):
        """Pass in a minibatch of training examples, then slice into the
        inputs for each sub-series and run the corresponding forward
        pass on each.

        """

    @staticmethod
    def delete_model(path):
        """Given the path to the model, delete it (and all its submodels).

        """
        rmtree(path)

    @staticmethod
    def delete_pdf(path):
        """Given the path to the PDF with the images of the model, delete it
        (and all its submodel pictures).

        """
        rmtree(path)

    @abstractmethod
    def _get_model_paths(self, outpath, extension):
        """Each subclass can have its own way to get the model paths."""

    def save_python(self, outpath):
        """Here we save the component models to a DIR with name outpath.

        """
        for model, model_path in zip(self.models, self._get_model_paths(outpath, ".pt")):
            model.save_python(model_path)

    def save_viz(self, outpath):
        """Plot all the model parameters for each model and save the files as
        PDFs at outpath.

        """
        for model, model_path in zip(self.models, self._get_model_paths(outpath, ".pdf")):
            plot_nn_model(model, model_path)
