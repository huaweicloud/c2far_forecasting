"""Generic long-short-term memory neural network that outputs a vector
of logits.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/nets/fc_lstm.py
import os
import torch
from torch import nn
from sutranets.ml.bin_tensor_maker import BinTensorMaker
from sutranets.ml.constants import INPUT_DTYPE
from sutranets.ml.nets.utils import make_extrema_mlps, plot_nn_model


class FCLSTM(nn.Module):
    """Generalization of any NN that passes hidden state along using a
    LSTM, but then generates output values using a linear affine
    mapping from this hidden state, which we can interpret as
    logits, or as parameters to a Gaussian, or whatever.

    """

    def __init__(self, ninput, nhidden, noutput, nlayers, *, extremas=False):
        super().__init__()
        assert not isinstance(nhidden, list), "Hidden layer has fixed size"
        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nlayers = nlayers
        self.extremas = extremas
        nlstm_input, _, _, nextrema_input = BinTensorMaker.compute_nsubsets(
            ninput, None, None, None, extremas=extremas)
        if self.extremas:
            nlstm_output = noutput - 2
        else:
            nlstm_output = noutput
        self.lstm = nn.LSTM(nlstm_input, self.nhidden, self.nlayers)
        self.fc_out = nn.Linear(self.nhidden, nlstm_output)
        self.ex_mlp_low, self.ex_mlp_high = make_extrema_mlps(nextrema_input, self.nhidden, extremas)
        self.hidden = self.run_init_hidden(None, 1)

    def run_init_hidden(self, device, batch_size):
        """Initialize the LSTM hidden state, e.g., before doing each new
        example, we typically re-init the hidden state to zero.

        """
        self.hidden = self.__init_hidden(device, batch_size)
        return self.hidden

    def __init_hidden(self, device, batch_size):
        """Typically init hidden state like this:

        hidden = self.__init_hidden(self.device, batch_size)

        Note hidden gets updated after every forward pass as well.

        """
        hid0 = torch.zeros(self.nlayers, batch_size, self.nhidden,
                           dtype=INPUT_DTYPE)
        c_hid0 = torch.zeros(self.nlayers, batch_size, self.nhidden,
                             dtype=INPUT_DTYPE)
        if device is not None:
            hid0 = hid0.to(device)
            c_hid0 = c_hid0.to(device)
        return (hid0, c_hid0)

    def set_dropout(self, dropout_val):
        """Helper to set the dropout on the LSTM."""
        self.lstm.dropout = dropout_val

    def coarse_forward(self, minibatch):
        """Run the main LSTM, without the extremas part."""
        lstm_out, self.hidden = self.lstm(
            minibatch, self.hidden)
        all_outputs = self.fc_out(lstm_out)
        return all_outputs

    def forward(self, minibatch):
        """Pass in a minibatch of training examples, then run the forward
        pass.

        """
        if not self.extremas:
            return self.coarse_forward(minibatch)
        coarse_inputs, extrema_inputs = BinTensorMaker.extract_coarse_extremas(minibatch, True)
        coarse_outputs = self.coarse_forward(coarse_inputs)
        extrema_low = self.ex_mlp_low(extrema_inputs)
        extrema_high = self.ex_mlp_high(extrema_inputs)
        all_outputs = torch.cat([coarse_outputs, extrema_low, extrema_high], dim=2)
        return all_outputs

    def save_python(self, outfn):
        """Enable re-loading later into Python by saving.

        """
        torch.save(self.state_dict(), outfn)

    @staticmethod
    def delete_model(path):
        """Given the path to the model, delete it.

        """
        os.remove(path)

    @staticmethod
    def delete_pdf(path):
        """Given the path to the model PDF (visualizations), delete it.

        """
        os.remove(path)

    def save_viz(self, outfn):
        """Plot all the model parameters and save the file as a PDF at outfn.

        """
        plot_nn_model(self, outfn)

    @classmethod
    def create_from_path(cls, filename, *, device=None):
        """Factory method that returns an instance of this class, given the
        model at the current filename.

        """
        if device is not None:
            torch_device = torch.device(device)
            state_dict = torch.load(filename,
                                    map_location=torch_device)
        else:
            state_dict = torch.load(filename)
        if 'lstm.weight_ih_l0' not in state_dict:
            raise RuntimeError(
                f"Cannot read model from {filename} -"
                f" doesn't look like an {cls.__name__}")
        nhidden = state_dict['fc_out.weight'].shape[1]
        noutput = state_dict['fc_out.weight'].shape[0]
        if 'ex_mlp_low.0.weight' in state_dict:
            extremas = True
            noutput += 2
        else:
            extremas = False
        ntotal_vars = len(state_dict.keys())
        if extremas:
            ntotal_vars -= 8
        nlstm_vars = ntotal_vars - 2
        nlayers = nlstm_vars // 4
        ninput = state_dict['lstm.weight_ih_l0'].shape[1]
        new_model = cls(ninput, nhidden, noutput, nlayers, extremas=extremas)
        new_model.load_state_dict(state_dict)
        return new_model
