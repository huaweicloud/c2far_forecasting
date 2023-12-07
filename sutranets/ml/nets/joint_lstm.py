"""Long-short-term memory neural network that has two tracks: one that
pays attention to one subset of "coarse" features and another that
pays attention to a subset of "fine" features, each going into a
separate LSTM.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/nets/joint_lstm.py
import os
import torch
from torch import nn
from sutranets.ml.constants import INPUT_DTYPE
from sutranets.ml.bin_tensor_maker import BinTensorMaker
from sutranets.ml.nets.utils import make_extrema_mlps, plot_nn_model


class JointLSTM(nn.Module):
    """The joint LSTM, with the dual coarse/fine LSTM tracks.

    """

    def __init__(self, ninput, ncoarse, nfine, nhidden, noutput,
                 nlayers, *, extremas=False):
        super().__init__()
        assert not isinstance(nhidden, list), "Hidden layer has fixed size"
        self.ninput = ninput
        self.ncoarse = ncoarse
        self.nfine = nfine
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.extremas = extremas
        self.noutput = noutput
        if self.extremas:
            noutput -= 2
        if not noutput == (ncoarse + nfine):
            msg = f"noutput={noutput} != ncoarse={ncoarse} + nfine={nfine}"
            raise RuntimeError(msg)
        ncoarse_subset, nfine_subset, _, nextrema_subset = BinTensorMaker.compute_nsubsets(
            ninput, ncoarse, nfine, None, extremas=extremas)
        self.coarse_lstm = nn.LSTM(ncoarse_subset, self.nhidden, self.nlayers)
        self.coarse_out = nn.Linear(self.nhidden, ncoarse)
        self.fine_lstm = nn.LSTM(nfine_subset, self.nhidden, self.nlayers)
        self.fine_out = nn.Linear(self.nhidden, nfine)
        self.ex_mlp_low, self.ex_mlp_high = make_extrema_mlps(nextrema_subset, self.nhidden, extremas)
        self.coarse_hidden, self.fine_hidden = self.run_init_hidden(None, 1)

    def run_init_hidden(self, device, batch_size):
        """Initialize the two LSTM's hidden states. I.e., before doing each
        new example, we typically re-init the hidden state to zero.

        """
        self.coarse_hidden = self.__init_hidden(device, batch_size)
        self.fine_hidden = self.__init_hidden(device, batch_size)
        return self.coarse_hidden, self.fine_hidden

    def __init_hidden(self, device, batch_size):
        hid0 = torch.zeros(self.nlayers, batch_size, self.nhidden,
                           dtype=INPUT_DTYPE)
        c_hid0 = torch.zeros(self.nlayers, batch_size, self.nhidden,
                             dtype=INPUT_DTYPE)
        if device is not None:
            hid0 = hid0.to(device)
            c_hid0 = c_hid0.to(device)
        return (hid0, c_hid0)

    def set_dropout(self, dropout_val):
        """Helper to set the dropout on both the coarse and the fine LSTMs."""
        self.coarse_lstm.dropout = dropout_val
        self.fine_lstm.dropout = dropout_val

    def coarse_forward(self, coarse_inputs):
        """Do forward on just the coarse part of the input."""
        coarse_lstm_out, self.coarse_hidden = self.coarse_lstm(
            coarse_inputs, self.coarse_hidden)
        coarse_logits = self.coarse_out(coarse_lstm_out)
        return coarse_logits

    def fine_forward(self, fine_inputs):
        """Do forward on just the fine part of the input."""
        fine_lstm_out, self.fine_hidden = self.fine_lstm(
            fine_inputs, self.fine_hidden)
        fine_logits = self.fine_out(fine_lstm_out)
        return fine_logits

    def forward(self, minibatch):
        """Pass in a minibatch of training examples, then run the forward
        pass.

        """
        coarse_inputs, fine_inputs, extrema_inputs = BinTensorMaker.extract_coarse_fine(
            minibatch, self.ncoarse, extremas=self.extremas)
        coarse_logits = self.coarse_forward(coarse_inputs)
        fine_logits = self.fine_forward(fine_inputs)
        all_outputs_lst = [coarse_logits, fine_logits]
        if extrema_inputs is not None:
            extrema_low = self.ex_mlp_low(extrema_inputs)
            extrema_high = self.ex_mlp_high(extrema_inputs)
            all_outputs_lst += [extrema_low, extrema_high]
        all_logits = torch.cat(all_outputs_lst, dim=2)
        return all_logits

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
        if 'coarse_lstm.weight_ih_l0' not in state_dict:
            raise RuntimeError(
                f"Cannot read model from {filename} -"
                f" doesn't look like an {cls.__name__}")
        nhidden = state_dict['coarse_out.weight'].shape[1]
        ncoarse = state_dict['coarse_out.weight'].shape[0]
        nfine = state_dict['fine_out.weight'].shape[0]
        noutput = ncoarse + nfine
        if 'ex_mlp_low.0.weight' in state_dict:
            extremas = True
            noutput += 2
        else:
            extremas = False
        ntotal_vars = len(state_dict.keys())
        if extremas:
            ntotal_vars -= 8
        ntotal_coarse_vars = ntotal_vars // 2
        ntotal_coarse_lstm_vars = ntotal_coarse_vars - 2
        nlayers = (ntotal_coarse_lstm_vars) // 4
        ncoarse_subset = state_dict['coarse_lstm.weight_ih_l0'].shape[1]
        ninput = BinTensorMaker.compute_joint_ninput(ncoarse_subset, ncoarse)
        new_model = cls(ninput, ncoarse, nfine, nhidden, noutput,
                        nlayers, extremas=extremas)
        new_model.load_state_dict(state_dict)
        return new_model
