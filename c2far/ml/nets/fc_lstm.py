"""Code for a generic long-short-term memory Neural Network that
outputs a vector of logits.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/nets/fc_lstm.py

import torch
from torch import nn
from c2far.ml.bin_tensor_maker import BinTensorMaker
from c2far.ml.constants import INPUT_DTYPE
from c2far.ml.nets.utils import make_extrema_mlps


class FCLSTM(nn.Module):
    """RNN that passes hidden state along using a LSTM, but then generates
    output values using a linear affine mapping from this hidden
    state, which we can interpret as logits, or as parameters to a
    Gaussian, etc.  Optionally with MLPs for extremas.

    """

    def __init__(self, ninput, nhidden, noutput, nlayers, *, extremas=False):
        """Depending on the sizes of the input, output, and hidden layers, add
        attributes for the inner LSTM and the fully-connected layers
        (including both weights and a bias term).

        Note, the PARAMETERS of the LSTM are initialized using the
        default init method in Pytorch.  The hidden state is set by us
        (currently setting it to all zeros).

        Arguments:

        ninput: Int, how many values in the input layer. E.g. a
        1-hot-encoding of the coarse cutoffs.

        nhidden: Int, how big the hidden layer should be, in order.
        When nlayers > 1, each layer has the same number of neurons.

        noutput: Int, how many output - e.g. a 1-hot-encoding of the
        output value.

        nlayers: Int, how many layers in the LSTM.

        Optional arguments:

        extremas: Boolean, if True, add additional architecture for
        generating the an extreme-bin parameter.

        """
        super().__init__()
        assert not isinstance(nhidden, list), "Hidden layer has fixed size"
        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nlayers = nlayers
        self.extremas = extremas
        nlstm_input, _, _, nextrema_input = BinTensorMaker.compute_nsubsets(
            ninput, None, None, None, extremas=extremas)
        # If using these, two of the outputs are reserved for extrema values:
        if self.extremas:
            nlstm_output = noutput - 2
        else:
            nlstm_output = noutput
        self.lstm = nn.LSTM(nlstm_input, self.nhidden, self.nlayers)
        self.fc_out = nn.Linear(self.nhidden, nlstm_output)
        # If doing extremas, we have these MLPs too (each outputs 1 value):
        self.ex_mlp_low, self.ex_mlp_high = make_extrema_mlps(nextrema_input, self.nhidden, extremas)
        self.hidden = self.run_init_hidden(None, 1)

    def run_init_hidden(self, device, batch_size):
        """Initialize the LSTM hidden state, e.g., before doing each new
        example, we typically re-init the hidden state.

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

        Arguments:

        minibatch: Tensor, LENGTH x BATCHSIZE x ninput

        Returns:

        Tensor, LENGTH x BATCHSIZE x noutput

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
        """When you want to be able to re-load it later, use this method.

        """
        torch.save(self.state_dict(), outfn)

    @classmethod
    def create_from_path(cls, filename, *, device=None):
        """This is a factory method that will return an instance of this
        class, given the model at the current filename.

        Arguments:

        filename: String, the path to the place where we previously
        stored one of these using the save method above.

        device: String, if given, we dynamically move the model to the
        given device.

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
        # Exclude extremas, if we have them (2 mlps, 2 layers, weights/biases)::
        if extremas:
            ntotal_vars -= 8
        # Exclude output weights/biases variables:
        nlstm_vars = ntotal_vars - 2
        # Divide out the four ih/hh weights/biases:
        nlayers = nlstm_vars // 4
        ninput = state_dict['lstm.weight_ih_l0'].shape[1]
        new_model = cls(ninput, nhidden, noutput, nlayers, extremas=extremas)
        new_model.load_state_dict(state_dict)
        return new_model
