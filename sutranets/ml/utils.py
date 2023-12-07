"""Utilities that can be shared by different ML code for sutranetsing.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/utils.py
from glob import glob
import logging
import torch
from sutranets.ml.evaluators.lstm_evaluator import \
    LSTMEvaluator, JointLSTMEvaluator, TripleLSTMEvaluator
from sutranets.ml.evaluators.multi_lstm_evaluator import MultiLSTMEvaluator
from sutranets.ml.nets.fc_lstm import FCLSTM
from sutranets.ml.nets.joint_lstm import JointLSTM
from sutranets.ml.nets.triple_lstm import TripleLSTM
from sutranets.ml.nets.multivariate_lstm import MultivariateLSTM
logger = logging.getLogger("sutranets.ml.utils")


def __determine_lstm_class(nfine_bins, nfine2_bins):
    """Helper to take care of determining the appropriate PyTorch model
    class given the args.

    """
    if nfine_bins is None:
        return FCLSTM
    if nfine2_bins is None:
        return JointLSTM
    return TripleLSTM


def __load_multivariate_net(lstm_path, subnet_cls, nsub_series, mydevice):
    """Helper to make a multivariate net, with one submodel for each sub-series"""
    net = MultivariateLSTM.create_from_path(
        lstm_path, subnet_cls, nsub_series, device=mydevice)
    return net


def load_lstm_model(nfine_bins, nfine2_bins, lstm_path, nsub_series,
                    mydevice):
    """Helper to take care of creating the appropriate PyTorch model given
    the args.

    """
    cls = __determine_lstm_class(nfine_bins, nfine2_bins)
    if nsub_series is not None:
        model = __load_multivariate_net(lstm_path, cls, nsub_series, mydevice)
    else:
        model = cls.create_from_path(lstm_path, device=mydevice)
    return model


def make_lstm_evaluator(net, device, nfine_bins, nfine2_bins,
                        nsub_series, do_init_hidden):
    """Take a net object and return an evaluator that runs batch_forward
    on this object.

    """
    if nfine_bins is None:
        eval_cls = LSTMEvaluator
    elif nfine2_bins is None:
        eval_cls = JointLSTMEvaluator
    else:
        eval_cls = TripleLSTMEvaluator
    if nsub_series is None:
        lstm_eval = eval_cls(net, device, do_init_hidden)
    else:
        lstm_eval = MultiLSTMEvaluator(net, device, do_init_hidden, eval_cls)
    return lstm_eval


def load_lstm_evaluator(device_str, lstm_model, do_init_hidden, *,
                        nfine_bins=None, nfine2_bins=None,
                        nsub_series=None):
    """Set up net (read from disk, configure the device, etc.) from some
    generic argparser args, as well initialize the LSTMEvaluator that
    will be used to get 1-step-ahead predictions from the net.

    """
    device = torch.device(device_str)
    try:
        lstm_path = glob(lstm_model)[0]
    except IndexError as excep:
        logger.error("Can't find lstm path: %s", lstm_model)
        raise excep
    net = load_lstm_model(nfine_bins, nfine2_bins, lstm_path,
                          nsub_series, device_str)
    net = net.to(device)
    logger.info("Testing net: %s", str(net))
    net.eval()
    lstm_eval = make_lstm_evaluator(
        net, device, nfine_bins, nfine2_bins, nsub_series, do_init_hidden)
    return lstm_eval
