import sys
import torch
from models import RNN, GRU
from helper import get_vocab_size


def load_config_as_dict(config_file):
    with open(config_file, 'r') as cf:
        config_text = cf.read()
    config_list = config_text.split('\n')
    args_dict = {}
    for arg in config_list:
        if '    ' in arg:
            key, val = arg.split('    ')
            args_dict[key] = val
    return args_dict


def load_model(path, device):
    config = load_config_as_dict(path / 'exp_config.txt')
    if config['model'] == 'RNN':
        model_type = RNN
    elif config['model'] == 'GRU':
        model_type = GRU
    else:
        raise NotImplementedError("Not implemented for model {}".format(config['model']))

    model = model_type(
        emb_size=int(config['emb_size']),
        hidden_size=int(config['hidden_size']),
        seq_len=int(config['seq_len']),
        batch_size=int(config['batch_size']),
        vocab_size=get_vocab_size(),
        num_layers=int(config['num_layers']),
        dp_keep_prob=float(config['dp_keep_prob'])
    )
    model.load_state_dict(
        torch.load(path / 'best_params.pt', map_location=lambda storage, loc: storage)
    )

    model.to(device)
    return model
