import argparse
import torch
import torch.nn as nn
import numpy as np
from models import RNN, GRU
from utils import load_model
from pathlib import Path
from helper import ptb_raw_data, ptb_iterator
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rnn_model_folder",
        "-mrnn",
        type=Path,
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--gru_model_folder",
        "-mgru",
        type=Path,
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--save_path", "-p", type=Path, required=True, help="Path to save the plot"
    )
    args = parser.parse_args()
    return args


def get_one_validation_batch(batch_size, num_steps):
    _, valid_data, _, _, _ = ptb_raw_data("./data")
    x, y = next(ptb_iterator(valid_data, batch_size, num_steps))
    return x, y


def convert_to_torch(x, device):
    torch_x = torch.from_numpy(x.astype(np.int64))

    # put seq dim first
    x_t = torch_x.transpose(0, 1).contiguous()
    # move to appropriate devide
    return x_t.to(device)


def get_norm_t(all_hidden):
    norm_t = []
    for t, h_t in enumerate(all_hidden):
        g_h_t_concat = torch.cat([h_t_l.grad for h_t_l in h_t], dim=1)
        norm = torch.norm(g_h_t_concat, p=2, dim=1)
        norm_t.append(norm.mean().cpu().numpy())
    normalized_norm = norm_t / max(norm_t)
    return normalized_norm


def get_grad_norm(model, x, y, device):
    loss_fn = torch.nn.CrossEntropyLoss()

    hidden = model.init_hidden()
    hidden = hidden.to(device)

    model.zero_grad()

    outputs, all_hidden = model(x, hidden, keep_hidden_grad=True)
    output_last = outputs[-1]
    target_last = y[-1]
    loss_last = loss_fn(output_last, target_last)
    loss_last.backward()

    norm_t = get_norm_t(all_hidden)
    norm_t = np.asarray(norm_t)
    return norm_t


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rnn_model = load_model(args.rnn_model_folder, device)
    gru_model = load_model(args.gru_model_folder, device)

    assert(rnn_model.batch_size == gru_model.batch_size)
    assert(rnn_model.seq_len == gru_model.seq_len)

    x, y = get_one_validation_batch(
        rnn_model.batch_size, rnn_model.seq_len)
    x = convert_to_torch(x, device)
    y = convert_to_torch(y, device)

    norm_t_rnn = get_grad_norm(rnn_model, x, y, device)
    norm_t_gru = get_grad_norm(gru_model, x, y, device)

    x = np.arange(1, len(norm_t_rnn) + 1)
    plt.plot(x, norm_t_rnn, label="RNN")
    plt.plot(x, norm_t_gru, label="GRU")
    plt.title("Gradient norm vs t")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Normalized gradient")
    plt.savefig(args.save_path)
    plt.close()


if __name__ == "__main__":
    main()
