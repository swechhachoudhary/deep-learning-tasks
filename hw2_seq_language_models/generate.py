import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy
import matplotlib.pyplot as plt

from models import RNN, GRU
from models import make_model as TRANSFORMER


##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=10,
                    help='size of one minibatch')
parser.add_argument('--hidden_size', type=int, default=1500,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')
# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files


def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


# LOAD DATA
print('Loading data from ' + args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
#
# GENERATE SAMPLE SEQUENCES
#
###############################################################################


gen_seq_len = [35, 70]
for generated_seq_len in gen_seq_len:

    for _model in ['RNN', 'GRU']:

        # MODEL SETUP
        if _model == 'RNN':
            model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                        seq_len=args.seq_len, batch_size=args.batch_size,
                        vocab_size=vocab_size, num_layers=args.num_layers,
                        dp_keep_prob=args.dp_keep_prob)
        elif _model == 'GRU':
            model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                        seq_len=args.seq_len, batch_size=args.batch_size,
                        vocab_size=vocab_size, num_layers=args.num_layers,
                        dp_keep_prob=args.dp_keep_prob)
        else:
            print("Model type not recognized.")

        # Load the saved best model
        model.load_state_dict(torch.load('best_models/' + _model + '/best_params.pt',
                                         map_location=lambda storage, loc: storage))
        model = model.to(device)

        model.eval()

        # First word for each sequence is randomly selected from the vocabulary
        first_word_index = np.random.randint(0, high=model.vocab_size, size=model.batch_size)
        input = torch.from_numpy(first_word_index.astype(np.int64)).to(device)
        # initialize initial hidden state
        init_hidden = model.init_hidden()
        # generate
        generated_seq = model.generate(input, init_hidden, generated_seq_len)

        sentences = []
        with open('generated_sample/' + _model + 'generated_seq_len_' + str(generated_seq_len) + '.txt', 'w') as file:
            for i in range(model.batch_size):
                sentence = [id_2_word[id_] for id_ in generated_seq[:, i]]
                sentences.append(sentence)
                file.write(str(i) + ": " + str(id_2_word[first_word_index[i]]) + ':: ' + ' '.join(sentence) + '\n')
