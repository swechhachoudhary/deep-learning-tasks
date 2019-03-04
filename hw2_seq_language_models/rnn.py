import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


############################################################################################
# Problem 1
############################################################################################
class BaseRNN(nn.Module):

  def __init__(self, input_size, hidden_size, output_size, dropout_keep_rate):
    """
    input_size:   The number of units in the input layer
    hidden_size:  The number of units in hidden layer
    output_size:  The number of units in output layer
    
    """
    
    super(BaseRNN, self).__init__()

    self.Wx = nn.Linear(input_size,  hidden_size, bias=False) 
    self.Wh = nn.Linear(hidden_size, hidden_size)             
    
    self.Wy = nn.Linear(hidden_size, output_size)                    
    self.tanh = nn.Tanh()                             

  def forward(self, inputs, hidden):
    """
    Arguments:
      - inp: input to the cell            (batch_size, emb_size)
      - hidden: hidden state to the cell  (batch_size, hidden_size)
      
    Returns:
      - y: output of the cell             (batch_size, output_size)
      - h: hidden state of the cell       (batch_size, hidden_size)  
    """

    h = self.tanh(self.Wh(hidden) + self.Wx(inputs)) 
    y = self._dense(hidden)                
    
    return y, h


class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers. 
    # Your implementation should support any number of stacked hidden layers 
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules 
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
    # provided clones function (as opposed to a regular python list), in order 
    # for Pytorch to recognize these parameters as belonging to this nn.Module 
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.
    
    super().__init__()
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob
    
    self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
    
    self.model = nn.ModuleList()
    
    self.model.append(BaseRNN(input_size  = self.emb_size if layer==0 else self.hidden_size,
		        hidden_size = self.hidden_size,
		        output_size = self.hidden_size if layer < self.num_layers else self.vocab_size,
		        dropout_keep_rate = self.dp_keep_prob if i < self.num_layers else 1.0
		              )for layer in range(self.num_layers))

    self.init_weights_uniform()

  def init_weights_uniform(self):
    # TODO ========================
    # Initialize all the weights uniformly in the range [-0.1, 0.1]
    if module.weight in self.modules():
      torch.nn.init.uniform_(module.weight, -0.1, 0.1)
      #nn.init.xavier_normal_(module.weight)  
    
    # and all the biases to 0 (in place)
    if module.bias in self.modules():
      torch.nn.init.zeros_(module.bias)
    
  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
    return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])

  
  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using a nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    
  
    logits = torch.Tensor(self.seq_len, self.batch_size, self.vocab_size)
    hidden_t = torch.Tensor(self.num_layers, self.batch_size, self.hidden_size)

    for timestep in range(self.seq_len):
      for layer in range(self.num_layers):
        inputs = self.embedding(inputs)[timestep] if layer==0 else outputs
        h_t = hidden[layer] if timestep==0 else hidden_t[layer]

        outputs         = self.model[layer](inputs)        
	hidden_t[layer] = self.model[layer](h_t)

      logits[timestep] = outputs  

    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden_t

 



