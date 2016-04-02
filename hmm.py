from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import *
import tensorflow as tf

class HMMCell(RNNCell):
  """Hidden Markov Model hidden transition cell."""

  def __init__(self, num_units, trans_weights, input_size=None):
    self._num_units = num_units
    self._weights = trans_weights
    self._input_size = num_units if input_size is None else input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    raise NotImplementedError("Abstract method")


class HMMCellFW(HMMCell):
  """Hidden Markov Model hidden transition cell."""
  def __call__(self, inputs, state, scope=None):
    """HMM: output = new_state = tanh(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):  
    #######################
     # "HMMForward computation. Complete the line below using variables: inputs, state, self._weights:"
     output = 

    #######################
    return output, output

class HMMCellBW(HMMCell):
  """Hidden Markov Model hidden transition cell."""
  def __call__(self, inputs, state, scope=None):
    """HMM: output = new_state = tanh(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__): 
    #######################
     # "HMMBackward computation. Complete the line below using variables: inputs, state, self._weights:"
     output = 

    #######################
    return output, output


