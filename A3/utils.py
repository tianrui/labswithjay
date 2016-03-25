import tensorflow as tf
import numpy as np

def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
  """Computes the sum of elements across dimensions of a tensor in log domain.
     
     It uses a similar API to tf.reduce_sum.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. 
    keep_dims: If true, retains reduced dimensions with length 1.
  Returns:
    The reduced tensor.
  """
  max_input_tensor1 = tf.reduce_max(input_tensor, 
                                    reduction_indices, keep_dims=keep_dims)
  max_input_tensor2 = max_input_tensor1
  if not keep_dims:
    max_input_tensor2 = tf.expand_dims(max_input_tensor2, 
                                       reduction_indices) 
  return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

def logsoftmax(input_tensor):
  """Computes normal softmax nonlinearity in log domain.

     It can be used to normalize log probability.
     The softmax is always computed along the second dimension of the input Tensor.     
 
  Args:
    input_tensor: Unnormalized log probability.
  Returns:
    normalized log probability.
  """
  return input_tensor - reduce_logsumexp(input_tensor, keep_dims=True)

def L2_dist(A, B):
    """Computes L2 distance of points in two matrices
    sum(x-y)^2 = sum x^2 + sum y^2 + sum xy
    Args:
        A: MxD tensor
        B: NxD tensor
    Returns:
        MxN tensor of L2 distance
    """
    t1 = (tf.reduce_sum(A*A, reduction_indices=1, keep_dims=True))
    t2 = tf.transpose((tf.reduce_sum(B*B, reduction_indices=1, keep_dims=True)))
    t3 = 2 * tf.matmul(A, tf.transpose(B))
    return tf.sqrt(t1+t2-t3)

def L2_dist_np(A, B):
    M, D = A.shape
    N, D1 = B.shape
    assert D==D1
    return np.sum(A**2, axis=1)[:, None] + np.transpose(np.sum(B**2, axis=1))[None, :] - 2 * np.matmul(A, np.transpose(B))

def load_data(fname):
    return np.load(fname)

