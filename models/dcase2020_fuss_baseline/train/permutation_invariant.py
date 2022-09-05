# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Permutation invariance, as applicable to loss functions."""

import functools
import itertools
import typing
import numpy as np
import tensorflow.compat.v1 as tf

def calcloss_wave_perm(loss_fn: typing.Callable[..., tf.Tensor],x,y):
  #x,y is [batch,class,samplenum]
  #make loss matrix
  batch = x.shape[0]
  source = x.shape[1]


  assert x.shape == y.shape

  x_formatrix = tf.expand_dims(x,axis = 1)
  y_formatrix = tf.expand_dims(y,axis = 2)
  x_formatrix = tf.tiles(x_formatrix,[1,source,1,1])
  y_formatrix = tf.tiles(y_formatrix,[1,1,source,1])

  loss_matrix = loss_fn(x_formatrix,y_formatrix)

  loss_matrix = tf.reshape(loss_matrix,[batch,source,source])
  return 


def My_resolve_permutation(loss_matrix):
  """Resolves permutation from an all-pairs loss_matrix input.

  Args:
    loss_matrix: tensor of shape [batch, source, source]
      axis 1 refers to the estimate.
      axis 2 refers to the reference.
  Returns:
    permutation: tensor of shape [batch, source, 2] such that
      tf.gather_nd(estimates, permutation, 1) returns the permuted estimates
      that achieves the lowest loss.
  """
  batch = loss_matrix.shape[0]
  source = loss_matrix.shape[1]

  # Compute permutations as vectors of indices into flattened loss matrix.
  # permutations will have shape [batch, source!, source, 1].
  permutations = tf.constant(list(itertools.permutations(range(source))))
  permutations = tf.expand_dims(permutations, 0)
  permutations = tf.tile(permutations, [batch, 1, 1])

  # Expand loss dimensions for gather.
  # loss_matrix.shape will be (batch, source!, source, source)
  loss_matrix = tf.expand_dims(loss_matrix, 1)
  loss_matrix = tf.tile(loss_matrix, [1, permutations.shape[1], 1, 1])

  # Compute the total loss for each permutation.
  # permuted_loss.shape will be (batch, source!)
  permuted_loss = tf.gather_nd(loss_matrix, permutations, batch_dims=3)
  permuted_loss = tf.reduce_sum(permuted_loss, axis=2)

  # Get and return the permutation with the lowest total loss.
  # loss_argmin.shape will be (batch, 1)
  loss_argmin = tf.argmin(permuted_loss, axis=1)
  loss_argmin = tf.expand_dims(loss_argmin, 1)

  # permutation.shape will be (batch, source, 1)
  permutation = tf.gather_nd(permutations, loss_argmin, batch_dims=1)

  return permutation


def _resolve_permutation(loss_matrix):
  """Resolves permutation from an all-pairs loss_matrix input.

  Args:
    loss_matrix: tensor of shape [batch, source, source]
      axis 1 refers to the estimate.
      axis 2 refers to the reference.
  Returns:
    permutation: tensor of shape [batch, source, 2] such that
      tf.gather_nd(estimates, permutation, 1) returns the permuted estimates
      that achieves the lowest loss.
  """
  batch = loss_matrix.shape[0]
  source = loss_matrix.shape[1]

  # Compute permutations as vectors of indices into flattened loss matrix.
  # permutations will have shape [batch, source!, source, 1].
  permutations = tf.constant(list(itertools.permutations(range(source))))
  permutations = tf.expand_dims(tf.expand_dims(permutations, 0), 3)
  permutations = tf.tile(permutations, [batch, 1, 1, 1])

  # Expand loss dimensions for gather.
  # loss_matrix.shape will be (batch, source!, source, source)
  loss_matrix = tf.expand_dims(loss_matrix, 1)
  loss_matrix = tf.tile(loss_matrix, [1, permutations.shape[1], 1, 1])

  # Compute the total loss for each permutation.
  # permuted_loss.shape will be (batch, source!)
  permuted_loss = tf.gather_nd(loss_matrix, permutations, batch_dims=3)
  permuted_loss = tf.reduce_sum(permuted_loss, axis=2)

  # Get and return the permutation with the lowest total loss.
  # loss_argmin.shape will be (batch, 1)
  loss_argmin = tf.argmin(permuted_loss, axis=1)
  loss_argmin = tf.expand_dims(loss_argmin, 1)

  # permutation.shape will be (batch, source, 1)
  permutation = tf.gather_nd(permutations, loss_argmin, batch_dims=1)

  return permutation


def _apply(loss_fn: typing.Callable[..., tf.Tensor],
           reference: tf.Tensor,
           estimate: tf.Tensor,
           allow_repeated: bool,
           enable: bool) -> typing.Any:
  """Return permutation invariant loss.

  Note that loss_fn must in general handle an arbitrary number of sources, since
  this function may expand in that dimention to get losses on all
  reference-estimate pairs.

  Args:
    loss_fn: function with the following signature:
      Args
        reference [batch, source', ...] tensor
        estimate [batch, source', ...] tensor
      Returns
        A [batch, source'] tensor of dtype=tf.float32
    reference: [batch, source, ...] tensor.
    estimate: [batch, source, ...] tensor.
    allow_repeated: If true, allow the same estimate to be used to match
      multiple references.
    enable: If False, apply the loss function in fixed order and return its
      value and the unpermuted estimates.

  Returns:
    loss, A [batch, source] tensor of dtype=tf.float32
    permuted_estimate, A tensor like estimate.
  """
  reference = tf.convert_to_tensor(reference)
  estimate = tf.convert_to_tensor(estimate)

  if not enable:
    return loss_fn(reference, estimate), estimate

  assert reference.shape[:2] == estimate.shape[:2]
  batch = reference.shape[0]
  source = reference.shape[1]

  # Replicate estimate on axis 1
  # estimate.shape will be (batch, source * source, ...)
  multiples = np.ones_like(estimate.shape)
  multiples[1] = source
  estimate_tiled = tf.tile(estimate, multiples)

  # Replicate reference on new axis 2, then combine axes [1, 2].
  # reference.shape will be (batch, source * source, ...)
  reference_tiled = tf.expand_dims(reference, 2)
  multiples = np.ones_like(reference_tiled.shape)
  multiples[2] = source
  reference_tiled = tf.tile(reference_tiled, multiples)
  reference_tiled = tf.reshape(reference_tiled, estimate_tiled.shape)

  # Compute the loss matrix.
  # loss_matrix.shape will be (batch, source, source).
  # Axis 1 is the estimate.  Axis 2 is the reference.
  #print("source :{} , reftiledhsape : {}".format(source,reference_tiled.shape))
  #print("source :{} , esttiledhsape : {}".format(source,estimate_tiled.shape))
  #print("lossfn :{}".format(loss_fn))
  loss_matrix = tf.reshape(loss_fn(reference_tiled, estimate_tiled),
                           [batch, source, source])
  #print(loss_matrix.eval())
  # Get the best permutation.
  # permutation.shape will be (batch, source, 1)
  if allow_repeated:
    permutation = tf.argmin(loss_matrix, axis=2, output_type=tf.int32)
    permutation = tf.expand_dims(permutation, 2)
  else:
    permutation = _resolve_permutation(loss_matrix)
  assert permutation.shape == (batch, source, 1), permutation.shape
  print("permutaion_normal")
  print(permutation)
  # Permute the estimates according to the best permutation.
  estimate_permuted = tf.gather_nd(estimate, permutation, batch_dims=1)
  loss_permuted = tf.gather_nd(loss_matrix, permutation, batch_dims=2)

  return loss_permuted, estimate_permuted

from . import final_model

def _apply_label(loss_fn: typing.Callable[..., tf.Tensor],loss_fn_label: typing.Callable[..., tf.Tensor],
           reference: tf.Tensor,
           estimate: tf.Tensor,
           reference_label: tf.Tensor,
           estimate_label: tf.Tensor,
           allow_repeated: bool,
           enable: bool,
           useonlystandardloss: bool=False) -> typing.Any:
  """Return permutation invariant loss.

  Note that loss_fn must in general handle an arbitrary number of sources, since
  this function may expand in that dimention to get losses on all
  reference-estimate pairs.

  Args:
    loss_fn: function with the following signature:
      Args
        reference [batch, source', ...] tensor
        estimate [batch, source', ...] tensor
      Returns
        A [batch, source'] tensor of dtype=tf.float32
    reference: [batch, source, ...] tensor.
    estimate: [batch, source, ...] tensor.
    allow_repeated: If true, allow the same estimate to be used to match
      multiple references.
    enable: If False, apply the loss function in fixed order and return its
      value and the unpermuted estimates.

  Returns:
    loss, A [batch, source] tensor of dtype=tf.float32
    permuted_estimate, A tensor like estimate.
  """
  reference = tf.convert_to_tensor(reference)
  estimate = tf.convert_to_tensor(estimate)
  reference_label = tf.convert_to_tensor(reference_label)
  estimate_label = tf.convert_to_tensor(estimate_label)

  if not enable:
    return loss_fn(reference, estimate), estimate

  assert reference.shape[:2] == estimate.shape[:2]
  assert reference_label.shape[:2] == estimate_label.shape[:2]
  batch = reference.shape[0]
  source = reference.shape[1]


  
  def get_tiled(ref,est):

    # Replicate estimate on axis 1
    # estimate.shape will be (batch, source * source, ...)
    multiples = np.ones_like(est.shape)
    multiples[1] = source
    estimate_tiled = tf.tile(est, multiples)

    # Replicate reference on new axis 2, then combine axes [1, 2].
    # reference.shape will be (batch, source * source, ...)
    reference_tiled = tf.expand_dims(ref, 2)
    multiples = np.ones_like(reference_tiled.shape)
    multiples[2] = source
    reference_tiled = tf.tile(reference_tiled, multiples)
    reference_tiled = tf.reshape(reference_tiled, estimate_tiled.shape)
    return reference_tiled , estimate_tiled
  reference_tiled,estimate_tiled = get_tiled(reference,estimate)
  reference_label_tiled,estimate_label_tiled = get_tiled(reference_label,estimate_label)
  # Compute the loss matrix.
  # loss_matrix.shape will be (batch, source, source).
  # Axis 1 is the estimate.  Axis 2 is the reference.


  
  #if not useonlystandardloss:
  print("tiled shape")
  #loss_fn = final_model.log_mse_loss
  #loss_fn_label = final_model.loss_label
  #print(reference_tiled.shape)
  #print(estimate_tiled.shape)
  #print(reference_label_tiled.shape)
  #rint(estimate_label_tiled.shape)
  #print(loss_fn(reference_tiled, estimate_tiled).shape)
  #print(loss_fn_label(reference_label_tiled, estimate_label_tiled).shape)



  if True:
    loss = final_model.different_loss_connect( loss_fn(reference_tiled, estimate_tiled) , loss_fn_label(reference_label_tiled, estimate_label_tiled))
    #print(loss_fn_label(reference_label_tiled, estimate_label_tiled))
    #print(loss_fn(reference_tiled, estimate_tiled))
    #print("↑ label loss matrics")
  else:
    loss = loss_fn(reference_tiled, estimate_tiled)
  loss_matrix = tf.reshape(loss,[batch, source, source])
  # Get the best permutation.
  # permutation.shape will be (batch, source, 1)
  if allow_repeated:
    permutation = tf.argmin(loss_matrix, axis=2, output_type=tf.int32)
    permutation = tf.expand_dims(permutation, 2)
  else:
    permutation = _resolve_permutation(loss_matrix)
  assert permutation.shape == (batch, source, 1), permutation.shape
  print("permutaion")
  print(permutation)
  # Permute the estimates according to the best permutation.
  estimate_permuted = tf.gather_nd(estimate, permutation, batch_dims=1)
  loss_permuted = tf.gather_nd(loss_matrix, permutation, batch_dims=2)
  estimate_label_permuted = tf.gather_nd(estimate_label, permutation, batch_dims=1)

  return loss_permuted, estimate_permuted, estimate_label_permuted

def wrap(loss_fn: typing.Callable[..., tf.Tensor],
         allow_repeated: bool = False,
         enable: bool = True) -> typing.Callable[..., typing.Any]:
  """Returns a permutation invariant version of loss_fn.

  Args:
    loss_fn: function with the following signature:
      Args
        reference [batch, source', ...] tensor
        estimate [batch, source', ...] tensor
        **args Any remaining arguments to loss_fn
      Returns
        A [batch, source'] tensor of dtype=tf.float32
    allow_repeated: If true, allow the same estimate to be used to match
      multiple references.
    enable: If False, return a fuction that applies the loss function in fixed
      order, returning its value and the (unpermuted) estimate.

  Returns:
    A function with same arguments as loss_fn returning loss, permuted_estimate

  """
  def wrapped_loss_fn(reference, estimate, **args):
    return _apply(functools.partial(loss_fn, **args),
                  reference,
                  estimate,
                  allow_repeated,
                  enable)
  return wrapped_loss_fn

def wrap_label(loss_fn: typing.Callable[..., tf.Tensor],loss_fn_label: typing.Callable[..., tf.Tensor],
         allow_repeated: bool = False,
         enable: bool = True) -> typing.Callable[..., typing.Any]:
  """Returns a permutation invariant version of loss_fn.

  Args:
    loss_fn: function with the following signature:
      Args
        reference [batch, source', ...] tensor
        estimate [batch, source', ...] tensor
        **args Any remaining arguments to loss_fn
      Returns
        A [batch, source'] tensor of dtype=tf.float32
    allow_repeated: If true, allow the same estimate to be used to match
      multiple references.
    enable: If False, return a fuction that applies the loss function in fixed
      order, returning its value and the (unpermuted) estimate.

  Returns:
    A function with same arguments as loss_fn returning loss, permuted_estimate

  """
  def wrapped_loss_fn(reference, estimate,
                  reference_label,
                  estimate_label,
                  **args):
    return _apply_label(functools.partial(loss_fn, **args),functools.partial(loss_fn_label, **args),
                  reference,
                  estimate,
                  reference_label,
                  estimate_label,
                  allow_repeated,
                  enable,
                  useonlystandardloss=False)
  return wrapped_loss_fn

def wrap_label_useonlystandard(loss_fn: typing.Callable[..., tf.Tensor],loss_fn_label: typing.Callable[..., tf.Tensor],
         allow_repeated: bool = False,
         enable: bool = True) -> typing.Callable[..., typing.Any]:
  """Returns a permutation invariant version of loss_fn.

  Args:
    loss_fn: function with the following signature:
      Args
        reference [batch, source', ...] tensor
        estimate [batch, source', ...] tensor
        **args Any remaining arguments to loss_fn
      Returns
        A [batch, source'] tensor of dtype=tf.float32
    allow_repeated: If true, allow the same estimate to be used to match
      multiple references.
    enable: If False, return a fuction that applies the loss function in fixed
      order, returning its value and the (unpermuted) estimate.

  Returns:
    A function with same arguments as loss_fn returning loss, permuted_estimate

  """
  def wrapped_loss_fn(reference, estimate,
                  reference_label,
                  estimate_label,
                  **args):
    return _apply_label(functools.partial(loss_fn, **args),functools.partial(loss_fn_label, **args),
                  reference,
                  estimate,
                  reference_label,
                  estimate_label,
                  allow_repeated,
                  enable,
                  useonlystandardloss=True)
  return wrapped_loss_fn