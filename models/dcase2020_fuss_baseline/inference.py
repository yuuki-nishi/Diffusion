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
"""Inference for trained DCASE 2020 task 4 separation model."""

import numpy as np
import tensorflow.compat.v1 as tf


class SeparationModel(object):
  """Tensorflow audio separation model."""

  def __init__(self, checkpoint_path, metagraph_path, outtensorname):
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    print("check : {}".format(checkpoint_path))
    print("meta : {}".format(metagraph_path))
    print("outtensor name : {}".format(outtensorname))
    if outtensorname == None:
      outtensorname = "denoised_waveforms"

    with self.graph.as_default():
      new_saver = tf.train.import_meta_graph(metagraph_path)
      new_saver.restore(self.sess, checkpoint_path)
    self.input_placeholder = self.graph.get_tensor_by_name(
        'input_audio/receiver_audio:0')
    print("input_placeholder")
    print(self.input_placeholder)
    self.output_tensor = self.graph.get_tensor_by_name('{}:0'.format(outtensorname))

  def separate(self, mixture_waveform):
    """Separates a mixture waveform into sources.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    mixture_waveform_input = np.reshape(mixture_waveform, (1, 1, -1))
    print(mixture_waveform_input.shape)
    print(self.input_placeholder.shape)
    print(self.output_tensor.shape)
    #mixture_waveform_input = tf.constant([1])
    separated_waveforms = self.sess.run(
        self.output_tensor,
        feed_dict={self.input_placeholder: mixture_waveform_input})[0]#???????????????batch????????????[0]?
    print("seped shape : {}".format(separated_waveforms.shape))
    return separated_waveforms
class FinalSeparationModel(object):
  """Tensorflow audio separation model."""

  def __init__(self, checkpoint_path, metagraph_path, outtensorname):
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    print("check : {}".format(checkpoint_path))
    print("meta : {}".format(metagraph_path))
    print("outtensor name : {}".format(outtensorname))
    if outtensorname == None:
      outtensorname = "denoised_waveforms"

    with self.graph.as_default():
      new_saver = tf.train.import_meta_graph(metagraph_path)
      new_saver.restore(self.sess, checkpoint_path)
    self.input_placeholder = self.graph.get_tensor_by_name(
        'input_audio/receiver_audio:0')
    self.discinput_placeholder = self.graph.get_tensor_by_name(
        'discinput:0')
    print("input_placeholder")
    print(self.input_placeholder)
    self.output_tensor = self.graph.get_tensor_by_name('{}:0'.format("final_prediction"))
    self.output_tensor_prob = self.graph.get_tensor_by_name('{}:0'.format("final_probabilities"))

  def separate(self, mixture_waveform,labels):
    """Separates a mixture waveform into sources.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    mixture_waveform_input = np.reshape(mixture_waveform, (1, 1, -1))
    labels = np.reshape(labels, (1, 1, -1))
    #print(mixture_waveform_input.shape)
    #print(self.input_placeholder.shape)
    #print(self.output_tensor.shape)
    #mixture_waveform_input = tf.constant([1])
    tmplabelholder = self.graph.get_tensor_by_name(
        '{}:0'.format("input_audio/label"))
    ret,ret2 = self.sess.run(
        [self.output_tensor,self.output_tensor_prob],
        feed_dict={self.input_placeholder: mixture_waveform_input,tmplabelholder : labels})
    #print(ret.shape)#[1,2-5,160000]
    #print(ret2.shape)#[1,2-5,2-5]
    return ret[0],ret2[0]
  def getprobs(self, waveforms):
    """Separates a mixture waveform into sources.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    waveforms = tf.squeeze(waveforms)
    waveforms=np.array([waveforms])
    ret = self.sess.run(
        [self.output_tensor_prob],
        feed_dict={self.discinput_placeholder: waveforms})
    #print(ret.shape)#[1,2-5,160000]
    #print(ret2.shape)#[1,2-5,2-5]
    return ret[0]
  def getwavs(self,mixture_waveform,labels):
    """Separates a mixture waveform into sources.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    mixture_waveform_input = np.reshape(mixture_waveform, (1, 1, -1))
    labels = np.reshape(labels, (1, 1, -1))
    #print(mixture_waveform_input.shape)
    #print(self.input_placeholder.shape)
    #print(self.output_tensor.shape)
    #mixture_waveform_input = tf.constant([1])
    tmplabelholder = self.graph.get_tensor_by_name(
        '{}:0'.format("input_audio/label"))
    ret = self.sess.run(
        [self.output_tensor],
        feed_dict={self.input_placeholder: mixture_waveform_input,tmplabelholder : labels})
    #print(ret.shape)#[1,2-5,160000]
    #print(ret2.shape)#[1,2-5,2-5]
    return ret[0]

class DiscriminateModel(object):
  """Tensorflow audio separation model."""

  def __init__(self, checkpoint_path, metagraph_path):
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      new_saver = tf.train.import_meta_graph(metagraph_path)
      new_saver.restore(self.sess, checkpoint_path)
    
    """for node in self.graph.as_graph_def().node:
      pass
      print(node.name)"""
    self.input_placeholder = self.graph.get_tensor_by_name(
        'input_audio/source_image:0')
    self.output_tensor = self.graph.get_tensor_by_name('disc_func/probability:0')

  def getprobability(self, mixture_waveform):
    """Separates a mixture waveform into sources.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    #waveform_input = np.reshape(mixture_waveform, (1,1, -1))
    if hasattr(mixture_waveform,"numpy"):
      waveform_input = mixture_waveform.numpy().copy()
    else:
      waveform_input = mixture_waveform

    separated_waveforms = self.sess.run(
        self.output_tensor,
        feed_dict={self.input_placeholder: waveform_input})
    return separated_waveforms


def sqrt_hann_window(length, dtype):
  return tf.sqrt(tf.signal.hann_window(length, dtype=dtype, periodic=True))


class OracleBinaryMasking(object):
  """Oracle binary masking with STFT, implemented in tensorflow."""

  def __init__(self, ws=0.032, hs=0.008, sr=16000.0):
    self.stft_win = int(np.round(ws * sr))
    self.stft_hop = int(np.round(hs * sr))
    self.fft_length = int(2**np.ceil(np.log2(self.stft_win)))

  def _pad_beginning(self, waveform):
    pad_len = int(self.stft_win - self.stft_hop)
    pad_spec = [(0, 0)] * (len(waveform.shape) - 1) + [(pad_len, 0)]
    return tf.pad(waveform, pad_spec)

  def _clip_beginning(self, waveform):
    clip = int(self.stft_win - self.stft_hop)
    return waveform[..., clip:]

  def _stft_forward(self, inp):
    waveform = self._pad_beginning(inp)
    return tf.signal.stft(
        waveform, self.stft_win, self.stft_hop,
        fft_length=self.fft_length,
        window_fn=sqrt_hann_window,
        pad_end=True)

  def _stft_inverse(self, inp):
    waveform = tf.signal.inverse_stft(
        inp, self.stft_win, self.stft_hop,
        fft_length=self.fft_length,
        window_fn=tf.signal.inverse_stft_window_fn(
            self.stft_hop, forward_window_fn=sqrt_hann_window))
    return self._clip_beginning(waveform)

  def _oracle_binary_mask(self, amplitudes_sources):
    amplitudes_max = tf.reduce_max(amplitudes_sources, axis=0, keepdims=True)
    return tf.cast(tf.equal(amplitudes_sources, amplitudes_max),
                   amplitudes_sources.dtype)

  def separate(self, mixture_waveform, source_waveforms):
    """Separates a mixture with oracle binary mask computed from references.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).
      source_waveforms: numpy.ndarray of shape (num_sources, num_samples).

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    stft_sources = self._stft_forward(tf.convert_to_tensor(source_waveforms))
    mask = self._oracle_binary_mask(tf.abs(stft_sources))

    stft_mixture = self._stft_forward(tf.convert_to_tensor(mixture_waveform))
    mask = tf.cast(mask, stft_mixture.dtype)
    separated_waveforms = self._stft_inverse(mask * stft_mixture)
    return separated_waveforms[:, :mixture_waveform.shape[-1]]
