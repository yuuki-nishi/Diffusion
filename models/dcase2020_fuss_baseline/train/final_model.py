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
"""A model to separate waveforms using TDCN++."""

import attr
import typing
import copy
import numpy as np
import tensorflow as tf
import tensorflow
import tensorflow_datasets as tfds
from . import consistency
from . import groupwise
from . import network
from . import network_config
from . import signal_transformer
from . import signal_util
from . import summaries
from . import summary_util
from . import shaper
from . import model
from . import mynetwork
from . import discmodel
from . import data_io
from . import callbacks
from . import metrics
from . import metrics as imported_metrics
import sys
from tensorflow.python.framework import meta_graph
from tensorflow.python import pywrap_tensorflow
from . import soundclass
import os
Shaper = shaper.Shaper
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# Define loss functions.
mse_loss = lambda source, separated: tf.nn.l2_loss(source - separated)


def _stabilized_log_base(x, base=10., stabilizer=1e-8):
  """Stabilized log with specified base."""
  logx = tf.math.log(x + stabilizer)
  logb = tf.math.log(tf.constant(base, dtype=logx.dtype))
  return logx / logb


def loss_label(x,y):
  batch=x.shape[0]
  num=x.shape[1]
  classnum=x.shape[2]
  x = tf.reshape(x,[batch*num,classnum])
  y = tf.reshape(y,[batch*num,classnum])
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,axis=-1,
    reduction=tensorflow.keras.losses.Reduction.NONE)
  ret = tf.reshape(bce(x,y),[batch,num])
  return ret
def mse_label(x,y):
  x=tf.cast(x, tf.float32),
  y=tf.cast(y, tf.float32),
  return tf.math.reduce_mean((x-y)**2,axis=-1)
def mse(x,y):
  x=tf.cast(x, tf.float32),
  y=tf.cast(y, tf.float32),
  print("mse shape")
  print(x)
  print(y)
  print(x.shape)
  print(y.shape)
  return tf.math.reduce_mean((x-y)**2)
def log_mse_loss(source, separated,max_snr=1e6, bias_ref_signal=None):
  """Negative log MSE loss, the negated log of SNR denominator."""
  err_pow = tf.reduce_sum(tf.square(source - separated), axis=-1)
  snrfactor = 10.**(-max_snr / 10.)
  if bias_ref_signal is None:
    ref_pow = tf.reduce_sum(tf.square(source), axis=-1)
  else:
    ref_pow = tf.reduce_sum(tf.square(bias_ref_signal), axis=-1)
  bias = snrfactor * ref_pow
  return 10. * _stabilized_log_base(bias + err_pow)
def zero_log_mse_loss(source, separated,max_snr=1e6, bias_ref_signal=None):
  return tf.zeros_like(tf.reduce_sum(tf.square(source - separated), axis=-1))


def _weights_for_nonzero_refs(source_waveforms):
  """Return shape (batch, source) weights for signals that are nonzero."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  return tf.greater(source_norms, 1e-8)


def _weights_for_num_sources(source_waveforms, num_sources):
  """Return shape (batch, source) weights for examples with num_sources."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  max_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)
  num_sources_per_example = tf.reduce_sum(
      tf.cast(tf.greater(source_norms, 1e-8), tf.float32),
      axis=1, keepdims=True)
  has_num_sources = tf.equal(num_sources_per_example, num_sources)
  return tf.tile(has_num_sources, (1, max_sources))

def different_loss_connect(x,y):
  return x+y

@attr.attrs
class HParams(object):
  """Model hyperparameters."""
  # Weight on loss component for zero reference signals.
  loss_zero_ref_weight = attr.attrib(type=float, default=1.0)
  # mix_weights_type = Type of weights to use for mixture consistency. Options:
  #     ``: No mixture consistency.
  #     `uniform`: Uniform weight of 1 / num_sources (also called unweighted).
  #     `magsq`: Weight for source j is \sum_{mic, time} \hat{x}_j ^ 2
  #              over \sum_{source, mic, time} \hat{x}_j ^ 2.
  #     `pred_source`: Predict weights with shape (batch, source, 1).
  mix_weights_type = attr.attrib(type=typing.Text, default='pred_source')
  # List of signal names, e.g. ['signal_1', 'signal_2'].
  signal_names = attr.attrib(type=typing.List[typing.Text],
                             default=['background', 'foreground_1',
                                      'foreground_2', 'foreground_3'])
  # A list of strings same length as signal_names specifying signal type, used
  # for groupwise permutation-invariance.
  signal_types = attr.attrib(type=typing.List[typing.Text],
                             default=['source'] * 4)
  # Sample rate of the input audio in hertz.
  sr = attr.attrib(type=float, default=16000.0)
  # Initial learning rate used by the optimizer.
  lr = attr.attrib(type=float, default=1e-4)
  #lr = attr.attrib(type=float, default=1e-12)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_steps = attr.attrib(type=int, default=2000000)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_rate = attr.attrib(type=float, default=0.5)
  # STFT window size in seconds.
  ws = attr.attrib(type=float, default=0.032)
  # STFT hop size in seconds.
  hs = attr.attrib(type=float, default=0.008)
  # disc parameter size
  discparam_size = attr.attrib(type=typing, default="small")
  # class number for disc
  classnum = attr.attrib(type=int, default=-1)
  # GRU unit
  unit =  attr.attrib(type=int, default=64)



def get_model_hparams():
  return HParams()

def printvars(path):
  print("vars from {}".format(path))
  #variables = tf.train.list_variables(path)
  ckpt_reader = tf.train.NewCheckpointReader(path)
  for v in  ckpt_reader.get_variable_to_shape_map():
    value = ckpt_reader.get_tensor(v)
    print(" {} : {}".format(v,value))
  print("end")
  return

def createdictfromckpt(path,s):
  reader = tf.train.NewCheckpointReader(path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  #print(var_to_shape_map)
  ret =dict()
  vars_to_load = [i[0] for i in tf.train.list_variables(path)]
  #assignment_map = {variable.op.name.replace(s,s+"_func"): variable for variable in tf.global_variables(scope=s) if variable.op.name.replace(s,s+"_func") in vars_to_load}
  for key in var_to_shape_map:
    #print("created map : {}".format(key))
    if (not "global_step" in key) and (s + "_func" in key):
      inkey = key.replace("_func","")
      #print("inkey : {}".format(inkey))
      if not(hasattr(inkey,"__hash__")):
        print("unhashable : {}".format(inkey))

      #print(value)#ok
      ret[key] = inkey
  #print(ret)
  return {s + "_func/": s + "/"}
  #return ret

def calc_loss(params,option=0):#if option is 1:for label

  def _calc_loss(x,y):
    labels = y["label"]
    source_waveforms = y["sources"]
    separated_waveforms = x["separated_waveforms"]
    probabilities = x["probabilities"]
    hparams = params['hparams']
    #print(labels.shape)
    #print(separated_waveforms.shape)
    #print(source_waveforms.shape)
    #print(probabilities.shape)
    batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
    # Permute separated to match references.
    labels = tf.reshape(tf.cast(labels,dtype=tf.float32),probabilities.shape)
    #print("label shape : {}".format(labels.shape))#label shape : (3, 5, 5)
    #print("source shape : {}".format(source_waveforms.shape))#source shape : (3, 5, 160000)
    def label_loss_func(label, probs):
      bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,axis=-1)
        #reduction=tensorflow.keras.losses.Reduction.NONE)#バージョン注意
      #labelloss=loss_label(label,probs)
      #mge_loss=log_mse_loss(source,separated,max_snr,bias_ref_signal)
      return params["omega"]*bce(label,probs)
      #return labelloss
    unique_signal_types = list(set(hparams.signal_types))
    loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}#辞書
    loss_fns_label = {signal_type: label_loss_func for signal_type in unique_signal_types}#辞書
    #probs=tf.map_fn(fn,separated_waveforms)
    #probs=tf.reshape(probs,tf.shape(probs_sub))
    #probs=probs_sub#信用してみる
    _, separated_waveforms,probabilities = groupwise.apply_withlabel(
        loss_fns, loss_fns_label,hparams.signal_types, source_waveforms, separated_waveforms,#これで本と同じようにあつかう#label_2はこう
        labels,probabilities,unique_signal_types)
    
    # Get batch size and (max) number of sources.
    num_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)


    weight = 1. / tf.cast(batch_size * num_sources, tf.float32)
    loss = tf.reduce_sum(log_mse_loss(source_waveforms,
                                      separated_waveforms,
                                      max_snr=30))
    loss_nonzero = tf.identity(weight * loss, name='loss_ref_nonzero')

    #注意

    labelloss=tf.constant(params["omega"])*loss_label(labels,probabilities)#labelperm_2
    recloss = params["omega2"]*loss_nonzero
    
    return recloss + labelloss

  return _calc_loss
def calc_loss_fortrain(params,option=0):#if option is 1:for label

  def _calc_loss(x,y):
    labels = y["label"]
    source_waveforms = y["sources"]
    separated_waveforms = x["separated_waveforms"]
    probabilities = x["probabilities"]
    hparams = params['hparams']
    #print(labels.shape)
    #print(separated_waveforms.shape)
    #print(source_waveforms.shape)
    #print(probabilities.shape)
    batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
    # Permute separated to match references.
    labels = tf.reshape(tf.cast(labels,dtype=tf.float32),probabilities.shape)
    #print("label shape : {}".format(labels.shape))#label shape : (3, 5, 5)
    #print("source shape : {}".format(source_waveforms.shape))#source shape : (3, 5, 160000)
    def label_loss_func(label, probs):
      bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,axis=-1,
        reduction=tensorflow.keras.losses.Reduction.NONE)#バージョン注意
      #labelloss=loss_label(label,probs)
      #mge_loss=log_mse_loss(source,separated,max_snr,bias_ref_signal)
      return params["omega"]*bce(label,probs)
      #return labelloss
    unique_signal_types = list(set(hparams.signal_types))
    loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}#辞書
    loss_fns_label = {signal_type: label_loss_func for signal_type in unique_signal_types}#辞書
    #probs=tf.map_fn(fn,separated_waveforms)
    #probs=tf.reshape(probs,tf.shape(probs_sub))
    #probs=probs_sub#信用してみる
    _, separated_waveforms,probabilities = groupwise.apply_withlabel(
        loss_fns, loss_fns_label,hparams.signal_types, source_waveforms, separated_waveforms,#これで本と同じようにあつかう#label_2はこう
        labels,probabilities,unique_signal_types)
    
    # Get batch size and (max) number of sources.
    num_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)


    weight = 1. / tf.cast(batch_size * num_sources, tf.float32)
    loss = tf.reduce_sum(log_mse_loss(source_waveforms,
                                      separated_waveforms,
                                      max_snr=30))
    loss_nonzero = tf.identity(weight * loss, name='loss_ref_nonzero')

    #注意

    labelloss=tf.constant(params["omega"])*loss_label(labels,probabilities)#labelperm_2
    recloss = params["omega2"]*loss_nonzero
    
    return recloss + labelloss

  return _calc_loss
# yuuki nishi added
@attr.attrs
class HParamswithNoise(HParams):
  """Model hyperparameters."""
  # List of signal names, e.g. ['signal_1', 'signal_2'].
  signal_names = attr.attrib(type=typing.List[typing.Text],
                             default=['background', 'foreground_1',
                                      'foreground_2', 'foreground_3','noise'])
  # A list of strings same length as signal_names specifying signal type, used
  # for groupwise permutation-invariance.
  signal_types = attr.attrib(type=typing.List[typing.Text],
                             default=['source'] * 5)
                             
def get_model_hparams_withNoise():
  return HParamswithNoise()

@attr.attrs
class HParamswithNoise_Max2(HParams):
  """Model hyperparameters."""
  # List of signal names, e.g. ['signal_1', 'signal_2'].
  signal_names = attr.attrib(type=typing.List[typing.Text],
                             default=['background', 'foreground_1',
                                      'foreground_2','noise'])
  # A list of strings same length as signal_names specifying signal type, used
  # for groupwise permutation-invariance.
  #signal_types = attr.attrib(type=typing.List[typing.Text],
  #                           default=['source'] * 5)
                             
def get_model_hparams_withNoise_Max2():
  return HParamswithNoise_Max2()
@attr.attrs
class HParamswithOutNum(HParams):
  #なぜか__init__がうまく動かない
  def init(self,maxoutnum):
    max_out = maxoutnum
    """Model hyperparameters."""
    # List of signal names, e.g. ['signal_1', 'signal_2'].
    names = ["sound{}".format(i) for i in range(10)]
    #self.signal_names = ['background', 'foreground_1',
    #                                    'foreground_2','foreground_3','foreground_4','foreground_5','foreground_6','foreground_7','foreground_8'][0:max_out]
    self.signal_names =names[0:max_out]
    # A list of strings same length as signal_names specifying signal type, used
    # for groupwise permutation-invariance.
    print("max_out {}".format(max_out))
    print(self.signal_names)
    self.signal_types = ['source'] * max_out
    return
                             
def get_model_hparams_withOutNum(outnum):
  ret = HParamswithOutNum()
  ret.init(outnum)
  return ret
"""class MyFinalModel(tf.keras.Model):

  def __init__(self, params):
    super(MyModel, self).__init__()
    self.SepModel = tf.keras.models.load_model(params["sepmodel"])
    self.DiscModel = tf.keras.models.load_model(params["discmodel"])
  def call(self, inputs):
    mixtured = inputs["mixtures"]
    separated_wavrforms = 
    return x
  def compute_loss(self, x, y, y_pred, sample_weight):
    loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
    loss += tf.add_n(self.losses)
    self.loss_tracker.update_state(loss)
    return loss

  def reset_metrics(self):
    self.loss_tracker.reset_states()

  @property
  def metrics(self):
    return [self.loss_tracker]"""
def predict_byamixture(model,mixtured,params):
  batch_size=params["train_batch_size"]
  dummylabels=tf.random.uniform(shape=[batch_size,params["outnum"],params["classnum"]])
  dummysources=tf.random.uniform(shape=[batch_size,params["outnum"],params["io_params"]["num_samples"]])
  seped,probs=model.predict({"mixtured":[mixtured]*batch_size,"label":dummylabels,"sources":dummysources,"option":[0]})
  seped = tf.reduce_mean(seped,axis=0)
  probs = tf.reduce_mean(probs,axis=0)
  return seped,probs
def predict_bymixtures(model,mixtures,params):
  print(mixtures)
  batch_size=params["train_batch_size"]
  dummylabels=tf.random.uniform(shape=[batch_size,params["outnum"],params["classnum"]])
  dummysources=tf.random.uniform(shape=[batch_size,params["outnum"],params["io_params"]["num_samples"]])
  inputs = {"mixtured":np.array(mixtures),"label":np.array(dummylabels),"sources":np.array(dummysources),"option":np.array([[1,1]]*batch_size)}
  #inputs = [mixtures.numpy(),dummylabels,dummysources,[[1,1]]*batch_size]
  ret = model.predict(inputs)
  #print("ret is {}".format(ret))
  seped=ret["separated_waveforms"]
  probs=ret["probabilities"]
  #seped = tf.reduce_mean(seped,axis=0)
  #probs = tf.reduce_mean(probs,axis=0)
  return seped,probs
def predict_bymixtures_andsource(model,mixtures,source,params):
  print(mixtures)
  batch_size=params["train_batch_size"]
  dummylabels=tf.random.uniform(shape=[batch_size,params["outnum"],params["classnum"]])
  dummysources=tf.random.uniform(shape=[batch_size,params["outnum"],params["io_params"]["num_samples"]])
  inputs = {"mixtured":np.array(mixtures),"label":np.array(dummylabels),"sources":np.array(source),"option":np.array([[0,0]]*batch_size)}#順番は変えさせる
  #inputs = [mixtures.numpy(),dummylabels,dummysources,[[1,1]]*batch_size]
  ret = model.predict(inputs)
  #print("ret is {}".format(ret))
  seped=ret["separated_waveforms"]
  probs=ret["probabilities"]
  #seped = tf.reduce_mean(seped,axis=0)
  #probs = tf.reduce_mean(probs,axis=0)
  return seped,probs
def predict_byawave(model,waveform,params):
  batch_size=params["train_batch_size"]
  dummylabels=tf.random.uniform(shape=[batch_size,params["outnum"],params["classnum"]])
  dummymixtured=tf.random.uniform(shape=[batch_size,params["io_params"]["num_samples"]])
  #seped,probs=model.predict({"mixtured":dummymixtured,"label":[dummylabels]*batch_size,"sources":[waveform]*batch_size*params["outnum"],"option":[1]})
  modelinput=[np.array(dummymixtured),np.array(dummylabels),np.array([waveform]*batch_size),np.array([[1,1]]*batch_size)]#順番はそのままで出力させる、sourceを出力判定させる
  seped,probs=model.predict_on_batch(modelinput)
  seped = tf.reduce_mean(seped,axis=0)
  probs = tf.reduce_mean(probs,axis=[0])
  return probs
def predict_bywaves(model,waveforms,params):
  batch_size=params["train_batch_size"]
  dummylabels=tf.random.uniform(shape=[batch_size,params["outnum"],params["classnum"]])
  dummymixtured=tf.random.uniform(shape=[batch_size,params["io_params"]["num_samples"]])
  ret=model.predict({"mixtured":np.array(dummymixtured),"label":np.array(dummylabels),"sources":np.array(waveforms),"option":np.array([[1,1]]*batch_size)})
  #modelinput=[np.array(dummymixtured),np.array(dummylabels),np.array(waveforms),np.array([[1,1]]*batch_size)]#順番はそのままで出力させる、sourceを出力判定させる
  #ret=model.predict_on_batch(modelinput)
  seped=ret["separated_waveforms"]
  probs=ret["probabilities"]
  return probs
def sortbylabel(metrics):
  return
def metrics_forlog(config,params):#とりあえず並び替えは一旦保留
  """Permutation-invariant SI-SNR, powers, and under/equal/over-separation."""

 
  sourcenum = params["outnum"]
  classnum = params["classnum"]
    
  def sisnr_separated(y_true, y_pred):
    sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      y_pred, y_true)
    return sisnr_separated

  def sisnr_improvement(y_true, y_pred):
    sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      y_pred, y_true)
    mixture_waveform = tf.reduce_sum(y_true,axis=-2)
    sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      tf.tile(tf.expand_dims(mixture_waveform,axis=-2), (1,sourcenum,1)),
      y_true)
    return sisnr_separated-sisnr_mixture

  def sisnr_mixture(y_true, y_pred):
    mixture_waveform = tf.reduce_sum(y_true,axis=-2)
    sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      tf.tile(tf.expand_dims(mixture_waveform,axis=-2), (1,sourcenum,1)),
      y_true)
    return sisnr_mixture
  
  """ret={"sisnr_separated" : sisnr_separated,
       "sisnr_improvement" : sisnr_improvement,
       "sisnr_mixture" : sisnr_mixture}"""
  ret=[sisnr_separated,sisnr_improvement,sisnr_mixture]
  return ret
def compute_metrics(source_waveforms, separated_waveforms, mixture_waveform,probs,labels,config,params):#とりあえず並び替えは一旦保留
  """Permutation-invariant SI-SNR, powers, and under/equal/over-separation."""

  # Align separated sources to reference sources.
  #print(source_waveforms.shape)
  #print(separated_waveforms.shape)
  """perm_inv_loss = permutation_invariant.wrap(
  lambda tar, est: -metrics.signal_to_noise_ratio_gain_invariant(est, tar))
  _, separated_waveforms = perm_inv_loss(source_waveforms,
                                    separated_waveforms)"""

  """perm_inv_loss_withlabel = permutation_invariant.wrap_label_useonlystandard(
      lambda tar, est: -metrics.signal_to_noise_ratio_gain_invariant(est, tar),
      lambda lprob,llabel: np.sum(np.multiply(lprob,llabel)))
  _, separated_waveforms,_ = perm_inv_loss(source_waveforms[tf.newaxis],
                                                            separated_waveforms[tf.newaxis],
                                                            [label],#add batch axis
                                                            [prob])"""
  #sourcenum=signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
  #samplenum=signal_util.static_or_dynamic_dim_size(source_waveforms, -1)
  sourcenum=tf.shape(source_waveforms)[-2]
  samplenum=tf.shape(source_waveforms)[-1]
  classnum = labels.shape[-1]
  hparams = params['hparams']
  unique_signal_types = list(set(hparams.signal_types))
  def label_loss_func(label, probs):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,axis=-1,name="labelloss_fortraining",
      reduction=tensorflow.keras.losses.Reduction.NONE)
    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,name="labelloss_fortraining")
    return bce(label,probs)
  loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}#辞書
  loss_fns_zeros = {signal_type: zero_log_mse_loss for signal_type in unique_signal_types}#辞書
  loss_fns_label = {signal_type: label_loss_func for signal_type in unique_signal_types}#辞書
  print(source_waveforms.shape)
  print(labels.shape)
  tmps = separated_waveforms[1]
  #テスト用に順番を変える
  pseparated_waveforms=tf.reverse(separated_waveforms,axis=[0])
  pprobs=tf.reverse(probs,axis=[0])
  #pseparated_waveforms=separated_waveforms
  #pprobs=probs
  _, permed_separated_waveforms = groupwise.apply(
      loss_fns, hparams.signal_types, tf.expand_dims(source_waveforms,axis=0), tf.expand_dims(pseparated_waveforms,axis=0),
      unique_signal_types)
  """_, permed_probs = groupwise.apply(
      loss_fns_label, hparams.signal_types, tf.expand_dims(labels,axis=0), tf.expand_dims(probs,axis=0),
      unique_signal_types)"""
      
  _, _,permed_probs,idxs_nd = groupwise.apply_withlabel(
      loss_fns_zeros, loss_fns_label,hparams.signal_types, tf.expand_dims(source_waveforms,axis=0), tf.expand_dims(separated_waveforms,axis=0),#これで本と同じようにあつかう#label_2はこう
      tf.expand_dims(labels,axis=0),tf.expand_dims(pprobs,axis=0),unique_signal_types)
  diffsep = (permed_separated_waveforms - separated_waveforms)**2
  diffprobs = (permed_probs - probs)**2
  diffsum = tf.reduce_mean(diffsep)+tf.reduce_mean(diffprobs)

  separated_waveforms = separated_waveforms  # Remove batch axis.
  # Compute separated and source powers.
  power_separated = tf.reduce_mean(separated_waveforms ** 2, axis=-1)
  power_sources = tf.reduce_mean(source_waveforms ** 2, axis=-1)

  # Compute SI-SNR.
  sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      separated_waveforms, source_waveforms)
  sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      #tf.tile(mixture_waveform[tf.newaxis], (source_waveforms.shape[0], 1)),
      #tf.tile(mixture_waveform, (1,sourcenum, 1)),
      tf.tile(tf.expand_dims(mixture_waveform,axis=-2), (sourcenum,1)),
      source_waveforms)


  hparams = params['hparams']

  
  argmax_probs=tf.argmax(probs,axis=-1)
  argmax_labels=tf.argmax(labels,axis=-1)
  isequalprob=tf.math.equal(argmax_probs,argmax_labels)
  isequal=tf.where(isequalprob,  tf.ones_like(argmax_probs),   tf.zeros_like(argmax_probs))
  isnotequal=tf.where(isequalprob, tf.zeros_like(argmax_probs), tf.ones_like(argmax_probs))
  ret={'sisnr_separated': sisnr_separated,
        'sisnr_mixture': sisnr_mixture,
        'sisnr_improvement': sisnr_separated - sisnr_mixture,
        'power_separated': power_separated,
        'power_sources': power_sources,
        'isequal': isequal,
        'isnotequal': isnotequal,
        "diffsum":diffsum}
  return ret

def model_fn(params):
  print("params")
  print(params)
  #tf.enable_eager_execution()

  
  """Constructs a spectrogram_lstm model with summaries.

  Args:
    features: Dictionary {name: Tensor} of model inputs.
    labels: Any training-only inputs.
    mode: Build mode, one of tf.estimator.ModeKeys.
    params: Dictionary of Model hyperparameters.

  Returns:
    EstimatorSpec describing the model.
  """
  print(params)
  hparams = params['hparams']
  sample_num=160000
  outnum = params["outnum"]
  classnum = params["classnum"]
  print("model start")

  train_params = copy.deepcopy(params)
  train_params["split"] = "train"
  batch_size = params["train_batch_size"]
  train_params['batch_size'] = batch_size
  train_params['example_num'] = params["train_examples"]


  dataset,dataset_forsource,dataset_forlabel = data_io.input_fn_dict(train_params)
  print(dataset)
  datanum=dataset.cardinality().numpy()
  """for d in dataset:
    #print(d["source_image"].shape)
    dataset_waveform=np.append(dataset_waveform,[d["source_image"]],axis=0)
    dataset_label=np.append(dataset_label,[d["label"]],axis=0)"""
  #データセットのテスト
  #datatest(dataset,params)
  print("size1")
  print("source shape")
  print("size2")
  print(sys.getsizeof(dataset))
  print("size3")

  inputs_sep = tf.keras.Input(shape=(sample_num),batch_size=batch_size,name="mixtured")
  inputs_sep_forsource = tf.keras.Input(shape=(params["outnum"],sample_num),batch_size=batch_size,name="sources")
  labels = tf.keras.Input(shape=(params["outnum"],params["classnum"]),batch_size=batch_size,name="label")
  inputs_option = tf.keras.Input(shape=(2,),batch_size=batch_size,name="option")#[discinput,permutaion,]
  config = mynetwork.MyConfig(outnum=params["outnum"])
  separated_waveforms = mynetwork.Separate(inputs_sep,params,config)
  print("sep shape : {}".format(separated_waveforms))
  
  SepModel = tf.keras.Model(inputs=inputs_sep, outputs=separated_waveforms)
  #tmppath=tf.train.latest_checkpoint(params["sepdir"]+"/weights")
  alligned_waveforms = tf.reshape(separated_waveforms,[batch_size*params["outnum"],sample_num])
  discinput_option=inputs_option[:,0]#0なら訓練時そのまま,1ならsourceを判定する
  discinput_option = tf.tile(tf.expand_dims(discinput_option,axis=1),[params["outnum"],sample_num])#次元の帳尻合わせ
  print("allign hsape {}".format(alligned_waveforms.shape))
  discinput = tf.math.multiply((1-discinput_option),alligned_waveforms) + tf.math.multiply(discinput_option,tf.reshape(inputs_sep_forsource,[batch_size*params["outnum"],sample_num]))
  print("discinput hsape {}".format(discinput))
  probabilities = discmodel.get_probs(discinput,hparams)
  DiscModel = tf.keras.Model(inputs=discinput, outputs=probabilities)
  #並べ替えたのを元に戻す
  probabilities = tf.reshape(probabilities,[batch_size,outnum,params["classnum"]])
  print("prob shae : {}".format(probabilities.shape))
  
  #更新設定
  SepModel.trainable = True
  DiscModel.trainable = False

    
  unique_signal_types = list(set(hparams.signal_types))
  def label_loss_func(label, probs):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,axis=-1,name="labelloss_fortraining")
      #reduction=tensorflow.keras.losses.Reduction.NONE)#バージョン注意
    return params["omega"]*bce(label,probs)
  def label_loss_func_notomega(label, probs):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,axis=-1,name="labelloss_fortraining_notomega")
      #reduction=tensorflow.keras.losses.Reduction.NONE)#バージョン注意
    return bce(label,probs)
  def label_loss_func2(label, probs):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,axis=-1,name="labelloss_fortraining2")
      #reduction=tensorflow.keras.losses.Reduction.NONE)#バージョン注意
    return params["omega"]*bce(label,probs)
  def rec_loss_func(sources, seps):
    sourcenum=tf.shape(sources)[-2]
    samplenum=tf.shape(sources)[-1]
    mixture_waveform = tf.reduce_sum(sources,axis=-2)

    # Compute SI-SNR.
    sisnr_separated = imported_metrics.signal_to_noise_ratio_gain_invariant(
        seps, sources)
    sisnr_mixture = imported_metrics.signal_to_noise_ratio_gain_invariant(
        tf.tile(tf.expand_dims(mixture_waveform,axis=-2), (1,sourcenum,1)),
        sources)
    return model.Separated_loss_notperm(params)(sources, seps)
  def _forfit_label_loss_func(y_true,y_pred):
    labels=y_true["label"]
    probs=y_pred["probabilities"]
    return label_loss_func(labels, probs)
  def _forfit_rec_loss_func(y_true,y_pred):
    sources=y_true["sources"]
    seps=y_pred["separated_waveforms"]
    return rec_rec_func(sources, seps)
    
  loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}#辞書
  loss_fns_label = {signal_type: label_loss_func for signal_type in unique_signal_types}#辞書
  #resolve permute
  print("permute shape")
  print(inputs_sep_forsource.shape)
  print(separated_waveforms.shape)
  print(labels.shape)
  print(probabilities.shape)
  """_, permed_separated_waveforms = groupwise.apply(
      loss_fns, hparams.signal_types, inputs_sep_forsource, separated_waveforms,#これで本と同じようにあつかう#label_2はこう
      unique_signal_types)"""
  _, permed_separated_waveforms,permed_probabilities,ndx = groupwise.apply_withlabel(
      loss_fns, loss_fns_label,hparams.signal_types, inputs_sep_forsource, separated_waveforms,#これで本と同じようにあつかう#label_2はこう
      labels,probabilities,unique_signal_types)
  permutation_option=inputs_option[:,1]#0なら並び替える、1なら順番そのまま
  permutation_option=tf.expand_dims(permutation_option,axis=1)#次元の帳尻合わせ
  permutation_option=tf.expand_dims(permutation_option,axis=2)
  permutation_option_forwave = tf.tile(permutation_option,[1,outnum,sample_num])
  permutation_option_forprob = tf.tile(permutation_option,[1,outnum,classnum])
  separated_waveforms = tf.math.multiply((1-permutation_option_forwave),permed_separated_waveforms) + tf.math.multiply(permutation_option_forwave,separated_waveforms)
  probabilities = tf.math.multiply((1-permutation_option_forprob),permed_probabilities) + tf.math.multiply(permutation_option_forprob,probabilities)
  
  def my_summary(x):
    probs,labels,seps,sources = x[0],x[1],x[2],x[3],
    rec_loss = rec_loss_func(sources,seps)
    label_loss = label_loss_func(labels,probs)
    sourcenum=tf.shape(sources)[-2]
    samplenum=tf.shape(sources)[-1]
    mixture_waveform = tf.reduce_sum(sources,axis=-2)

    # Compute SI-SNR.
    sisnr_separated = imported_metrics.signal_to_noise_ratio_gain_invariant(
        seps, sources)
    sisnr_mixture = imported_metrics.signal_to_noise_ratio_gain_invariant(
        tf.tile(tf.expand_dims(mixture_waveform,axis=-2), (1,sourcenum,1)),
        sources)
    
    tf.summary.scalar(name = "sisnr_separated",data=tf.reduce_mean(sisnr_separated))
    tf.summary.scalar(name = "sisnr_mixture",data=tf.reduce_mean(sisnr_mixture))
    tf.summary.scalar(name = "sisnr_improvement",data=tf.reduce_mean(sisnr_separated - sisnr_mixture))
    tf.summary.scalar(name = "losslog/rec_loss",data=tf.reduce_mean(rec_loss),description="reconstruction loss")
    tf.summary.scalar(name = "losslog/label_loss",data=tf.reduce_mean(label_loss),description="binarycrossentropy loss loss")
    return [probs,labels,seps,sources,rec_loss,label_loss]

  #tensorboardに出力させる
  """summary_ret= tf.keras.layers.Lambda(my_summary)([probabilities,labels,separated_waveforms,inputs_sep_forsource])
  probabilities,_,separated_waveforms,_,rec_loss,label_loss =summary_ret[0],summary_ret[1],summary_ret[2], \
    summary_ret[3],summary_ret[4],summary_ret[5]"""

  separated_waveforms = tf.reshape(separated_waveforms,[batch_size,outnum,sample_num],name="separated_waveforms")
  probabilities = tf.reshape(probabilities,[batch_size,outnum,params["classnum"]],name="probabilities")

  full_model = tf.keras.Model(inputs={"mixtured":inputs_sep,"label":labels, \
    "sources":inputs_sep_forsource,"option":inputs_option},
    outputs={"separated_waveforms":separated_waveforms,"probabilities":probabilities})
    #outputs=[separated_waveforms,probabilities])
  wave_metrics = metrics_forlog(config,params)
  wave_metrics.append(rec_loss_func)
  print("sepdir : {}".format(params["sepdir"]+"/weights.tf"))
  SepModel.load_weights(params["sepdir"]+"/weights.tf")
  DiscModel.load_weights(params["discdir"]+"/weights.tf")
  full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.5*1e-4),
              loss={"separated_waveforms":rec_loss_func,"probabilities":label_loss_func},
              #loss=[log_mse_loss,forlossfunc_labelloss])
              metrics={"separated_waveforms":wave_metrics,"probabilities":[label_loss_func2,label_loss_func_notomega]})

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=params["model_dir"] + "/checkpoint",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
  print("model summery:")
  print(full_model.summary())
  ret = model.calckerassize.get_model_memory_usage(params["train_batch_size"],full_model)
  #tensorboardのcallback
  tb_cb = tf.keras.callbacks.TensorBoard(log_dir=params["model_dir"]+"/logs", update_freq="batch")
  #GB単位
  print(ret)
  eval_params = copy.deepcopy(params)
  eval_params["split"] = "eval"
  eval_params["batch_size"] = params["eval_batch_size"]
  eval_params["example_num"] = params["eval_examples"]

  evaldataset,evaldataset_forsource,evaldataset_forlabel = data_io.input_fn_dict(eval_params)

  #datatest(evaldataset,params)#eval,trainのデータも大丈夫っぽい
  
  print("fit start")
  """for l in full_model.layers:
    print(l)
    print(l.get_weights())"""

  print("dataset")
  #okっぽい?
  """for inputs in dataset.take(1):  # only take first element of dataset
    print(inputs)
    mixtured=inputs["mixtured"]
    numpy_images = inputs["sources"]
    numpy_labels = inputs["label"]
    print(numpy_images)
    print(numpy_labels)
    testoutput = tf.random.uniform(shape=tf.shape(numpy_images)).numpy()
    #testoutput = tf.reduce_sum(testoutput,axis=-2)
    testoutputlabel = tf.random.uniform(shape=tf.shape(numpy_labels)).numpy()
    print("testloss : {}".format(loss({"separated_waveforms":testoutput,"probabilities":testoutputlabel},inputs)))
    #test1,test2,test22,test23,test3,test4,testpredict = full_model.predict(numpy_images)
    testpredict = full_model.predict(inputs)
    print("test output : {}".format(testpredict))
    t1 = testpredict["separated_waveforms"]
    t2 = testpredict["probabilities"]
    print("test output1 : {}".format(t1.shape))
    print("test output2 : {}".format(t2.shape))"""
  
  if False:
    print(params["model_dir"])
    full_model.load_weights(params["model_dir"]+"/weights.tf")
    #full_model.load_weights(params["model_dir"])
    print("load ok")
  else:
    #まったく訓練しないと、普通にいい出力結果だった
    full_model.fit(
                x=dataset,
                #y={"separated_waveforms":dataset_forsource,"labels":dataset_forlabel},#ここの数字が3らしい
                #y=[dataset_forsource,dataset_forlabel],#ここの数字が3らしい
                epochs = params["train_epoch"],
                verbose = 2,
                batch_size=params["train_batch_size"],
                callbacks=[model_checkpoint_callback,tb_cb],
                validation_data=evaldataset)
    full_model.save_weights(params["model_dir"]+"/weights.h5")
    full_model.save_weights(params["model_dir"]+"/weights.tf")
    #full_model.save(params["model_dir"]+"/savedmodel.h5")
  
  #test disc load params
  """print("test prob precidt")
  for tmpdata in evaldataset.take(1):
    print(tmpdata)
    print("test disc predict")
    print(predict_bymixtures(full_model,tmpdata[0]["mixtured"],params))
    print("test disc label")
    print(tmpdata[0]["label"])"""
  print("after params")
  """for l in full_model.layers:
    print(l)
    print(l.get_weights())"""
  count = 0
  allcount=0
  time=0
  allmetrics = {}
  metricskeys = ['sisnr_separated',
                'sisnr_mixture',
                'sisnr_improvement',
                'power_separated',
                'power_sources',
                'isequal',
                'isnotequal']
  for key in metricskeys:
    allmetrics[key] = [[]]*classnum#[sample,class]
  allmetrics["probs"] = [[]]*classnum#[sample,class]
  diffsumes=[]
  for data in evaldataset.take(100):
    images = data[0]["mixtured"]
    sources = data[0]["sources"]
    labels = data[0]["label"]
    testpredict,testprobs = predict_bymixtures_andsource(full_model,images,sources,params)
    assertprobs = predict_bywaves(full_model,sources,params)
    #print(testpredict)
    time += 1
    tmpdata=sources
    #print(tmpdata)
    #testpredict = tf.reshape(testpredict,[params["train_batch_size"],params["outnum"],160000])
    testpredict = tf.reshape(testpredict,[batch_size,params["outnum"],sample_num])
    for i in range(batch_size):
      for j in range(params["outnum"]):
        if time <= 10:
          path = params["model_dir"] + "/tmpout{}_{}.wav".format(i,j)
          model.writewav(testpredict[i][j],path)
    for i in range(batch_size):
      metrics = compute_metrics(tmpdata[i],testpredict[i],images[i],testprobs[i],labels[i],config,params)
      print("diffsum : {}".format(metrics["diffsum"]))
      diffsumes.append(metrics["diffsum"])
      nowoutindex=0
      for label in labels[i]:#batch,outnum,classnumのはず
        classindex_fromlabel = np.argmax(np.array(label))
        allmetrics["probs"][classindex_fromlabel] = np.append(allmetrics["probs"][classindex_fromlabel],testprobs[i][nowoutindex])
        nowoutindex += 1
      #print("metrics : {}".format(metrics))
      for key in metricskeys:#keyは5種類
        #print("mean test {} : {}".format(key,np.mean(metrics[key])))
        nowoutindex=0
        for label in labels[i]:#batch,outnum,classnumのはず
          classindex_fromlabel = np.argmax(np.array(label))
          forappend=metrics[key][nowoutindex]
          allmetrics[key][classindex_fromlabel] = np.append(allmetrics[key][classindex_fromlabel],forappend)
          nowoutindex = nowoutindex + 1
    
    testpredict,testprobs = predict_bymixtures_andsource(full_model,images,sources,params)
    """for batchindex in range(len(images)):
      allcount+=1
      labelindex = np.argmax(labels[batchindex])
      probindex = np.argmax(testpredict[batchindex])
      if labelindex == probindex:
        count += 1"""
  metricskeys.append("probs")
        
  for key in metricskeys:
    print(np.array(allmetrics[key]))
    #print("shape : {}".format(np.array(allmetrics[key]).shape))
    #allmetrics[key] = np.mean(allmetrics[key],axis=-1)
  path_w = params["model_dir"]+"/evaldata.txt"
  with open(path_w, mode='w') as f:
      for key in metricskeys:
        line = "{} : {}\n".format(key,np.mean(allmetrics[key]))
        print(line)
        f.write(line)
        line = "var {} : {}\n".format(key,np.var(allmetrics[key],axis=-1))
        print(line)
        f.write(line)
        avestr = "average over class: {}, var : {}\n".format(np.mean(allmetrics[key]),np.var(allmetrics[key]))
        print(avestr)
        f.write(avestr)
      acc = np.sum(allmetrics["isequal"])/(np.sum(allmetrics["isequal"])+np.sum(allmetrics["isnotequal"]))
      print("acc : {}".format(acc))
      f.write("acc : {}".format(acc))
      print("diffsum : {}".format(np.mean(diffsumes)))
      f.write("diffsum : {}".format(np.mean(diffsumes)))
      
      print("raw metrics")
      """f.write("raw metrics")
      
      for key in metricskeys:
        if key in allmetrics2.keys():
          line = "{} : {}\n".format(key,tf.reduce_mean(allmetrics2[key]))
          print(line)
          f.write(line)"""
  
  #full_model.save(params["model_dir"])
  print("all end")
  