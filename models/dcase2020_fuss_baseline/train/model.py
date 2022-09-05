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
import sys
import numpy as np
import tensorflow as tf
import copy
from . import calckerassize
from . import consistency
from . import groupwise
from . import network
from . import data_io
from . import network_config
from . import signal_transformer
from . import signal_util
from . import summaries
from . import summary_util
from . import shaper
from . import mynetwork
from . import callbacks
from . import metrics
from . import permutation_invariant
from . import model_sub

import soundfile
Shaper = shaper.Shaper

# Define loss functions.
mse_loss = lambda source, separated: tf.nn.l2_loss(source - separated)

def _weights_for_nonzero_refs(source_waveforms):
  """Return shape (batch, source) weights for signals that are nonzero."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  return tf.greater(source_norms, 1e-8)


# 一番小さい音から計算している
def _weights_for_active_seps(power_sources, power_separated):
  """Return (source,) weights for active separated signals."""
  min_power = tf.reduce_min(power_sources, axis=-1, keepdims=True)
  return tf.greater(power_separated, 0.5 * min_power)
  #return tf.greater(power_separated, 0.5 * min_power)

def _weights_for_num_sources(source_waveforms, num_sources):
  """Return shape (batch, source) weights for examples with num_sources."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  max_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)
  num_sources_per_example = tf.reduce_sum(
      tf.cast(tf.greater(source_norms, 1e-8), tf.float32),
      axis=1, keepdims=True)
  has_num_sources = tf.equal(num_sources_per_example, num_sources)
  return tf.tile(has_num_sources, (1, max_sources))
def raw_compute_metrics(source_waveforms, separated_waveforms, mixture_waveform):
  """Permutation-invariant SI-SNR, powers, and under/equal/over-separation."""

  # Align separated sources to reference sources.
  perm_inv_loss = permutation_invariant.wrap(
      lambda tar, est: -metrics.signal_to_noise_ratio_gain_invariant(est, tar))
  _, separated_waveforms = perm_inv_loss(source_waveforms[tf.newaxis],
                                         separated_waveforms[tf.newaxis])
  separated_waveforms = separated_waveforms[0]  # Remove batch axis.

  # Compute separated and source powers.
  power_separated = tf.reduce_mean(separated_waveforms ** 2, axis=-1)
  power_sources = tf.reduce_mean(source_waveforms ** 2, axis=-1)

  # Compute weights for active (separated, source) pairs where source is nonzero
  # and separated power is above threshold of quietest source power - 20 dB.
  weights_active_refs = _weights_for_nonzero_refs(source_waveforms)
  weights_active_seps = _weights_for_active_seps(
      tf.boolean_mask(power_sources, weights_active_refs), power_separated)
  weights_active_pairs = tf.logical_and(weights_active_refs,
                                        weights_active_seps)

  # Compute SI-SNR.
  sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      separated_waveforms, source_waveforms)
  num_active_refs = tf.reduce_sum(tf.cast(weights_active_refs, tf.int32))
  num_active_seps = tf.reduce_sum(tf.cast(weights_active_seps, tf.int32))
  num_active_pairs = tf.reduce_sum(tf.cast(weights_active_pairs, tf.int32))
  sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      tf.tile(mixture_waveform[tf.newaxis], (source_waveforms.shape[0], 1)),
      source_waveforms)

  # Compute under/equal/over separation.
  under_separation = tf.cast(tf.less(num_active_seps, num_active_refs),
                             tf.float32)
  equal_separation = tf.cast(tf.equal(num_active_seps, num_active_refs),
                             tf.float32)
  over_separation = tf.cast(tf.greater(num_active_seps, num_active_refs),
                            tf.float32)

  return {'sisnr_separated': sisnr_separated,
          'sisnr_mixture': sisnr_mixture,
          'sisnr_improvement': sisnr_separated - sisnr_mixture,
          'power_separated': power_separated,
          'power_sources': power_sources,
          'under_separation': under_separation,
          'equal_separation': equal_separation,
          'over_separation': over_separation,
          'weights_active_refs': weights_active_refs,
          'weights_active_seps': weights_active_seps,
          'weights_active_pairs': weights_active_pairs,
          'num_active_refs': num_active_refs,
          'num_active_seps': num_active_seps,
          'num_active_pairs': num_active_pairs}

def compute_metrics(source_waveforms, separated_waveforms, mixture_waveform,config,params):
  """Permutation-invariant SI-SNR, powers, and under/equal/over-separation."""

  # Align separated sources to reference sources.
  #print(source_waveforms.shape)
  #print(separated_waveforms.shape)
  perm_inv_loss = permutation_invariant.wrap(
      lambda tar, est: -metrics.signal_to_noise_ratio_gain_invariant(est, tar))
  _, separated_waveforms = perm_inv_loss(source_waveforms,
                                        separated_waveforms)
  """else:
    #print("oooookkkkkkk")#ok
    perm_inv_loss = permutation_invariant.wrap_label_useonlystandard(
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
  #print(sourcenum)
  #print(samplenum)
  #separated_waveforms=tf.reshape(separated_waveforms,[sourcenum,1,samplenum])
    

  separated_waveforms = separated_waveforms  # Remove batch axis.
  """if discmodel is not None:
    reshaped=tf.reshape(separated_waveforms,[sourcenum,1,samplenum])
    permed_prob=discmodel.getprobability(reshaped)"""
    
  # Compute separated and source powers.
  power_separated = tf.reduce_mean(separated_waveforms ** 2, axis=-1)
  power_sources = tf.reduce_mean(source_waveforms ** 2, axis=-1)

  # Compute SI-SNR.
  sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      separated_waveforms, source_waveforms)
  sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      #tf.tile(mixture_waveform[tf.newaxis], (source_waveforms.shape[0], 1)),
      tf.tile(mixture_waveform, (1,sourcenum, 1)),
      source_waveforms)


  hparams = params['hparams']
  transformer = signal_transformer.SignalTransformer(
    sample_rate=hparams.sr,
    window_time_seconds=hparams.ws,
    hop_time_seconds=hparams.hs)
  source_spectrograms=transformer.forward(source_waveforms)
  mixture_spectrograms =transformer.forward(mixture_waveform )
  separated_spectrograms=transformer.forward(separated_waveforms)
  
  ret={'sisnr_separated': sisnr_separated,
        'sisnr_mixture': sisnr_mixture,
        'sisnr_improvement': sisnr_separated - sisnr_mixture,
        'power_separated': power_separated,
        'power_sources': power_sources,}
  return ret

def _stabilized_log_base(x, base=10., stabilizer=1e-8):
  """Stabilized log with specified base."""
  logx = tf.math.log(x + stabilizer)
  logb = tf.math.log(tf.constant(base, dtype=logx.dtype))
  return logx / logb

def writewav(data,path):
  soundfile.write(path, data, 16000, subtype='PCM_16')
  return


def log_mse_loss(source, separated, max_snr=1e6, bias_ref_signal=None):
  """Negative log MSE loss, the negated log of SNR denominator."""
  err_pow = tf.reduce_sum(tf.square(source - separated), axis=-1)
  snrfactor = 10.**(-max_snr / 10.)
  if bias_ref_signal is None:
    ref_pow = tf.reduce_sum(tf.math.square(source), axis=-1)
  else:
    ref_pow = tf.reduce_sum(tf.math.square(bias_ref_signal), axis=-1)
  bias = snrfactor * ref_pow
  return 10. * _stabilized_log_base(bias + err_pow)




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
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_steps = attr.attrib(type=int, default=2000000)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_rate = attr.attrib(type=float, default=0.5)
  # STFT window size in seconds.
  ws = attr.attrib(type=float, default=0.032)
  # STFT hop size in seconds.
  hs = attr.attrib(type=float, default=0.008)


def get_model_hparams():
  return HParams()
def Separated_loss_notperm(params):
  def _Separated_loss(sources,separated_waveforms):#y_true, y_predの形
    sources = tf.squeeze(sources)
    separated_waveforms = tf.squeeze(separated_waveforms)
    source_waveforms = sources
    labels = sources
    sample_num=160000
    mixture_waveforms = tf.reduce_sum(source_waveforms,axis=1)#sourceから合計する
    mixture_waveforms = tf.expand_dims(source_waveforms,axis=1)#sourceから合計する
    
    hparams = params['hparams']
    batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
    
    shape = [params["train_batch_size"],params["outnum"],params["io_params"]["num_samples"]]

    separated_waveforms = tf.reshape(separated_waveforms,shape)
    source_waveforms = tf.reshape(source_waveforms,shape)
    #mixture_waveforms= tf.reshape(mixture_waveforms,[params["io_params"]["num_samples"]])
    unique_signal_types = list(set(hparams.signal_types))
    
    weight = 1. / tf.cast(params["train_batch_size"] * params["outnum"], tf.float32)
    loss = weight*tf.reduce_sum(log_mse_loss(source_waveforms,
                                      separated_waveforms,
                                      max_snr=20,
                                      bias_ref_signal =mixture_waveforms ))
    return loss
  return _Separated_loss
def Separated_loss(params):
  def _Separated_loss(sources,separated_waveforms):#y_true, y_predの形
    sources = tf.squeeze(sources)
    separated_waveforms = tf.squeeze(separated_waveforms)
    source_waveforms = sources
    labels = sources
    mixture_waveforms = tf.reduce_sum(source_waveforms,axis=1)#sourceから合計する
    mixture_waveforms = tf.expand_dims(source_waveforms,axis=1)#sourceから合計する
    hparams = params['hparams']
    batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
    

    shape = [params["train_batch_size"],params["outnum"],params["io_params"]["num_samples"]]

    separated_waveforms = tf.reshape(separated_waveforms,shape)
    source_waveforms = tf.reshape(source_waveforms,shape)
    #mixture_waveforms= tf.reshape(mixture_waveforms,[params["io_params"]["num_samples"]])
    unique_signal_types = list(set(hparams.signal_types))
    loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}#辞書
    
    _, separated_waveforms = groupwise.apply(#rec
      loss_fns, hparams.signal_types, source_waveforms, separated_waveforms,
      unique_signal_types)
    weight = 1. / tf.cast(params["train_batch_size"] * params["outnum"], tf.float32)
    loss = weight*tf.reduce_sum(log_mse_loss(source_waveforms,
                                      separated_waveforms,
                                      max_snr=20,
                                      bias_ref_signal =mixture_waveforms ))
    return loss
  return _Separated_loss
def separate_waveforms(mixture_waveforms, hparams):
  """Computes and returns separated waveforms.

  Args:
    mixture_waveforms: Waveform of audio to separate, shape (batch, mic, time).
    hparams: Model hyperparameters.
  Returns:
    Separated audio tensor, shape (batch, source, time), same type as mixture.
  """
  num_sources = len(hparams.signal_names)
  #num_sources = hparams['outnum']
  print("num_sources : {}".format(num_sources))
  num_mics = signal_util.static_or_dynamic_dim_size(mixture_waveforms, 1)#おそらく1
  shaper = Shaper({'source': num_sources, '1': 1})

  # Compute encoder coefficients.
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)
  mixture_coeffs = transformer.forward(mixture_waveforms)
  inverse_transform = transformer.inverse
  mixture_coeffs_input = tf.abs(mixture_coeffs)
  mixture_coeffs_input = network.LayerNormalizationScalarParams(
      axis=[-3, -2, -1],
      name='layer_norm_on_mag').apply(mixture_coeffs_input)
  shaper.register_axes(mixture_coeffs, ['batch', 'mic', 'frame', 'bin'])
  mixture_coeffs_input = shaper.change(mixture_coeffs_input[:, :, tf.newaxis],
                                       ['batch', 'mic', '1', 'frame', 'bin'],
                                       ['batch', 'frame', '1', ('mic', 'bin')])

  # Run the TDCN++ network.
  net_config = network_config.improved_tdcn()
  core_activations = network.improved_tdcn(mixture_coeffs_input, net_config)
  shaper.register_axes(core_activations, ['batch', 'frame', 'out_depth'])

  # Apply a dense layer to increase output dimension.
  bins = signal_util.static_or_dynamic_dim_size(mixture_coeffs, -1)
  dense_config = network.update_config_from_kwargs(
      network_config.DenseLayer(),
      num_outputs=num_mics * bins * num_sources,
      activation='linear')
  activations = network.dense_layer(core_activations, dense_config)
  shaper.register_axes(
      activations, ['batch', 'frame', ('mic', 'source', 'bin')])

  # Create a mask from the output activations.
  activations = shaper.change(
      activations, ['batch', 'frame', ('mic', 'source', 'bin')],
      ['batch', 'source', 'mic', 'frame', 'bin'])
  mask = network.get_activation_fn('sigmoid')(activations)
  mask = tf.identity(mask, name='mask')

  # Apply the mask to the mixture coefficients.
  mask = tf.cast(mask, dtype=mixture_coeffs.dtype)
  mask_input = mixture_coeffs[:, tf.newaxis]
  shaper.register_axes(mask_input, ['batch', '1', 'mic', 'frame', 'bin'])
  separated_coeffs = mask * mask_input
  shaper.register_axes(
      separated_coeffs, ['batch', 'source', 'mic', 'frame', 'bin'])

  # Reconstruct the separated waveforms from the masked coefficients.
  mixture_length = signal_util.static_or_dynamic_dim_size(mixture_waveforms, -1)
  separated_waveforms = inverse_transform(separated_coeffs)
  separated_waveforms = separated_waveforms[..., :mixture_length]

  # Apply mixture consistency, if specified.
  if hparams.mix_weights_type:
    if hparams.mix_weights_type == 'pred_source':
      # Mean-pool across time.
      mix_weights = tf.reduce_mean(core_activations, axis=1)
      # Dense layer to num_sources.
      dense_config = network.update_config_from_kwargs(
          network_config.DenseLayer(),
          num_outputs=num_sources,
          activation='linear')
      with tf.variable_scope('mix_weights'):
        mix_weights = network.dense_layer(mix_weights, dense_config)
      # Softmax across sources.
      mix_weights = tf.nn.softmax(
          mix_weights, axis=-1)[:, :, tf.newaxis, tf.newaxis]
      shaper.register_axes(
          mix_weights, ['batch', 'source', '1', '1'])
    elif (hparams.mix_weights_type == 'uniform'
          or hparams.mix_weights_type == 'magsq'):
      mix_weights = None
    else:
      raise ValueError('Unknown mix_weights_type of "{}".'.format(
          hparams.mix_weights_type))
    separated_waveforms = consistency.enforce_mixture_consistency_time_domain(
        mixture_waveforms, separated_waveforms,
        mix_weights=mix_weights,
        mix_weights_type=hparams.mix_weights_type)

  # If multi-mic, just use the reference microphone.
  separated_waveforms = separated_waveforms[:, :, 0]

  separated_waveforms = tf.identity(separated_waveforms,
                                    name='seped_waveforms')
  return separated_waveforms
def compute_summary_tensorboard(separated_waveforms,source_waveforms,params):
  mixture_waveforms = tf.reduce_sum(source_waveforms,axis=-1)
  outnum = params["outnum"]
  tiled_mixture_waveforms=tf.tile(tf.expand_dims(mixture_waveforms,axis=-1),[1,outnum,1])
  
  return
def model_fn(params):
  print("params")
  print(params)
  #tf.enable_eager_execution()#tf2にはない

  
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
  classnum = params["outnum"]
  print("model start")

  train_params = copy.deepcopy(params)
  train_params["split"] = "train"
  batch_size = params["train_batch_size"]
  train_params['batch_size'] = batch_size
  train_params['example_num'] = params["train_examples"]


  dataset = data_io.input_fn(train_params)
  print(dataset)
  datanum=dataset.cardinality().numpy()
  dataset_waveform = np.empty((0,batch_size, sample_num))
  dataset_label = np.empty((0,batch_size, classnum),dtype=float)
  file_writer = tf.summary.create_file_writer(params["model_dir"] + "/metrics")
  file_writer.set_as_default()
  """for d in dataset:
    #print(d["source_image"].shape)
    dataset_waveform=np.append(dataset_waveform,[d["source_image"]],axis=0)
    dataset_label=np.append(dataset_label,[d["label"]],axis=0)"""
  #データセットのテスト
  #datatest(datatest)
  print("size1")
  print("source shape")
  print("size2")
  print(sys.getsizeof(dataset))
  print("size3")

  inputs = tf.keras.Input(shape=(sample_num),name="input_mixture")
  #test1,test2,test22,test23,test3,test4,probabilities =  get_probs(inputs,hparams)
  config = mynetwork.MyConfig(outnum=classnum)
  Separated_wavforms =  mynetwork.Separate(inputs,params,config)
  #Separated_wavforms =  mynetwork.Separate_nomyactivate(inputs,params,config)
  #e_inputs = tf.reshape(inputs,[params["train_batch_size"],1,160000])
  #Separated_wavforms =  model_sub.separate_waveforms(e_inputs,hparams)
  print("sep shape : {}".format(Separated_wavforms.shape))


  loss = Separated_loss(params)
  #loss = model_sub.loss_fn(hparams)

  
  """learning_rate = tf.train.exponential_decay(
      hparams.lr,
      tf.train.get_or_create_global_step(),
      decay_steps=hparams.lr_decay_steps,
      decay_rate=hparams.lr_decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)"""
  #test dataset
  """time=0
  for image,source in dataset.take(10):
    time += 1
    tmpdata=source
    #print(tmpdata)
    #tmpout = sep_model.predict(image)
    #tmpout = tf.reshape(tmpout,[params["train_batch_size"],params["outnum"],160000])
    for i in range(batch_size):
      for j in range(params["outnum"]):
        if time <= 10:
          path = params["model_dir"] + "/fromdata_tmpout{}_{}.wav".format(i,j)
          writewav(tmpdata[i][j],path)"""
  #metrics={"variance":tf.keras.metrics.Mean(variance),"labelloss" :tf.keras.metrics.Mean(losses2) }
  """tf.summary.scalar(
      name, data, step=None, description=None
  )"""
  def dummyloss(sources,separated_waveforms):
    return 0
  sep_model = tf.keras.Model(inputs, {"out":Separated_wavforms})
  sep_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss={"out":loss})
              #metrics=metrics)

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=params["model_dir"],
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
  print("model summery:")
  print(sep_model.summary())
  ret = calckerassize.get_model_memory_usage(params["train_batch_size"],sep_model)
  #tensorboardのcallback
  tb_cb = tf.keras.callbacks.TensorBoard(log_dir=params["model_dir"], histogram_freq=1)
  #GB単位
  print(ret)


  eval_params = copy.deepcopy(params)
  eval_params["split"] = "eval"
  eval_params["batch_size"] = params["eval_batch_size"]
  eval_params["example_num"] = params["eval_examples"]

  evaldataset = data_io.input_fn(eval_params)
  print(evaldataset)
  evaldatanum=evaldataset.cardinality().numpy()
  evaldataset_waveform = np.empty((0,batch_size, sample_num))
  evaldataset_label = np.empty((0,batch_size, params["outnum"]),dtype=float)
  """for d in evaldataset:
    #print(d["source_image"].shape)
    evaldataset_waveform=np.append(evaldataset_waveform,[d["source_image"]],axis=0)
    evaldataset_label=np.append(evaldataset_label,[d["label"]],axis=0)"""
  
  print("fit start")
  """for l in full_model.layers:
    print(l)
    print(l.get_weights())"""

  print("dataset")
  #okっぽい?
  """test_input=tf.random.uniform(shape=[params["train_batch_size"],160000])
  test_output=sep_model.predict(test_input)
  reduce_test_output = tf.reduce_sum(test_output,axis=1)
  #test_input2=tf.random_uniform(shape=[7,4,160000])
  print(test_input)
  print(reduce_test_output)
  print(test_output.shape)
  #print("testloss : {}".format(loss(test_input,reduce_test_output)))
  print("testassert : {}".format(tf.reduce_mean((test_input-reduce_test_output)**2)))"""
  #return
  print("dataset")
  print(dataset)
  print("is GPU : {}".format(tf.config.experimental.list_physical_devices('GPU')))
  if True:
    #sep_model.load_weights(params["model_dir"]+"/weights.tf")
    sep_model.fit(dataset,
                 #epochs = 1,
                 verbose = 2,
                 epochs = 30,
                 batch_size=params["train_batch_size"],
                 callbacks=[model_checkpoint_callback,tb_cb],
                 validation_data = evaldataset)
    sep_model.save_weights(params["model_dir"]+"/weights.h5")
    sep_model.save_weights(params["model_dir"]+"/weights.tf")
    sep_model.save(params['model_dir']+"/savedmodel")
  else:
    sep_model.load_weights(params["model_dir"]+"/weights")
  allmetrics = {}
  allmetrics2 = {}
  metricskeys = ['sisnr_separated',
                'sisnr_mixture',
                'sisnr_improvement',
                'power_separated',
                'power_sources']
  for key in metricskeys:
    allmetrics[key] = []
    allmetrics2[key] = []
  time = 0
  for image,source in evaldataset.take(100):
    time += 1
    tmpdata=source
    #print(tmpdata)
    tmpout = sep_model.predict(image)
    tmpout = tf.reshape(tmpout,[params["train_batch_size"],params["outnum"],160000])
    print(image.shape)
    print(tmpout.shape)
    for i in range(batch_size):
      raw_metrics=raw_compute_metrics(source[i], tmpout[i], image[i])
      print(raw_metrics)
      for key in metricskeys:
        allmetrics2[key].append(tf.reduce_mean(raw_metrics[key]))
      for j in range(params["outnum"]):
        if time <= 10:
          path = params["model_dir"] + "/tmpout{}_{}.wav".format(i,j)
          writewav(tmpout[i][j],path)
    #tmpout = tf.squeeze(tmpout,axis=-1)
    tmpdata = tf.squeeze(tmpdata)
    print(tmpdata)
    print(image.shape)
    image = tf.expand_dims(tf.squeeze(image),axis=-2)
    #image = tf.tile(image,[1,params["outnum"],1])
    mix = tf.cast(tf.expand_dims(tf.reduce_sum(tmpdata,axis=-2),axis=-2),dtype = tf.float32)
    metrics = compute_metrics( tmpdata,tmpout,image,config,params)
    for key in metricskeys:
      allmetrics[key].append(tf.reduce_mean(metrics[key]))
  print("metrics")
  path_w = params["model_dir"]+"/evaldata.txt"


  with open(path_w, mode='w') as f:
      for key in metricskeys:
        line = "{} : {}\n".format(key,tf.reduce_mean(allmetrics[key]))
        print(line)
        f.write(line)
      print("raw metrics")
      f.write("raw metrics")
      
      for key in metricskeys:
        if key in allmetrics2.keys():
          line = "{} : {}\n".format(key,tf.reduce_mean(allmetrics2[key]))
          print(line)
          f.write(line)

  print("after params")
  """for l in full_model.layers:
    print(l)
    print(l.get_weights())"""
  print(params['model_dir'])
  sep_model.save_weights(params["model_dir"]+"/weights.h5")
  sep_model.save_weights(params["model_dir"]+"/weights.tf")
  sep_model.save(params['model_dir']+"/savedmodel")
  print("all end")
  return
  
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