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
#import cupy as cp
import attr
import typing
import copy
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow
import sys
from . import consistency
from . import groupwise
from . import network
from . import network_config
from . import signal_transformer
from . import signal_util
from . import summaries
from . import summary_util
from . import shaper
from . import soundclass
from . import data_io
from . import calckerassize
from . import callbacks
import shutil
import soundfile as sf
import os
Shaper = shaper.Shaper

# Define loss functions.
mse_loss = lambda source, separated: tf.nn.l2_loss(source - separated)


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
                             default=['source'])
  # A list of strings same length as signal_names specifying signal type, used
  # for groupwise permutation-invariance.
  signal_types = attr.attrib(type=typing.List[typing.Text],
                             default=['source'])
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
  # tdcn size
  discparam_size =attr.attrib(type=typing,default="large")
  #class num
  classnum =attr.attrib(type=int,default=-1)


def get_model_hparams():
  return HParams()

use_Dense=True
def get_probability(wavform,hparams):
    """Computes and returns separated waveforms.

    Args:
      wavform: Waveform of audio to separate, shape (batch, mic, time).
      hparams: Model hyperparameters.
    Returns:
      Separated audio tensor, shape (batch, source, time), same type as mixture.
    """
    batch_size = signal_util.static_or_dynamic_dim_size(wavform, 0)
    wavform = tf.reshape(wavform,[batch_size,1,160000])
    num_sources = len(hparams.signal_names)
    print(hparams)
    print("get_probability")
    print(wavform.shape)
    #num_mics = signal_util.static_or_dynamic_dim_size(wavform, 1)
    #print("num_mics : {}".format(num_mics))
    shaper = Shaper({'source': num_sources, '1': 1})

    # Compute encoder coefficients.
    transformer = signal_transformer.SignalTransformer(
        sample_rate=hparams.sr,
        window_time_seconds=hparams.ws,
        hop_time_seconds=hparams.hs)
    mixture_coeffs = transformer.forward(wavform)
    inverse_transform = transformer.inverse
    mixture_coeffs_input = tf.abs(mixture_coeffs)
    mixshape=mixture_coeffs.shape
    sum0=tf.reduce_sum(mixture_coeffs_input)
    print(mixture_coeffs_input.shape)#(frame,bins)
    #mixture_coeffs_input = tf.cond(sum0 < 0.00001,lambda: tf.random.uniform(mixshape,0,0.0001), lambda:mixture_coeffs_input)#無音の場合の対処
    mixture_coeffs_input = network.LayerNormalizationScalarParams(
        #axis=[-3, -2, -1],
        axis=[-2, -1],
        name='layer_norm_on_mag').apply(mixture_coeffs_input)
    print(tf.shape(mixture_coeffs_input))
    print(mixture_coeffs_input)
    print(mixture_coeffs_input.shape)
    #shaper.register_axes(mixture_coeffs, ['batch','1', 'frame' ,'bin'])
    #frame=mixture_coeffs.shape[1]
    frame=1250
    #batch_size=mixture_coeffs.shape[0]
    #bins = signal_util.static_or_dynamic_dim_size(mixture_coeffs, -1)
    bins = 257
    print("shapetest")
    print(mixture_coeffs_input[ :, :,tf.newaxis].shape)
    shaper.register_axes(mixture_coeffs, ['batch', 'mic', 'frame', 'bin'])
    mixture_coeffs_input = shaper.change(mixture_coeffs_input[:, :, tf.newaxis],
                                        ['batch','mic', 1,  'frame', 'bin'],
                                        ['batch', 'frame',1,  ('mic', 'bin')])

    #mixture_coeffs_input = mixture_coeffs_input[ :, :,tf.newaxis]
    #mixture_coeffs_input = tf.reshape(mixture_coeffs_input,[1,frame/10,1,bins])
    #return
    print(mixture_coeffs_input.shape)

    # Run the TDCN++ network.
    print("hparams.discparam_size {}".format( hparams.discparam_size ))
    if hparams.discparam_size =="small":
      net_config = network_config.small_improved_tdcn()#reduced parameters
    elif hparams.discparam_size == "middle":
      net_config = network_config.improved_tdcn_28()
    elif hparams.discparam_size == "large":
      net_config = network_config.improved_tdcn()
    else:
      net_config = network_config.improved_tdcn()

    print("config")
    print(net_config)
    core_activations = network.improved_tdcn(mixture_coeffs_input, net_config)
    shaper.register_axes(core_activations, ['batch', 'frame', 'out_depth'])
    print("get_probability_core_activations")
    print(tf.shape(core_activations))
    #use_Dense=False
    if use_Dense==False:
      norm=tf.keras.layers.BatchNormalization(axis=-1,name="norm")
      core_activations=norm(core_activations)

      # Apply a dense layer to decrease output dimension.
      bins = signal_util.static_or_dynamic_dim_size(mixture_coeffs, -1)
      print("bins")#binsってもしかして係数?
      print(bins)
      dense_config = network.update_config_from_kwargs(
          network_config.DenseLayer(),
          num_outputs=1,
          activation='linear')
      activations = network.dense_layer(core_activations, dense_config)
      print(activations.shape)
      activations = tf.squeeze(activations,axis=[-1])
      print("act")
      print(activations.shape)
      #print(batch_size)
      
      print(frame)
      activations=tf.reshape(activations, [batch_size,frame])
      norm2=tf.keras.layers.BatchNormalization(axis=-1,name="norm2")
      activations=norm2(activations)
      print("get_probability_activations")
      print(tf.shape(activations))
      print(activations.shape)
      #activations = tf.reshape(activations,[1,frame])
      dense_config2 = network.update_config_from_kwargs(
          network_config.DenseLayer(),
          num_outputs=hparams.classnum,
          activation='linear')
      activations = network.dense_layer(activations, dense_config2)
      #activations=tf.reshape(activations,[1],name="probability")
      #ret = tf.squeeze(activations,[-1])
      ret = activations
      #activations=tf.math.sigmoid(activations)
    else:
      core_activations = tf.reduce_mean(core_activations)
      ret=tf.math.sigmoid(core_activations)
      
    '''
    shaper.register_axes(
        activations, ['batch', 'frame', ('mic', 'source', 'bin')])
        '''
    print("get_probability_activations2")
    #print(tf.shape(activations))
    #print(tf.shape(activations[0]))
    
    ret = tf.nn.softmax(ret,axis=0)

    return ret
def wraped_get_probability(waveform,hparams):
  

  def map_wrap(input):#for batch axis
    print(input.shape)
    size=signal_util.static_or_dynamic_dim_size(input, 0)
    #ret = tf.map_fn( get_probability, input)
    ret = get_probability(input,hparams)#batch軸の正規化のため
    if use_Dense==True:
      ret = tf.reshape(ret,[size,hparams.classnum])
      pass
    return ret

  return map_wrap(waveform)


#データセットioテスト
def datatest(dataset,params):
  out_path = params["model_dir"]+"/data_io_test"
  if os.path.exists(out_path):
    shutil.rmtree(out_path)
  os.mkdir(out_path)
  labelfile="{}/labels.txt".format(out_path)
  f = open(labelfile, 'w')
  i=0
  _format = "WAV"
  for waves, labels in dataset.take(50):
    for j in range(len(waves)):
      wave = waves[j]
      label = labels[j]
      wave = np.reshape(wave,[160000,1])
      wavefilename = "{}/{}_{}_labelis_{}.wav".format(out_path,i,j,label)
      sf.write(wavefilename, wave, 16000, format=_format)
      writelabelname ="{}_var : {}\n{}\n".format(j,np.var(np.squeeze(wave)),label)
      f.write(writelabelname)
    i += 1
  f.close()
  return

def BatchNormalization(x,name="Not set name"):
  ret = tf.keras.layers.BatchNormalization(
      axis=[1],#軸を減らすとパラメータ薄も減るっぽい
      momentum=0.99,
      epsilon=0.001,
      name=name
  )(x)
  return ret
def GRU(x,hparams,mode=tf.estimator.ModeKeys.TRAIN,name="Not set name"):
  #print("eager : {}".format(tf.executing_eagerly()))#trueだった
  #print("mode : {}".format(mode))
  """x=tf.ones(#00全部0にしたら普通に出力された。
    shape=tf.shape(x),
    dtype=tf.dtypes.float32
  )"""
  ret, final_state=tf.keras.layers.GRU(
      #units=128,#1024だった、128はoverconfident
      units=hparams.unit,
      activation='tanh',
      #recurrent_activation = "ReLU",#nanのこれが原因?
      return_sequences=True,
      use_bias=True,
      time_major=False,
      return_state=True,
      dropout = 0.2,
      name=name
    )(inputs=x)
  return ret
def LReLU(x):
  return tf.keras.layers.LeakyReLU()(x)
def get_probs(waveform,hparams,mode=tf.estimator.ModeKeys.TRAIN):
  # wavform shape is [batch,samplenum]
  #sourceはbatch度一緒に並べることにする。[batch,sourcenum,samplenum] -> [batch*sourcenum,samplenum]
  #batch_size = signal_util.static_or_dynamic_dim_size(waveform, 0)
  
  samplenum = waveform.shape[-1]
  # Compute encoder coefficients.
  #print("input waveform shape is {}".format(waveform.shape))
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)
  mainsignal = transformer.forward(waveform)
  mainsignal = tf.abs(mainsignal)#スペクトルを計算する
  #print("transformed_waveform shape is {}".format(mainsignal.shape))
  #transformed_waveform is [batch,  frame, cols]#colsは係数
  mainsignal = tf.expand_dims(mainsignal, -1)#チャンネルがある場合の実装に合わせて軸を増やしておく
  #mainsignal is [batch,frame, cols, channel = 1]
  #print("mainsignal shape is {}".format(mainsignal.shape))
  #BN
  mainsignal = BatchNormalization(mainsignal,name="BatchNormalization1")
  #dropout
  layer = tf.keras.layers.Dropout(0.3,seed=114514)
  mainsignal = layer(mainsignal, training=True)
  #conv2D
  #チャンネル数が常に1と扱われていることに注意
  mainsignal=tf.keras.layers.Conv2D(
      filters = 64,
      kernel_size = (5,5),
      strides=(2, 2),
      data_format='channels_last',#最後の軸がチャンネルという意味
      activation=None
  )(mainsignal)
  #print("mainsignal after conv shape is {}".format(mainsignal.shape))
  test1 = mainsignal
  #BN
  mainsignal = BatchNormalization(mainsignal,name="BatchNormalization2")
  test2 = mainsignal
  #Average pooling
  mainsignal=tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    data_format='channels_last'
  )(mainsignal)
  #BN
  mainsignal = BatchNormalization(mainsignal,name="BatchNormalization3")
  #reshape to [batch,frame,col*channel]
  frame = tf.shape(mainsignal)[-3]
  col = tf.shape(mainsignal)[-2]
  channel = tf.shape(mainsignal)[-1]
  mainsignal = tf.reshape(mainsignal,[-1,frame, col*channel])
  #print("before GRU1 shape : {}".format(mainsignal.shape))
  #GRU
  test22=mainsignal
  mainsignal=GRU(mainsignal,hparams,mode=mode,name="GRU1")
  test23 = mainsignal
  mainsignal = LReLU(mainsignal)

  #print("after GRU1  shape : {}".format(mainsignal.shape))
  #Layer normalization
  mainsignal=tf.keras.layers.LayerNormalization(
    axis=-1,#これ具体的に何を指しているのかが書かれていない
    epsilon=0.001)(mainsignal)
  test3 = mainsignal
  #GRU
  mainsignal=GRU(mainsignal,hparams,mode=mode,name="GRU2")
  mainsignal = LReLU(mainsignal)
  #Layer normalization
  mainsignal=tf.keras.layers.LayerNormalization(
    axis=-1,#これ具体的に何を指しているのかが書かれていない
    epsilon=0.001)(mainsignal)
  #GRU
  mainsignal=GRU(mainsignal,hparams,mode=mode,name="GRU3")
  mainsignal = LReLU(mainsignal)
  #Layer normalization
  mainsignal=tf.keras.layers.LayerNormalization(
    axis=-1,#これ具体的に何を指しているのかが書かれていない
    epsilon=0.001)(mainsignal)
  #mainsignal is [batch, ..., ]
  #print("mainsignal shape is {}".format(mainsignal.shape))
  #Full connected
  mainsignal=tf.keras.layers.Dense(
    units=1
  )(mainsignal)
  #BN
  mainsignal = BatchNormalization(mainsignal,name="BatchNormalization4")
  test4 = mainsignal
  #次元削除
  mainsignal = tf.squeeze(mainsignal,axis=-1)
  #dense layer用にこの次元がいる
  mainsignal = tf.expand_dims(mainsignal,axis=1)
  #Full connected
  mainsignal=tf.keras.layers.Dense(
    units=hparams.classnum
  )(mainsignal)
  mainsignal = tf.squeeze(mainsignal,axis=[-2])
  #signal shape is [batch, classnum]
  #最後に確率の形にする
  mainsignal = tf.keras.activations.softmax(mainsignal, axis =-1)
  #print("final main signal shape {}".format(mainsignal.shape))
  #return test1,test2,test22,test23,test3,test4,mainsignal
  return mainsignal
def model_fn(params):
  print("params")
  print(params)

  
  """Constructs a spectrogram_lstm model with summaries.

  Args:
    features: Dictionary {name: Tensor} of model inputs.
    labels: Any training-only inputs.
    mode: Build mode, one of tf.estimator.ModeKeys.
    params: Dictionary of Model hyperparameters.

  Returns:
    EstimatorSpec describing the model.
  """
  hparams = params['hparams']
  sample_num=160000
  print("model start")

  train_params = copy.deepcopy(params)
  train_params["split"] = "train"
  batch_size = params["train_batch_size"]
  train_params['batch_size'] = batch_size
  train_params['example_num'] = params["train_examples"]


  dataset = data_io.input_fn_disc(train_params)
  print(dataset)
  datanum=dataset.cardinality().numpy()
  dataset_waveform = np.empty((0,batch_size, sample_num))
  dataset_label = np.empty((0,batch_size, hparams.classnum),dtype=float)
  """for d in dataset:
    #print(d["source_image"].shape)
    dataset_waveform=np.append(dataset_waveform,[d["source_image"]],axis=0)
    dataset_label=np.append(dataset_label,[d["label"]],axis=0)"""
  #データセットのテスト
  datatest(dataset,params)
  print("size1")
  print("source shape")
  print("size2")
  print(sys.getsizeof(dataset))
  print("size3")

  #inputs = tf.keras.Input(shape=(sample_num),batch_size=batch_size)
  inputs = tf.keras.Input(shape=(sample_num),batch_size=train_params['batch_size'])
  print("input shape : {}".format(inputs))
  #test1,test2,test22,test23,test3,test4,probabilities =  get_probs(inputs,hparams)
  probabilities =  get_probs(inputs,hparams)
  print("probabilities shape {}".format(probabilities.shape))
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                            name="Binarycrossentropy",
                                            axis=-1)

  variance = params["variance_omega"]*tf.math.reduce_variance(probabilities,axis=-1)
  
  #full_model = tf.keras.Model(inputs, (test1,test2,test22,test23,test3,test4,probabilities))
  full_model = tf.keras.Model(inputs, probabilities)
  full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=loss)
              #metrics=[loss])

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=params["model_dir"],
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
  print("model summery:")
  print(full_model.summary())
  ret = calckerassize.get_model_memory_usage(params["train_batch_size"],full_model)
  #tensorboardのcallback
  tb_cb = tf.keras.callbacks.TensorBoard(log_dir=params["model_dir"], histogram_freq=1)
  #GB単位
  print(ret)


  eval_params = copy.deepcopy(params)
  eval_params["split"] = "eval"
  eval_params["batch_size"] = params["eval_batch_size"]
  eval_params["example_num"] = params["eval_examples"]

  evaldataset = data_io.input_fn_disc(eval_params)
  
  #datatest(evaldataset,params)#eval,trainのデータも大丈夫っぽい
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
  for images, labels in dataset.take(1):  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    print(numpy_images)
    print(numpy_labels)
    testoutput = tf.random.uniform(shape=tf.shape(labels)).numpy()
    print("testloss : {}".format(loss(labels,testoutput)))
    #test1,test2,test22,test23,test3,test4,testpredict = full_model.predict(numpy_images)
    testpredict = full_model.predict(numpy_images)
    """print("test1 : {}".format(test1))
    print("test2 : {}".format(test2))
    print("test22 : {}".format(test22))
    print("test23 : {}".format(test23))
    print("test3 : {}".format(test3))
    print("test4 : {}".format(test4))
    print("mean test")
    print("test1 : {}".format(tf.math.reduce_mean(test1)))
    print("test2 : {}".format(tf.math.reduce_mean(test2)))
    print("test22 : {}".format(tf.math.reduce_mean(test22)))
    print("test23 : {}".format(tf.math.reduce_mean(test23)))
    print("test3 : {}".format(tf.math.reduce_mean(test3)))
    print("test4 : {}".format(tf.math.reduce_mean(test4)))"""
    print("test output : {}".format(testpredict))
  
  if False:
    print(params["model_dir"])
    full_model.load_weights(params["model_dir"]+"/weights")
    count = 0
    allcount=0
    tmp2 = [0.92,0.06,0.02]
    tmp1 = [0.02,0.02,0.96]
    tmp = [0,0,1]
    print(loss(tmp,tmp1))
    print(loss(tmp,tmp2))
    for images, labels in evaldataset.take(1000):
      predict = full_model.predict(images)
      print("label :")
      print(labels)
      print("probs : ")
      print(predict)
      allcount+=1
      for batchindex in range(len(images)):
        labelindex = np.argmax(labels[batchindex])
        probindex = np.argmax(predict[batchindex])
        if labelindex == probindex:
          count += 1
    print("acc num : {}".format(count))
    print("acc : {}".format(count/allcount))
    return
  else:
    #full_model.load_weights(params["model_dir"]+"/weights")
    full_model.fit(dataset,
                 epochs = params["train_epoch"],
                 batch_size=params["train_batch_size"],
                 verbose=2,
                 callbacks=[model_checkpoint_callback,tb_cb],
                 validation_data = evaldataset)
  print("after params")
  """for l in full_model.layers:
    print(l)
    print(l.get_weights())"""
  count = 0
  allcount=0
  for images, labels in evaldataset.take(1000):
    predict = full_model.predict(images)
    print("label :")
    print(labels)
    print("probs : ")
    print(predict)
    for batchindex in range(len(images)):
      allcount+=1
      labelindex = np.argmax(labels[batchindex])
      probindex = np.argmax(predict[batchindex])
      if labelindex == probindex:
        count += 1
  print("acc num : {}".format(count))
  print("acc : {}".format(count/allcount))
  full_model.save(params["model_dir"]+"/savedmodel")
  full_model.save_weights(params["model_dir"]+"/weights.tf")
  full_model.save_weights(params["model_dir"]+"/weights.h5")
  print("all end")
  