
"""Network layers."""

import copy

import tensorflow as tf
from . import signal_transformer
from . import network
from . import network_config


class MyConfig():


  def __init__(self,outnum):
    self.outnum = outnum
    self.Bottleneck=256
    self.X = 32#32
    self.C = 512#conv_block size
    self.Frame = 1250
    self.P=3
    self.Bin = 257
    tmp = 0
    self.dilations_array=[]
    #[1, 2, 4, 8, 16, 32, 64, 128]的な
    while True:
      forappend = 2**tmp
      if forappend >= self.Bottleneck:
        break
      self.dilations_array.append(2**tmp)
      tmp += 1
  
  def get_dilation_rate(self,b=None):
    index = b%(len(self.dilations_array))
    return self.dilations_array[index]

class SchalarMuliplier(tf.keras.layers.Layer):

  def __init__(self, init=1.0):
    super(SchalarMuliplier,self).__init__()
    self.initx = init
    self.x = self.add_weight(shape=(1,),
                               initializer=tf.keras.initializers.Constant(self.initx),
                               trainable=True,
                               name = "My_Scala_Layer")

  def get_config(self):
        return {"initx": self.initx,
                "x" : self.x}
  """def get_config(self): 
    config = super().get_config().copy()
    config.update({
        'initx': self.initx,
        'x': self.x
    })
    return config"""


  def call(self, inputs):  # Defines the computation from inputs to outputs
      return inputs*self.x

def Conv_Block(inputsignal,b,config : MyConfig):
  # signal shape is [batch,frame,bottleneck]
  # b is the number of the block

  signal = tf.keras.layers.Dense(config.C)(inputsignal)
  signal =tf.keras.layers.PReLU()(signal)
  signal = tf.keras.layers.LayerNormalization()(signal)
  signal = tf.keras.layers.Conv1D(
        filters = config.C,#入出力を同じ形に
        kernel_size=3,
        padding='SAME',
        dilation_rate=config.get_dilation_rate(b),
        )(signal)
  signal =tf.keras.layers.PReLU()(signal)
  signal = tf.keras.layers.LayerNormalization()(signal)
  #これが1x1 Conv
  signal = tf.keras.layers.Dense(config.Bottleneck)(signal)
  signal = SchalarMuliplier(0.9**b)(signal)
  #scalarvar = tf.Variable(0.9**b)
  #signal = scalarvar*signal
  #residual network
  final_signal= signal + inputsignal
  return final_signal


#元のコード使うほう
def Separate_nomyactivate(wave_inputsignal,params, config : MyConfig):
  hparams = params["hparams"]
  #signal shape is [batch,samplenum]
  print("inputsignal shape {}".format(wave_inputsignal))
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)
  inputsignal = transformer.forward(wave_inputsignal)
  print("signal shape {}".format(inputsignal))
  signal = tf.abs(inputsignal)#スペクトルを計算する
  signal = tf.keras.layers.LayerNormalization(
    #axis = [-2,-1]
    axis = [-2]
  )(signal)#normalization

  #row
  im_config = network_config.improved_tdcn()
  signal = tf.expand_dims(signal,axis = 2)
  signal = network.improved_tdcn(signal,im_config)
  signal = tf.keras.layers.Dense(config.Bin * config.outnum)(signal)
  
  #signal = tf.reshape(signal,[params["train_batch_size"],config.outnum, config.Frame, config.Bin],name = "beforemask reshape")
  signal = tf.reshape(signal,[params["train_batch_size"], config.Frame,config.outnum, config.Bin],name = "beforemask reshape")
  #[batch,frame,outnum,bin]を[batch,outnum,frame,bin]に軸を入れ替え
  signal=tf.transpose(signal, perm=[0,2,1,3]) 
  mask = tf.keras.activations.sigmoid(signal)

  ex_inputsignal =  tf.expand_dims(inputsignal,axis=1)
  #ex_inputsignal = tf.tile(ex_inputsignal,[1,config.outnum,1,1])
  ex_inputsignal = tf.tile(ex_inputsignal,[1,config.outnum,1,1])
  signal = tf.cast(mask,dtype = tf.complex64)* tf.cast(ex_inputsignal,dtype = tf.complex64)

  inverse_transform = transformer.inverse
  print("testinverse")
  print("bedore inverse")
  print(signal.shape)
  signal=inverse_transform(signal)[...,:tf.shape(wave_inputsignal)[-1]]#音の最後の部分は0埋めされるようにSTFTシテイルカラ
  print(signal.shape)
  #for consissstency
  sumed = (wave_inputsignal - tf.reduce_sum(signal,axis=1))/config.outnum#これがM
  print(sumed.shape)
  sumed = tf.expand_dims(sumed,axis = 1)
  print(sumed.shape)
  sumed = tf.tile(sumed,[1,config.outnum,1])
  print(sumed.shape)
  finalsignal = signal+sumed
  return finalsignal

def Separate(wave_inputsignal,params, config : MyConfig):
  hparams = params["hparams"]
  #signal shape is [batch,samplenum]
  print("inputsignal shape {}".format(wave_inputsignal))
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)
  inputsignal = transformer.forward(wave_inputsignal)
  print("signal shape {}".format(inputsignal))
  signal = tf.abs(inputsignal)#スペクトルを計算する
  signal = tf.keras.layers.LayerNormalization(
    #axis = [-2,-1]
    axis = [-2]
  )(signal)#normalization

  signal = tf.keras.layers.Dense(config.Bottleneck)(signal)
  #signal = SchalarMuliplier()(signal)

  signals = []#過去のシグナルの記録
  input_res_indexes = [0, 0, 0, 8, 8, 16]
  output_res_indexes = [8, 16, 24, 16, 24, 24]
  print("signal shape {}".format(signal))
  for b in range(config.X):#ここはちゃんとパスとして通ってる
    signals.append(signal)
    #resdual network
    for outindex in range(len(output_res_indexes)):
      if b == output_res_indexes[outindex]:
        in_index = input_res_indexes[outindex]
        ressignal = signals[in_index]
        ressignal = tf.keras.layers.Dense(config.Bottleneck)(ressignal)
        #ressignal = SchalarMuliplier()(ressignal)
        signal += ressignal
    signal = Conv_Block(signal,b,config)
  print("after conv shape {}".format(signal.shape))
  #1x1 conv
  signal = tf.keras.layers.Dense(config.Bin * config.outnum )(signal)
  print("after Dense {}".format(signal.shape))
  #signal = SchalarMuliplier()(signal)
  print("before reshape {}".format(signal.shape))
  
  #signal = tf.reshape(signal,[params["train_batch_size"],config.outnum, config.Frame, config.Bin],name = "beforemask reshape")
  signal = tf.reshape(signal,[params["train_batch_size"], config.Frame,config.outnum, config.Bin],name = "beforemask reshape")
  #[batch,frame,outnum,bin]を[batch,outnum,frame,bin]に軸を入れ替え
  signal=tf.transpose(signal, perm=[0,2,1,3]) 
  mask = tf.keras.activations.sigmoid(signal)

  ex_inputsignal =  tf.expand_dims(inputsignal,axis=1)
  #ex_inputsignal = tf.tile(ex_inputsignal,[1,config.outnum,1,1])
  ex_inputsignal = tf.tile(ex_inputsignal,[1,config.outnum,1,1])
  signal = tf.cast(mask,dtype = tf.complex64)* tf.cast(ex_inputsignal,dtype = tf.complex64)

  inverse_transform = transformer.inverse
  print("testinverse")
  print("bedore inverse")
  print(signal.shape)
  signal=inverse_transform(signal)[...,:tf.shape(wave_inputsignal)[-1]]#音の最後の部分は0埋めされるようにSTFTシテイルカラ
  print(signal.shape)
  #for consissstency
  sumed = (wave_inputsignal - tf.reduce_sum(signal,axis=1))/config.outnum#これがM
  print(sumed.shape)
  sumed = tf.expand_dims(sumed,axis = 1)
  print(sumed.shape)
  sumed = tf.tile(sumed,[1,config.outnum,1])
  print(sumed.shape)
  finalsignal = signal+sumed
  sumed_final = tf.reduce_sum(finalsignal,axis=1)
  #assertop=tf.Assert(tf.less_equal(tf.reduce_mean((sumed_final-wave_inputsignal)**2),0.0001), [sumed_final]) 
  #assert tf.reduce_mean((sumed_final-wave_inputsignal)**2) < 0.0001
  return finalsignal
