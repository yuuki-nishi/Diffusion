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

import argparse
import glob
import tensorflow.compat.v1 as tf
from train import model
from train import signal_transformer
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
def readwav(path):
  #print("path : {}".format(path))
  audio_bytes = tf.read_file(path)
  waveform, sample_rate	 = tf.audio.decode_wav(audio_bytes, desired_channels=1)#[audio,samplerate]
  #print(waveform.shape)
  waveform = tf.squeeze(waveform,axis=[-1])
  #specs = transformer.forward(waveform)
  waveform = scaling(waveform,1.0)
  return waveform , sample_rate#[samplerate]

def calcfromstfttospecor(x):#各係数の実数虚数を合わせる
  time = x.shape[0]
  num = x.shape[1]
  ret=[]
  for t in range(time):
    tmpret=[]
    for i in range(1,num,2):
      tmp = np.sqrt(x[t][2*i]**2 + x[t][x*i+1] )
      tmpret.append(tmp)
    ret.append(tmpret)
  return ret


def scaling(x,db):#[samplerate]
  #音の配列xをちょうどいい音量に変更する
  #print(x.shape)
  #num_samples = x.shape[-1]
  #x=tf.squeeze(x,axis=[1])#(None,1,160000)だから
  scales = tf.reduce_mean(x*x)
  altered_scales=tf.math.pow(10.0,db)#一様分布
  formul = tf.sqrt(tf.math.divide_no_nan(altered_scales,scales))
  #formul = tf.expand_dims(formul,-1)
  #formul = tf.tile(formul,[num_samples])
  sounds = tf.math.scalar_mul(formul,x)
  
  #with tf.Session() as sess:
  #  a=tf.debugging.Assert(sounds.shape ==x.shape,[sounds.shape])
  #sounds=tf.expand_dims(sounds,1)
  return sounds
def calcdir(dirpath):
  hparams = model.get_model_hparams()
  files=glob.glob(dirpath)
  results= None
  for file in files:
    wav,sample_rate = readwav(file)
    sample_rate = float(sample_rate)
    #print("sample rate : {}".format(sample_rate))
    #print("wav : {}".format(wav.shape))
    transformer = signal_transformer.SignalTransformer(
        sample_rate=sample_rate,
        window_time_seconds=hparams.ws,
        hop_time_seconds=hparams.hs)
    spec = transformer.forward(wav)
    spec = tf.abs(spec)
    #print(spec.shape)
    time_mean_spec = tf.math.reduce_mean(spec,[0])#時間軸で平均をとる
    time_mean_spec=tf.expand_dims(time_mean_spec, 0)
    time_mean_spec=tf.tile(time_mean_spec,[spec.shape[0],1])#specと同じ形状に
    #print(time_mean_spec.shape)
    subed_specs=spec-time_mean_spec
    vars = subed_specs**2
    if results is None:
        results= vars
    else:
        results= tf.concat([results, vars], axis=0)
  if results is not None:
    result = tf.math.reduce_mean(results)
  else:
    result = tf.constant(0)
  print("{} : {}".format(dirpath,result.numpy()))
  return result.numpy()
  
def main():
  
  parser = argparse.ArgumentParser(
      description='Train the DCASE2020 FUSS baseline source separation model.')
  parser.add_argument(
      '-nr', '--noiseroot', help='DiscriminatorModelDir',
      required=True)
      
  args = parser.parse_args()
  noisekinds=["bird","constructionSite","crowd","fountain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
  for noise in noisekinds:
    dir=args.noiseroot+"/"+noise+"_bg/**/"+noise+"_bg*.wav"
    calcdir(dir)


if __name__ == '__main__':
    main()