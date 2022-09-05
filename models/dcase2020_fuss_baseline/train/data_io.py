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
"""Tensorflow input/output utilities."""

import collections
import os
import glob
import random
import typing as tp
import tensorflow_datasets as tfds
#from models.dcase2020_fuss_baseline.train.discmodel import HParams
import tensorflow.compat.v1 as tf
import numpy as np
import sys
from . import soundclass,sounddata
from typing import List
from multiprocessing import Pool
import multiprocessing
import copy
num_samples = 160000
class Features(object):
  """Feature keys."""

  # Waveform(s) of audio observed at receiver(s).
  RECEIVER_AUDIO = 'receiver_audio'

  # Images of each source at each microphone, including reverberation.
  # Images are real valued with shape [sources * microphones, length].
  SOURCE_IMAGES = 'source_images'
  LABEL = 'label'

def get_wavsample_fromlabel(params,label,split,num):
  
  dir=params["processed_data_dir"]+"/"+label+"/"+split + "/*wav"
  files = glob.glob(dir)
  audio_bytes = tf.io.read_file(wav)
  #print(audio_bytes)
  waveform, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
  return
class SoundConfig(object):
  Name = "Default"
  Types : List[str] = []
  SoundNumDict = {}
  SoundPathDict = {}
  MaxNum=-1#最大個数、4,5とかを想定
  SoundDict={}
  SoundList=[]
  def __init__(self,sourceroot,split,path):#pathはconfigファイルへのpath
    if path == None:
      return
    #print("configpath : {}".format(path))
    with open(path) as f:
      minvsum=0
      for line in f:
        #a:1-2みたいに書く
        splited_line=line.split(":")
        content_nn=line.replace('\n', '')
        #print(splited_line)
        if splited_line[0]=="Name":
          self.Name=splited_line[1]
        elif content_nn=="zeros":
          self.SoundPathDict["zeros"]=[]#discriminator用 zerosを指定
        else:
          labelname=splited_line[0]
          content=splited_line[1]
          content=content.replace('\n', '')
          #print(content)
          minv=int(content.split(",")[0])
          maxv=int(content.split(",")[1])
          minvsum+=maxv-minv#数がランダムなら無音があり得る
          #print(minv)
          #print(maxv)
          if labelname=="person":
            #人だった場合
            self.SoundNumDict["person"]=[minv,maxv]#1,2は最低1地最高2人といういみ
            for p in sounddata.needpersion[0:maxv]:#閉区間であることに注意
              dir=sourceroot+"/"+p+"/"+split + "/*wav"
              files = glob.glob(dir)
              #print(files)#ok
              #files_file = [f for f in files if os.path.isfile(f)]#ファイルのみを抽出 ノイズはそのままだが、人はpersonとなることに注意
              self.SoundPathDict[p]=files#ファイル一覧を格納
          else:
            #ノイズ
            self.SoundNumDict[labelname]=[minv,maxv]
            #print(sourceroot)
            #print(labelname)
            #print(split)#これがnone
            dir=sourceroot+"/"+labelname+"/"+split + "/*wav"
            #print(dir)
            files = glob.glob(dir)
            #files_file = [f for f in files if os.path.isfile(f)]#ファイルのみを抽出 ノイズはそのままだが、人はpersonとなることに注意
            #print(files_file)
            self.SoundPathDict[labelname]=files#ファイル一覧を格納

      
      if minvsum>0:#ノイズ0があり得る場合
        self.SoundPathDict["zeros"]=[]
      self.setmaxnum()
      self.SoundDict = {k: v for v, k in enumerate(self.SoundPathDict.keys())}
      print("sounddict : {}".format(self.SoundDict))
      self.SoundList = [name for i,name in enumerate(self.SoundPathDict.keys())]
    return
  #preprocess用
  """def __init__(self,sourceroot,split,classes: list[str] ):#pathはconfigファイルへのpath
    
    
    minvsum=0
    print(classes)
    for c in classes:
      #print(content)
      minv=1
      maxv=1
      minvsum+=maxv-minv#数がランダムなら無音があり得る
      #print(minv)
      #print(maxv)
      labelname = c
      if labelname in sounddata.personid:
        #人だった場合
        dir=sourceroot+"/"+labelname+"/"+split + "/*wav"
        files = glob.glob(dir)
        #print(files)#ok
        #files_file = [f for f in files if os.path.isfile(f)]#ファイルのみを抽出 ノイズはそのままだが、人はpersonとなることに注意
        self.SoundPathDict[labelname]=files#ファイル一覧を格納
      else:
        #ノイズ
        self.SoundNumDict[labelname]=[1,1]
        #print(sourceroot)
        #print(labelname)
        #print(split)#これがnone
        dir=sourceroot+"/"+labelname+"/"+split + "/*wav"
        #print(dir)
        files = glob.glob(dir)
        #files_file = [f for f in files if os.path.isfile(f)]#ファイルのみを抽出 ノイズはそのままだが、人はpersonとなることに注意
        #print(files_file)
        self.SoundPathDict[labelname]=files#ファイル一覧を格納

    
    if minvsum>0:#ノイズ0があり得る場合
      self.SoundPathDict["zeros"]=[]
    self.setmaxnum()
    self.SoundDict = {k: v for v, k in enumerate(self.SoundPathDict.keys())}
    print("sounddict : {}".format(self.SoundDict))
    self.SoundList = [name for i,name in enumerate(self.SoundPathDict.keys())]
    return"""
  def setmaxnum(self):
    tmp=0
    for k in self.SoundNumDict.keys():
      tmp+=self.SoundNumDict[k][1]
    self.MaxNum=tmp
    return
  def getclassnum_fordisc(self):
    return len(list(self.SoundPathDict.keys()))
  def getminmaxnum(self,classname:str):#クラスの引数を見る
    return self.SoundNumDict[classname]
  def getdatalist(self):
    ret=[]
    forzero=self.MaxNum
    for k in self.SoundNumDict.keys():
        #具体的なフルパスを格納
          
        if k=="person":
          #1,1は最小1人最大1人という意味
          num=random.randint(self.SoundNumDict[k][0],self.SoundNumDict[k][1])#実際にしゃべる人の数
          pdata=sounddata.needpersion[0:self.SoundNumDict[k][1]].copy()
          random.shuffle(pdata)
          oklabels=pdata[0:num]
          forzero-=num
          for i in range(num):#num==1なら1回
            inte=i
            ret.append(random.choice(self.SoundPathDict[oklabels[inte]]))
        else:
          num=random.randint(self.SoundNumDict[k][0],self.SoundNumDict[k][1])#0か1のはず、とりあえず1の時のみ
          forzero-=num
          #print("k : {}".format(k))
          #print(self.SoundPathDict)
          if num==1:
            ret.append(random.choice(self.SoundPathDict[k]))
    return ret
  def getdata_fordisc(self):
    key = random.choice(list(self.SoundPathDict.keys()))
    if key=="zeros":
      return "zeros"
    ret=random.choice(self.SoundPathDict[key])
    return ret
  def getclasslist(self,label : str):
    #print(self.SoundDict)
    ret = [0 for i in range(len(self.SoundDict))]
    ret[self.SoundDict[label]]=1
    return ret
  def getname(self,labelarray):
    #print(self.SoundDict)
    return self.SoundList[np.argmax(labelarray)]
  def getindex(self,labelname:str) -> int:
    #print(self.SoundDict)
    return self.SoundDict[labelname]

def get_inference_spec(num_receivers=1,
                      classnum=-1,
                      num_samples=None):
  """Returns a specification of features in tf.Examples in roomsim format."""
  spec = {}
  spec[Features.RECEIVER_AUDIO] = tf.FixedLenFeature(
      [num_receivers, num_samples], tf.float32)
  spec[Features.LABEL] = tf.FixedLenFeature(
      [classnum], tf.int32)

  return spec

def get_inference_spec_disc(num_samples=None,classnum=-1):
  spec = {}
  spec["source_image"] = tf.FixedLenFeature(
      [num_samples], tf.float32)
  spec["label"] = tf.FixedLenFeature(
      [classnum],tf.int32)

  return spec


def get_roomsim_spec(num_sources,
                     num_receivers,
                     num_samples):
  """Returns a specification of features in tf.Examples in roomsim format.

  Args:
    num_sources: Expected number of sources.
    num_receivers: Number of microphones in array.
    num_samples: Expected length of sources in samples. 'None' for variable.

  Returns:
    Feature specifications suitable to pass to tf.parse_example.
  """
  spec = {}
  spec[Features.RECEIVER_AUDIO] = tf.FixedLenFeature(
      [num_receivers, num_samples], tf.float32)
  spec[Features.SOURCE_IMAGES] = tf.FixedLenFeature(
      [num_sources, num_receivers, num_samples], tf.float32)
  return spec

def placeholders_from_spec(feature_spec):
  """Returns placeholders compatible with a given feature spec."""
  placeholders = {}
  for key, feature in feature_spec.items():
    placeholders[key] = tf.placeholder(dtype=feature.dtype,
                                       #shape=[1] + feature.shape,
                                       shape=[None,1,None],
                                       name=key)
  print("placeholder_sep")
  print(placeholders)
  return placeholders

def placeholders_from_spec_disc(feature_spec):
  """Returns placeholders compatible with a given feature spec."""
  placeholders = {}
  for key, feature in feature_spec.items():
    print("placeholder key")
    print(key)
    print(feature)
    #print(feature.dtype)
    if key == "source_image":
      print("source image key")
      placeholders[key] = tf.placeholder(dtype=feature.dtype,
                                       #shape=[1] + feature.shape,
                                       shape=[None,None,160000],#added batch axis and other
                                       name=key)
    else:
      placeholders[key] = tf.placeholder(dtype=feature.dtype,
                                       shape=[None,None],
                                       name=key)
  print("placeholder")
  print(placeholders)
  return placeholders


def read_lines_from_file(file_list_path, skip_fields=0, base_path='relative'):
  """Read lines from a file.

  Args:
    file_list_path: String specifying absolute path of a file list.
    skip_fields: Skip first n fields in each line of the file list.
    base_path: If `relative`, use base path of file list. If None, don't add
        any base path. If not None, use base_path to build absolute paths.

  Returns:
    List of strings, which are tab-delimited absolute file paths.
  """
  # Read in and preprocess the file list.
  print("skip_fielsd : {}".format(skip_fields))
  with open(file_list_path, 'r') as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  lines = [line.split('\t')[skip_fields:] for line in lines]

  # Make each relative path point to an absolute path.
  lines_abs_path = []
  if base_path == 'relative':
    base_path = os.path.dirname(file_list_path)
  elif base_path is None:
    base_path = ''
  for line in lines:
    wavs_abs_path = []
    for wav in line:
      wavs_abs_path.append(os.path.join(base_path, wav))
    lines_abs_path.append(wavs_abs_path)
  #print(lines)
  lines = lines_abs_path

  # Rejoin the fields to return a list of strings.
  return ['\t'.join(fields) for fields in lines]

def read_lines_from_file_disc(file_list_path, skip_fields=0, base_path='relative'):
  """Read lines from a file.

  Args:
    file_list_path: String specifying absolute path of a file list.
    skip_fields: Skip first n fields in each line of the file list.
    base_path: If `relative`, use base path of file list. If None, don't add
        any base path. If not None, use base_path to build absolute paths.

  Returns:
    List of strings, which are tab-delimited absolute file paths.
  """
  # Read in and preprocess the file list.
  with open(file_list_path, 'r') as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  lines = [line.split(' ')[skip_fields:] for line in lines]
  # Make each relative path point to an absolute path.
  lines_abs_path = []
  if base_path == 'relative':
    base_path = os.path.dirname(file_list_path)
  elif base_path is None:
    base_path = ''
  for line in lines:
    wavs_abs_path = []
    wavs_abs_path.append(os.path.join(base_path, line[0]))
    wavs_abs_path.append(line[1])
    lines_abs_path.append(wavs_abs_path)
  lines = lines_abs_path
  return lines

def unique_classes_from_lines(lines):
  """Return sorted list of unique classes that occur in all lines."""
  # Build sorted list of unique classes.
  unique_classes = sorted(
      list({x.split(':')[0] for line in lines for x in line}))  # pylint: disable=g-complex-comprehension
  return unique_classes


def get_one_noisefile(noisedir):
  #noisedir = "/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Monoral_Isolated_urban_sound/background/bird_bg"
  noisefiles = glob.glob("{}/*wav".format(noisedir))
  #choice randomly 1 file
  noisefile = random.choice(noisefiles)
  return noisefile

def scaling(x,indb):
    #音の配列xをちょうどいい音量に変更する
    #x=tf.squeeze(x,axis=[1])#(1,160000)だから
    indb=tf.reshape(indb,[2])
    scale = tf.reduce_mean(x*x,axis=-1)
    altered_scale= tf.math.pow(10.0,tf.random_uniform([1],minval=indb[0],maxval=indb[1], dtype=tf.float32))#一様分布
    formul = tf.sqrt(tf.math.divide_no_nan(altered_scale,scale))#[1]のはず
    formul = tf.expand_dims(formul,0)
    formul = tf.tile(formul,[1,num_samples])
    sounds = formul*x
    #ここからは0対策
    sumfor0 = tf.reduce_sum(x*x)
    #0の時はやらんでいいから
    sounds=tf.cond(sumfor0 < 0.00001,lambda: tf.random.uniform(sounds.shape,-0.0001,0.0001), lambda:sounds)
    
    return sounds

# Read in wav files.
def decode_wav(wav):
  audio_bytes = tf.read_file(wav)
  waveform, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1,
                                    desired_samples=num_samples)
  waveform = tf.reshape(waveform, (1, num_samples))
  return waveform
def decode_wav_noise(wav):
  #print(wav)
  audio_bytes = tf.io.read_file(wav)
  #print(audio_bytes)
  waveform, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
  scalar = 0
  startindex = int(random.randrange(0,(scalar-num_samples+1)))
  waveform = tf.convert_to_tensor(waveform[startindex:startindex+num_samples])
  waveform = tf.reshape(waveform, [num_samples])
  return waveform
def getpersonid(x)->str:
  filename = os.path.basename(x)
  return x[0:3]#初めの3文字だから
def decode_wav_or_return_zeros(wav):
  return tf.cond(tf.equal(wav, 'zeros'),#ここをwav を 'zeros'にしてもだめだった
                  lambda: tf.zeros((1, num_samples), dtype=tf.float32),
                  lambda: decode_wav(wav))

def setdb(label):
  """if label in sounddata.personid:
    return [-5,-1]
  else:
    return [-4,-2]"""
  if label in sounddata.personid:
    ret= [-6,-3]
  else:
    ret= [-6,-3]
  ret = [(i/10)-1 for i in ret]
  return ret
def wavs_to_dataset(configpath,
                    split,
                    sourceroot,
                    batch_size,
                    datanum=2000,
                    num_samples=-1,
                    parallel_readers=1,
                    randomize_order=True,
                    combine_by_class=False,
                    max_sources_override=None,
                    num_examples=-1,
                    shuffle_buffer_size=50,
                    repeat=True):
  r"""Fetches features from list of wav files.

  Args:
    file_list: List of tab-delimited file locations of wavs. Each line should
        correspond to one example, where each field is a source wav.
    batch_size: The number of examples to read.
    num_samples: Number of samples in each wav file.
    parallel_readers: Number of fetches that should happen in parallel.
    randomize_order: Whether to randomly shuffle features.
    combine_by_class: Whether to add together events of the same class.
        Note that this assumes the file list has class annotations, in format:
        '<class 1>:<filename 1>\t<class 2>:<filename 2>
        The number of output sources N depends on fixed_classes:

        If fixed_classes contains all unique classes, N will be the number of
        unique classes in the file list. Each class will have a fixed output
        index, where the order of indices is order of fixed_classes.

        If fixed_classes contains a subset of unique classes, N will be number
        of fixed classes plus maximum number of nonfixed classes in any line
        in the file. For example, if a dataset contains unique classes 'dog',
        'cat', 'parrot', and fixed_classes is ['dog'], and every line only
        contains the classes ['dog', 'cat'] or ['dog', 'parrot'], then the
        number of output sources will be 2, and the 'dog' class will always be
        output at source index 0. If there are M fixed_classes, the first M
        sources will be fixed, and the remaining N - M sources will be nonfixed.

        If fixed_classes is empty, N will be the maximum number of
        unique class occurrences in any line in the file.

    fixed_classes: List of classes to place at fixed source indices.
        Classes that are not in these keys are placed in remaining source
        indices in the order they appear in the file list line.
    max_sources_override: Override maximum number of output sources. Raises
        error if this number is less than assumed max number of sources N.
    num_examples: Limit number of examples to this value.  Unlimited if -1.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: If True, repeat the dataset.

  Returns:
    A batch_size number of features constructed from wav files.

  Raises:
    ValueError if max_sources_override is less than assumed max number sources.
  """
  
  #print(noisedir)
  
  config = SoundConfig(sourceroot,split,configpath)
  max_combined_sources = config.MaxNum
  # Override max sources, if specified.
  
  if max_sources_override:
    if max_sources_override > max_combined_sources:
      max_combined_sources = max_sources_override
    elif max_sources_override < max_combined_sources:
      raise ValueError('max_sources_override of {} is less than assumed max'
                        'combined sources of {}.'.format(max_sources_override,
                                                        max_combined_sources))
  assert config.MaxNum <= max_combined_sources
  print(config.SoundDict)
  print(config.SoundNumDict)
  print("datanum : {}".format(datanum))
  lines = [config.getdatalist() for i in range(datanum)]#パスの配列
  # Examples that have fewer than max_component_sources are padded with zeros.
  lines = [line + ['zeros'] * (max_combined_sources - len(line)) for line in lines]
  #lines = [line.split('\t') for line in file_list]
  max_component_sources = max([len(line) for line in lines])#txtファイルの中にあるリストのうち、行の中で最もsourceが記載されている数
  print("combine_by_class : {}".format(combine_by_class))
  print("max_component_sources : {}".format(max_component_sources))
  print("max_source_override : {}".format(max_sources_override))
  
  print("max_combined_sources : {}".format(max_combined_sources))
  #print(lines)
  files1=lines
  print(files1[0])
  print(files1[1])
  print(files1[2])
  print(files1[3])
  files2=[ [config.getclasslist(soundclass.getlabelfrompath(comp)) for comp in line] for line in lines]
  files3=[ [tf.cast(setdb(soundclass.getlabelfrompath(lines[i][j])),dtype=tf.float32) for j in range(len(lines[i]))] for i in range(len(lines))]#音量db
  #print(files2)
  dataset = tf.data.Dataset.from_tensor_slices({"wav":files1,"label":files2,"db":files3})
  def decode_wav_wrap(x):
    wav=x["wav"]
    print(wav.shape)
    #wavss=tf.squeeze(wav,axis=[-1])
    wav2=tf.map_fn(decode_wav_or_return_zeros,wav,dtype=tf.float32)
    print(wav2)
    return {"wav":wav2,"label" : x["label"],"db":x["db"]}
  dataset = dataset.map(decode_wav_wrap)
  print("tmp data shape : {}".format(len(files1)))
  #dataset = dataset.batch(max_component_sources)

  @tf.function()
  def scaling(x,indb):
    #音の配列xをちょうどいい音量に変更する
    #print(x.shape)
    x=tf.squeeze(x,axis=[1])#(None,1,160000)だから
    indb=tf.reshape(indb,[x.shape[0],2])
    scales = tf.map_fn(lambda x:tf.reduce_mean(x*x),x)
    altered_scales= tf.map_fn(lambda db:tf.math.pow(10.0,tf.random_uniform([1],minval=db[0],maxval=db[1],seed = 114514, dtype=tf.float32)[0]),indb)#一様分布
    formul = tf.sqrt(tf.math.divide_no_nan(altered_scales,scales))
    formul = tf.expand_dims(formul,-1)
    formul = tf.tile(formul,[1,num_samples])
    sounds = formul*x
    
    #with tf.Session() as sess:
    #  a=tf.debugging.Assert(sounds.shape ==x.shape,[sounds.shape])
    sounds=tf.expand_dims(sounds,1)
    return sounds

  # Build mixture and sources waveforms.
  def combine_mixture_and_sources(dwaveforms):
    # waveforms is shape (max_combined_sources, 1, num_samples).
    #print("4ewew")
    waveforms = dwaveforms["wav"]
    waveforms=scaling(waveforms,dwaveforms["db"])
    label = dwaveforms["label"]
    """print(waveforms)
    print(waveforms.shape)
    tmp=0
    if withNoiseadd:
      waveforms = addnoise(waveforms)
      tmp=1
    print("waveforms.shape")
    print(waveforms.shape)"""
    mixture_waveform = tf.reduce_sum(waveforms, axis=0)#ここで混合をしている
    mixture_waveform = tf.reshape(mixture_waveform, (num_samples,))
    source_waveforms = tf.reshape(waveforms,
                                  #(max_combined_sources+tmp, 1, num_samples))
                                  (max_combined_sources, num_samples))
    """if not withNoiseadd:
      print("not withnoise add")
      mixture_waveform = mixnoise(mixture_waveform)
    print(mixture_waveform.shape)"""
    """return {'receiver_audio': mixture_waveform,
            'source_images': source_waveforms,
            'label':label}"""
    return (mixture_waveform,source_waveforms)
  dataset = dataset.map(combine_mixture_and_sources)
  #print(dataset.cardinality().eval())
  if randomize_order:
    dataset = dataset.shuffle(shuffle_buffer_size,seed=114514)
  dataset = dataset.prefetch(parallel_readers)
  dataset = dataset.take(num_examples)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  if repeat:
    dataset = dataset.repeat()

  iterator = dataset.make_one_shot_iterator()
  #iterator = dataset.make_initializable_iterator()
  #return iterator.get_next()

  return dataset






def wavs_to_dataset_disc(configpath,
                    split,
                    sourceroot,
                    batch_size,
                    datanum=2000,
                    num_samples=-1,
                    parallel_readers=1,
                    randomize_order=True,
                    num_examples=-1,
                    shuffle_buffer_size=50,
                    repeat=True):
  r"""Fetches features from list of wav files.

  Args:
    file_list: List of tab-delimited file locations of wavs. Each line should
        correspond to one example, where each field is a source wav.
    batch_size: The number of examples to read.
    num_samples: Number of samples in each wav file.
    parallel_readers: Number of fetches that should happen in parallel.
    randomize_order: Whether to randomly shuffle features.
    combine_by_class: Whether to add together events of the same class.
        Note that this assumes the file list has class annotations, in format:
        '<class 1>:<filename 1>\t<class 2>:<filename 2>
        The number of output sources N depends on fixed_classes:

        If fixed_classes contains all unique classes, N will be the number of
        unique classes in the file list. Each class will have a fixed output
        index, where the order of indices is order of fixed_classes.

        If fixed_classes contains a subset of unique classes, N will be number
        of fixed classes plus maximum number of nonfixed classes in any line
        in the file. For example, if a dataset contains unique classes 'dog',
        'cat', 'parrot', and fixed_classes is ['dog'], and every line only
        contains the classes ['dog', 'cat'] or ['dog', 'parrot'], then the
        number of output sources will be 2, and the 'dog' class will always be
        output at source index 0. If there are M fixed_classes, the first M
        sources will be fixed, and the remaining N - M sources will be nonfixed.

        If fixed_classes is empty, N will be the maximum number of
        unique class occurrences in any line in the file.

    fixed_classes: List of classes to place at fixed source indices.
        Classes that are not in these keys are placed in remaining source
        indices in the order they appear in the file list line.
    max_sources_override: Override maximum number of output sources. Raises
        error if this number is less than assumed max number of sources N.
    num_examples: Limit number of examples to this value.  Unlimited if -1.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: If True, repeat the dataset.

  Returns:
    A batch_size number of features constructed from wav files.

  Raises:
    ValueError if max_sources_override is less than assumed max number sources.
  """
  
  #print(lines)
  #max_component_sources = max([len(line) for line in lines])#txtファイルの中にあるリストのうち、行の中で最もsourceが記載されている数

  wav_filenames = []
  class_id_list = []
  wavs = []
  print("lines")
  #print(lines)
  #tf.print("lines")
  #tf.print(lines)
  
  
  config = SoundConfig(sourceroot,split,configpath)
  
  files1 = [config.getdata_fordisc() for i in range(datanum)]#パスの配列
  #files1=[lines[i][0] for i in range(len(lines))]
  files2=[ config.getclasslist(soundclass.getlabelfrompath(line)) for line in files1]
  files3=[ tf.cast(setdb(soundclass.getlabelfrompath(line)),dtype=tf.float32) for line in files1]#音量db
  
  dataset = tf.data.Dataset.from_tensor_slices({"source_image":files1,"label":files2,"db":files3})
  print("datanum : {}".format(datanum))
  #dataset = dataset.map(decode_wav_or_return_zeros)
  #dataset = dataset.batch(max_component_sources)

  
  @tf.function()
  def scaling(x,indb):
    #音の配列xをちょうどいい音量に変更する
    #x=tf.squeeze(x,axis=[1])#(1,160000)だから
    indb=tf.reshape(indb,[2])
    scale = tf.reduce_mean(x*x,axis=-1)
    altered_scale= tf.math.pow(10.0,tf.random_uniform([1],minval=indb[0],maxval=indb[1], dtype=tf.float32))#一様分布
    formul = tf.sqrt(tf.math.divide_no_nan(altered_scale,scale))#[1]のはず
    formul = tf.expand_dims(formul,0)
    formul = tf.tile(formul,[1,num_samples])
    sounds = formul*x
    #ここからは0対策
    sumfor0 = tf.reduce_sum(x*x)
    #0の時はやらんでいいから
    sounds=tf.cond(sumfor0 < 0.00001,lambda: tf.random.uniform(sounds.shape,-0.0001,0.0001), lambda:sounds)
    #絶対値の最大値が1になるようにする
    maxm = tf.reduce_max(tf.math.abs(sounds))
    sounds=tf.cond(maxm > 1.0,lambda: sounds/maxm, lambda:sounds)
    return sounds
  # Build mixture and sources waveforms.
  def make_labels_waveforms(wav):
    source_waveforms = decode_wav_or_return_zeros(wav["source_image"])
    source_waveforms = scaling(source_waveforms,wav["db"])
    source_waveforms = tf.squeeze(source_waveforms)#[16000]の形
    """return {'label': wav["label"],
            'source_image': source_waveforms}"""
    return (source_waveforms,wav["label"])
    #        'label': [label]}
  print("beforelabel")
  #writer = tf.data.experimental.TFRecordWriter("/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutput.txt")
  #writer.write(dataset)
  '''for item in dataset:
    print("item : ".format(item))'''
  dataset = dataset.map(make_labels_waveforms)
  #writer = tf.data.experimental.TFRecordWriter("/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutput2.txt")
  #writer.write(dataset)


  print("dataset beforebatch")
  #print(dataset)
  #print(len(list(dataset)))

  if randomize_order:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.prefetch(parallel_readers)
  #dataset = dataset.take(num_examples)
  print("batch : {}".format(batch_size))
  if batch_size > 0:
    dataset = dataset.batch(batch_size, drop_remainder=True)#model.firのところでやる
  if repeat:
    dataset = dataset.repeat()

  #iterator = dataset.make_one_shot_iterator()
  #iterator = dataset.make_initializable_iterator()
  #return iterator.get_next()
  return dataset

def wavs_to_dataset_formprocessed(configpath,
                    split,
                    sourceroot,
                    batch_size,
                    preprocessed_data_dir,
                    datanum=2000,
                    num_samples=-1,
                    parallel_readers=1,
                    randomize_order=True,
                    num_examples=-1,
                    shuffle_buffer_size=50,
                    repeat=True):
  r"""Fetches features from list of wav files.

  Args:
    file_list: List of tab-delimited file locations of wavs. Each line should
        correspond to one example, where each field is a source wav.
    batch_size: The number of examples to read.
    num_samples: Number of samples in each wav file.
    parallel_readers: Number of fetches that should happen in parallel.
    randomize_order: Whether to randomly shuffle features.
    combine_by_class: Whether to add together events of the same class.
        Note that this assumes the file list has class annotations, in format:
        '<class 1>:<filename 1>\t<class 2>:<filename 2>
        The number of output sources N depends on fixed_classes:

        If fixed_classes contains all unique classes, N will be the number of
        unique classes in the file list. Each class will have a fixed output
        index, where the order of indices is order of fixed_classes.

        If fixed_classes contains a subset of unique classes, N will be number
        of fixed classes plus maximum number of nonfixed classes in any line
        in the file. For example, if a dataset contains unique classes 'dog',
        'cat', 'parrot', and fixed_classes is ['dog'], and every line only
        contains the classes ['dog', 'cat'] or ['dog', 'parrot'], then the
        number of output sources will be 2, and the 'dog' class will always be
        output at source index 0. If there are M fixed_classes, the first M
        sources will be fixed, and the remaining N - M sources will be nonfixed.

        If fixed_classes is empty, N will be the maximum number of
        unique class occurrences in any line in the file.

    fixed_classes: List of classes to place at fixed source indices.
        Classes that are not in these keys are placed in remaining source
        indices in the order they appear in the file list line.
    max_sources_override: Override maximum number of output sources. Raises
        error if this number is less than assumed max number of sources N.
    num_examples: Limit number of examples to this value.  Unlimited if -1.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: If True, repeat the dataset.

  Returns:
    A batch_size number of features constructed from wav files.

  Raises:
    ValueError if max_sources_override is less than assumed max number sources.
  """
  
  #print(lines)
  #max_component_sources = max([len(line) for line in lines])#txtファイルの中にあるリストのうち、行の中で最もsourceが記載されている数
  if datanum > 10000:
    print("exceed data num")
  wav_filenames = []
  class_id_list = []
  wavs = []
  print("lines")
  #print(lines)
  #tf.print("lines")
  #tf.print(lines)
  
  def getwavs(label):
    pathes = preprocessed_data_dir + "/{}/{}/*.wav".format(label,split)
    return glob.glob(pathes)
  
  config = SoundConfig(sourceroot,split,configpath)
  random.seed(114514)
  keys=list(config.SoundDict.keys())#labelの集合
  files1 = [random.sample(keys,len(keys)) for i in range(datanum)]#並び替えたlabels
  #print(files1)
  files_label_wav = { key :getwavs(key) for key in keys}
  #print(files_label_wav)#okだった？
  #ランダム並べ替え
  for key in files_label_wav:
    random.shuffle(files_label_wav[key])
  files3 = [[ files_label_wav[label][i] for label in files1[i] ] for i in range(datanum)]
  labelsnum = copy.deepcopy(config.SoundDict)
  #出力テスト用
  #files1 = [config.getclasslist("400") for label in files1]#全部同じラベルにしたらちゃんとlossは減った
  files1 = [[config.SoundDict[label] for label in labels ] for labels in files1]#並び替えたlabelsをone-hot配列にした
  dataset = tf.data.Dataset.from_tensor_slices({"source_images":files3,"labels":files1})
  def loadwave(dwaveforms):
    pathes = dwaveforms["source_images"]
    def decodewave(path):
      audio_bytes = tf.io.read_file(path)
      #print(audio_bytes)
      waveform, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
      print(waveform)
      return waveform
    waveforms=tf.map_fn(decodewave,pathes,dtype=tf.float32)
    waveforms=tf.reshape(waveforms,[len(keys),num_samples])
    
    labels=dwaveforms["labels"]
    mixtured_waveform = tf.reduce_sum(waveforms,axis=0)#class axisでsumを取る
    mixtured_waveform=tf.reshape(mixtured_waveform,[num_samples])
    #return (mixtured_waveform,waveforms)
    return ({"input_mixture":mixtured_waveform},{"out":waveforms})
  dataset = dataset.map(loadwave)
  #writer = tf.data.experimental.TFRecordWriter("/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutput2.txt")
  #writer.write(dataset)


  print("dataset beforebatch")
  #print(dataset)
  #print(len(list(dataset)))

  if randomize_order:
    dataset = dataset.shuffle(datanum)
  dataset = dataset.prefetch(parallel_readers)
  #dataset = dataset.take(num_examples)
  print("batch : {}".format(batch_size))
  if batch_size > 0:
    dataset = dataset.batch(batch_size, drop_remainder=True)#model.firのところでやる
  if repeat:
    dataset = dataset.repeat()

  #iterator = dataset.make_one_shot_iterator()
  #iterator = dataset.make_initializable_iterator()
  #return iterator.get_next()
  return dataset
def wavs_to_dataset_dict_formprocessed(configpath,
                    split,
                    sourceroot,
                    batch_size,
                    preprocessed_data_dir,
                    datanum=2000,
                    num_samples=-1,
                    parallel_readers=1,
                    randomize_order=True,
                    num_examples=-1,
                    shuffle_buffer_size=50,
                    repeat=True):
  
  if datanum > 10000:
    print("exceed data num")
  wav_filenames = []
  class_id_list = []
  wavs = []
  print("lines")
  
  def getwavs(label):
    pathes = preprocessed_data_dir + "/{}/{}/*.wav".format(label,split)
    return glob.glob(pathes)
  
  config = SoundConfig(sourceroot,split,configpath)
  random.seed(114514)
  keys=list(config.SoundDict.keys())#labelの集合
  files1 = [random.sample(keys,len(keys)) for i in range(datanum)]#並び替えたlabels
  #print(files1)
  files_label_wav = { key :getwavs(key) for key in keys}
  #print(files_label_wav)#okだった？
  #ランダム並べ替え
  for key in files_label_wav:
    random.shuffle(files_label_wav[key])
  files3 = [[ files_label_wav[label][i] for label in files1[i] ] for i in range(datanum)]
  labelsnum = copy.deepcopy(config.SoundDict)
  #出力テスト用
  #files1 = [config.getclasslist("400") for label in files1]#全部同じラベルにしたらちゃんとlossは減った
  files1 = [[config.getclasslist(label) for label in labels ] for labels in files1]#並び替えたlabelsをone-hot配列にし
  def loadwave(dwaveforms):
    pathes = dwaveforms["source_images"]
    def decodewave(path):
      audio_bytes = tf.io.read_file(path)
      #print(audio_bytes)
      waveform, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
      print(waveform)
      return waveform
    waveforms=tf.map_fn(decodewave,pathes,dtype=tf.float32)
    waveforms=tf.reshape(waveforms,[len(keys),num_samples])
    labels=dwaveforms["labels"]
    mixtured_waveform = tf.reduce_sum(waveforms,axis=0)#class axisでsumを取る
    mixtured_waveform=tf.reshape(mixtured_waveform,[num_samples])
    #return (mixtured_waveform,waveforms)
    #return {"mixtured":mixtured_waveform,"sources":waveforms,"label":labels,"option":[0,0]}
    return ({"mixtured":mixtured_waveform,"sources":waveforms,"label":labels,"option":[0,0]},{"separated_waveforms":waveforms,"probabilities":labels})
  def loadwave_forsource(dwaveforms):
    pathes = dwaveforms["source_images"]
    def decodewave(path):
      audio_bytes = tf.io.read_file(path)
      waveform, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
      return waveform
    waveforms=tf.map_fn(decodewave,pathes,dtype=tf.float32)
    waveforms=tf.reshape(waveforms,[len(keys),num_samples])
    #return (mixtured_waveform,waveforms)
    return waveforms
  def load_label(dwaveforms):
    labels=dwaveforms["labels"]
    return labels
    
  dataset = tf.data.Dataset.from_tensor_slices({"source_images":files3,"labels":files1})
  dataset = dataset.map(loadwave)
  dataset_forsource = tf.data.Dataset.from_tensor_slices({"source_images":files3,"labels":files1})
  dataset_forlabel =tf.data.Dataset.from_tensor_slices({"source_images":files3,"labels":files1})
  dataset_forsource = dataset_forsource.map(loadwave_forsource)
  dataset_forlabel = dataset_forlabel.map(load_label)
  #writer = tf.data.experimental.TFRecordWriter("/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutput2.txt")
  #writer.write(dataset)


  print("dataset beforebatch")
  #print(dataset)
  #print(len(list(dataset)))

  """if randomize_order:
    dataset = dataset.shuffle(datanum)"""
  dataset = dataset.prefetch(parallel_readers)
  #dataset = dataset.take(num_examples)
  print("batch : {}".format(batch_size))
  if batch_size > 0:
    dataset = dataset.batch(batch_size, drop_remainder=True)#model.firのところでやる
    dataset_forsource = dataset_forsource.batch(batch_size, drop_remainder=True)#model.firのところでやる
    dataset_forlabel = dataset_forlabel.batch(batch_size, drop_remainder=True)#model.firのところでやる
  if repeat:
    dataset = dataset.repeat()
    dataset_forsource = dataset_forsource.repeat()
    dataset_forlabel = dataset_forlabel.repeat()
  return dataset,dataset_forsource,dataset_forlabel
def wavs_to_dataset_disc_formprocessed(configpath,
                    split,
                    sourceroot,
                    batch_size,
                    preprocessed_data_dir,
                    datanum=2000,
                    num_samples=-1,
                    parallel_readers=1,
                    randomize_order=True,
                    num_examples=-1,
                    shuffle_buffer_size=50,
                    repeat=True,
                    mixup=True):
  #print(lines)
  #max_component_sources = max([len(line) for line in lines])#txtファイルの中にあるリストのうち、行の中で最もsourceが記載されている数

  wav_filenames = []
  class_id_list = []
  wavs = []
  print("lines")
  #print(lines)
  #tf.print("lines")
  #tf.print(lines)
  
  def getwavs(label):
    pathes = preprocessed_data_dir + "/{}/{}/*.wav".format(label,split)
    return glob.glob(pathes)
  config = SoundConfig(sourceroot,split,configpath)
  random.seed(114514)
  keys=list(config.SoundDict.keys())
  files1 = [random.choice(keys) for i in range(datanum)]#labels
  labelsnum = copy.deepcopy(config.SoundDict)
  for label in keys:
    labelsnum[label]=0
  #個数を決定
  for i in range(datanum):
    label = random.choice(keys)
    labelsnum[label] += 1
  files1 = []#labelを格納
  files2=[]#wavを格納
  print(labelsnum)
  print(keys)
  for label in keys:
    pathes = getwavs(label)
    pathes = pathes[0:labelsnum[label]]
    files2.extend(pathes)
    files1.extend([label for i in range(labelsnum[label])])
    print(np.shape(files2))
  files1 = [config.getclasslist(label) for label in files1]
  #出力テスト用
  #files1 = [config.getclasslist("400") for label in files1]#全部同じラベルにしたらちゃんとlossは減った
  #mixup=True
  if not mixup:
    dataset = tf.data.Dataset.from_tensor_slices({"source_image":files2,"label":files1})
    def loadwave(dwaveforms):
      waveforms = dwaveforms["source_image"]
      label=dwaveforms["label"]
      audio_bytes = tf.io.read_file(waveforms)
      #print(audio_bytes)
      waveform, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
      return (waveform,label)
    dataset = dataset.map(loadwave)
  else:
    #mixupする
    formixup_files1 = copy.deepcopy(files1)
    formixup_files2 = copy.deepcopy(files2)
    formixup_files1.reverse()
    formixup_files2.reverse()
    alpha = 0.2
    Lambdas=tf.compat.v1.distributions.Beta(alpha,alpha).sample([datanum])
    #print(Lambdas)
    dataset = tf.data.Dataset.from_tensor_slices({"source_image":files2,"label":files1,"source_image2":formixup_files2,"label2":formixup_files1,"Lambda":Lambdas})
    
    def loadwave(dwaveforms):
      waveforms = dwaveforms["source_image"]
      label1=tf.cast(dwaveforms["label"],dtype=tf.float32)
      waveforms2 = dwaveforms["source_image2"]
      label2=tf.cast(dwaveforms["label2"],dtype=tf.float32)
      audio_bytes = tf.io.read_file(waveforms)
      audio_bytes2 = tf.io.read_file(waveforms2)
      #print(audio_bytes)
      Lambda=float(dwaveforms["Lambda"])
      waveform1, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
      waveform2, sample_rate = tf.audio.decode_wav(audio_bytes2, desired_channels=1)
      waveform=Lambda*waveform1 + (1-Lambda)*waveform2
      label=Lambda*label1 + (1-Lambda)*label2
      return (waveform,label)
    dataset = dataset.map(loadwave)
  #writer = tf.data.experimental.TFRecordWriter("/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutput2.txt")
  #writer.write(dataset)


  print("dataset beforebatch")
  #print(dataset)
  #print(len(list(dataset)))

  if randomize_order:
    dataset = dataset.shuffle(datanum)
  dataset = dataset.prefetch(parallel_readers)
  #dataset = dataset.take(num_examples)
  print("batch : {}".format(batch_size))
  if batch_size > 0:
    dataset = dataset.batch(batch_size, drop_remainder=True)#model.firのところでやる
  if repeat:
    dataset = dataset.repeat()

  #iterator = dataset.make_one_shot_iterator()
  #iterator = dataset.make_initializable_iterator()
  #return iterator.get_next()
  return dataset


def input_fn(params):
  """An input function that uses params['feature_spec'].

  Args:
    params: A dictionary of experiment params.

  Returns:
    Features specified by params['feature_spec'].  If 'inference' exists and is
    True in params, then placeholders will be returned based on the spec in
    params['inference_spec'], otherwise a dataset of examples read from
    params['input_data'] will be returned.
  """
  if params.get('inference', False):
    feature_spec = params['inference_spec']
    with tf.variable_scope('input_audio'):
      return placeholders_from_spec(feature_spec)
  else:
    configpath = params.get('configpath', None)
    io_params = params.get('io_params', {})
    batch_size = params.get('batch_size', None)
    example_num = params["example_num"]
    return wavs_to_dataset_formprocessed(configpath,
    #return wavs_to_dataset(configpath,
                          params.get('split', None),
                          params.get('source_root', None),
                          batch_size,
                          preprocessed_data_dir = params["processed_data_dir"],
                          datanum=example_num,
                          repeat=False,
                          **io_params)
def input_fn_dict(params):
  """An input function that uses params['feature_spec'].

  Args:
    params: A dictionary of experiment params.

  Returns:
    Features specified by params['feature_spec'].  If 'inference' exists and is
    True in params, then placeholders will be returned based on the spec in
    params['inference_spec'], otherwise a dataset of examples read from
    params['input_data'] will be returned.
  """
  if params.get('inference', False):
    feature_spec = params['inference_spec']
    with tf.variable_scope('input_audio'):
      return placeholders_from_spec(feature_spec)
  else:
    configpath = params.get('configpath', None)
    io_params = params.get('io_params', {})
    batch_size = params.get('batch_size', None)
    example_num = params["example_num"]
    return wavs_to_dataset_dict_formprocessed(configpath,
    #return wavs_to_dataset(configpath,
                          params.get('split', None),
                          params.get('source_root', None),
                          batch_size,
                          preprocessed_data_dir = params["processed_data_dir"],
                          datanum=example_num,
                          repeat=False,
                          **io_params)
def input_fn_disc(params):
  """An input function that uses params['feature_spec'].

  Args:
    params: A dictionary of experiment params.

  Returns:
    Features specified by params['feature_spec'].  If 'inference' exists and is
    True in params, then placeholders will be returned based on the spec in
    params['inference_spec'], otherwise a dataset of examples read from
    params['input_data'] will be returned.
  """
  if params.get('inference', False):#このfalseはgetができなかった場合の返り血なので、inferenceの時はここ以下の処理になる
    feature_spec = params['inference_spec']
    with tf.variable_scope('input_audio'):
      print("scope inference")
      print(feature_spec)
      return placeholders_from_spec_disc(feature_spec)
  else:
    print("before wavtodisc")
    
    configpath = params.get('configpath', None)
    io_params = params.get('io_params', {})
    batch_size = params.get('batch_size', None)
    example_num = params["example_num"]
    return wavs_to_dataset_disc_formprocessed(configpath,
                          params.get('split', None),
                          params.get('source_root', None),
                          batch_size,
                          preprocessed_data_dir = params["processed_data_dir"],
                          datanum=example_num,
                          repeat=False,
                          **io_params)