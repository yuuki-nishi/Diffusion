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
"""Evaluate separated audio from a DCASE 2020 task 4 separation model."""

import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import soundfile
import os
import glob
import csv
import inference
from train import data_io
from train import metrics
from train import soundclass

def evaluate(checkpoint_path, metagraph_path, data_list_path, output_path,configpath):
  """Evaluate a model on FUSS data."""
  
  #clist = glob.glob(checkpoint_path+"/*ckpt")
  #checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
  print(checkpoint_path)
  model = inference.DiscriminateModel(checkpoint_path, metagraph_path)
  print(data_list_path)
  #print(file_list)
  with model.graph.as_default():
    dataset = data_io.wavs_to_dataset_disc(
                                      configpath,
                                      "eval",
                                      data_list_path,
                                      batch_size=1,
                                      num_samples=160000,
                                      repeat=False)
                                    
    #print(dataset)
    
    '''import tensorflow.compat.v1 as tf
    dataset=tf.squeeze(dataset,[0])'''
    # reduce batch axis
    #dataset = tf.squeeze(dataset,[0])
    #dataset = dataset[0]
    # Strip batch and mic dimensions.
    '''
    dataset['voice_label'] = dataset['voice_label'][0, 0]
    dataset['source_image'] = dataset['source_image'][0, 0]
    dataset['source_name'] = dataset['source_name'][0, 0]
    '''
  # Separate with a trained model.
  i = 1
  tp = 0
  fn = 0
  tn = 0
  fp = 0
  scoresum=0
  config=data_io.SoundConfig(data_list_path,"eval",configpath)
  scores=[]
  def getindex(x):
    for i in range(len(x)):
      if x[i]==1:
        return i
    return -1
  oknum=0
  with open(output_path+'/result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    while True:
      try:
        waveform = model.sess.run(dataset)
      except tf.errors.OutOfRangeError:
        break
      #print(waveform['source_image'].shape)
      possiblity = model.getprobability(waveform['source_image'])[0]
      label = waveform['label'][0]
      score = 0
      argmax = np.argmax(possiblity)
      labelindex=np.argmax(label)
      #for x, y in zip(possiblity,label):
      #    score += x * float(y)
      score = possiblity[labelindex]
      scoresum+=score
      scores.append(score)
      print(possiblity)
      #labelindex=getindex(label)
      print("label={},{}, posibility={}, argmax={}".format(labelindex,config.SoundList[labelindex],possiblity[labelindex],argmax))
      if labelindex == argmax:
        oknum+=1
      labelname = config.SoundList[labelindex]
      writer.writerow([labelindex,possiblity[labelindex],labelname])
      i+=1

    print("score mean {}".format(scoresum/i)) 
    writer.writerow(["score mean, std,ateta ,examplenum%"])
    writer.writerow([scoresum/i,np.std(scores),oknum,i])
