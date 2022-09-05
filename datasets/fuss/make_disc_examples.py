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
r"""A script to mix sources.

Usage example:
python3 make_ss_examples.py -f ${FSD_DIR} -b ${FSD_DIR} \
  -o ${MIX_DIR} --allow_same 1 --num_train ${NUM_TRAIN_MIX} \
  --num_validation ${NUM_VALIDATION_MIX} --num_eval ${NUM_EVAL_MIX} \
  --random_seed ${RANDOM_SEED}
"""

import argparse
import os
import glob
import numpy as np
from scipy.io import wavfile
speakers = set()

def getclass(path):
  return os.path.basename(path)[0:3]
def fileterclass(file_list):
  return [i for i in file_list if getclass(i) in ['400', '002', '001']]
def makelist(output_root, voice_list_dir,noise_dir,rate,Class,
               num_train=200, num_validation=50, num_eval=50, random_seed=123):
    makdir(output_root)
    t_labels=open(output_root+"/train_labels.txt","w")
    v_labels=open(output_root+"/validation_labels.txt","w")
    e_labels=open(output_root+"/eval_labels.txt","w")
    np.random.seed(seed=random_seed)
    dur=160000
    def make(classname,num,writer):
      #rate means ratio of noise to 
      with open(os.path.join(voice_list_dir,classname+"_background.txt"), "r") as tf:
        lines = tf.read().split('\n')
      lines = fileterclass(lines)
      print(os.path.join(noise_dir,classname,"*.wav"))
      noiselist = glob.glob(os.path.join(noise_dir,classname,"*.wav"))
      if len(noiselist) == 0:
        return
      print(noiselist)
      for i in range(num):
        isVoice = np.random.rand() > rate
        if isVoice:
          wavfilename=np.random.choice(lines)
          wavrate,wav=wavfile.read(voice_list_dir+"/"+wavfilename)
          kind=os.path.basename(wavfilename)[0:3]
          speakers.add(kind)
          #continue
        else:
          wavfilename=np.random.choice(noiselist)
          wavrate,wav=wavfile.read(wavfilename)
          kind=Class
        name = os.path.basename(wavfilename)
        writer.write(classname+"/"+name+" "+kind+"\n")
        #wav.flags.writeable = True
        if(wav.shape[0]<dur):
          extracted=padding0(wav.T,dur)
        else:
          start=int((wav.shape[0]-dur)*np.random.rand())
          extracted=(wav.T)[start:start+dur]
        extracted=extracted.T
        #extracted=wav[start:start+dur]
        #wav=wav.T
        wdir=output_root+"/"+classname
        if not os.path.exists(wdir):# 無ければ作成
          os.makedirs(wdir)
        wavfile.write(output_root+"/"+classname+"/"+name,wavrate,extracted)
    make("train",num_train,t_labels)
    make("validation",num_validation,v_labels)
    make("eval",num_eval,e_labels)

def padding0(wav,dur):
  wavlen=len(wav)
  print(wav.shape)
  start=int((dur-wavlen)*np.random.rand())
  ret=np.concatenate([0]*start,wav,[0]*(dur-int(start+wavlen)))
  return ret
def makdir(Dir):
  if not os.path.exists(Dir):# 無ければ作成
    os.makedirs(Dir)

def main():
  parser = argparse.ArgumentParser(
      description='Mixes sources to produce mixed files.')
  parser.add_argument(
      '-v', '--voice_dir', help='Foreground sources directory.', required=True)
  parser.add_argument(
      '-n', '--noise_dir', help='Background sources directory.', required=True)
  parser.add_argument(
      '-o', '--output_dir', help='Output directory.', required=True)
  parser.add_argument(
      '-a', '--allow_same', type=bool, default=False,
      help='Allow same label in a mixture, this is necessary when using a '
      'single label class.')
  parser.add_argument(
      '-nt', '--num_train', type=int, default=200,
      help='Number of training examples to generate.')
  parser.add_argument(
      '-nv', '--num_validation', type=int, default=50,
      help='Number of validation examples to generate.')
  parser.add_argument(
      '-ne', '--num_eval', type=int, default=50,
      help='Number of eval examples to generate.')
  parser.add_argument(
      '-rs', '--random_seed', help='Random seed.', required=False, default=123,
      type=int)
  classes=["bird","counstructionSite","crowd","foutain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
  #classes=["bird"]
  args = parser.parse_args()
  for Class in classes:

    makelist(output_root=args.output_dir+"/"+Class, voice_list_dir=args.voice_dir,noise_dir=args.noise_dir+"/{}_bg".format(Class),
                  #allow_same_label=args.allow_same,
                  rate=0.5,
                  num_train=args.num_train,
                  num_validation=args.num_validation,
                  num_eval=args.num_eval,
                  random_seed=args.random_seed,
                  Class=Class)
  print(speakers)

if __name__ == '__main__':
  main()
