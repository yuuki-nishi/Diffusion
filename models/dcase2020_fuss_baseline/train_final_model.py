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
r"""Train the DCASE2020 FUSS baseline source separation model."""

import argparse
import os

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")
from train import data_io
from train import final_model
from train import train_with_estimator_final


def main():
  parser = argparse.ArgumentParser(
      description='Train the DCASE2020 FUSS baseline source separation model.')
  parser.add_argument(
      '-dd', '--data_dir', help='Data directory.',
      required=True)
  parser.add_argument(
      '-md', '--model_dir', help='Directory for checkpoints and summaries.',
      required=True)
  parser.add_argument(
      '-nograd', '--nograd', help='no gradient for discriminator',action='store_true')
  parser.add_argument(
      '-pd', '--part_grad', help='no gradient for discriminator',action='store_true')
  parser.add_argument(
      '-on', '--outnum', help='Output num',
      required=True,type=int)
  parser.add_argument(
      '-sepdir', '--SepModelDir', help='SeparationModelDir',
      required=True)
  parser.add_argument(
      '-discdir', '--DiscModelDir', help='DiscriminatorModelDir',
      required=True)
  parser.add_argument(
      '-omega2', '--omega2', help='omega2 for reconstruct',type=float,default=1.00)
  parser.add_argument(
      '-omega', '--omega', help='omega for labelloss',type=float,required=True,default=0.00)
  parser.add_argument(
      '-u', '--unit', help='unit for GRU',type=int,required=True,default=64)
  parser.add_argument(
      '-rlp', '--rawlabeltoprob', help='probability to raw label',
      type=str,default="prob")
  parser.add_argument(
      '-pdd', '--p_data_dir', help='Processed Data directory.',
      required=True)
  parser.add_argument(
      '-sz', '--ParamSize', help='ParameterSize',
      required=True)
  parser.add_argument(
      '-cf', '--configfile', help='config file path',
      required=True)
  parser.add_argument(
      '-iss', '--isstop', help='stop probability gradient or not ',action='store_true')
  
  args = parser.parse_args()
  print("omega : {}".format(args.omega))
  outnum = args.outnum
  '''
  if args.withnoise :
    #hparams = model.get_model_hparams_withNoise()
    hparams = model.get_model_hparams_withNoise_Max2()
    hparams.num_sources_for_summaries = [i+1 for i in range(outnum+1)]
  else:
    hparams = model.get_model_hparams()
    hparams.num_sources_for_summaries = [i+1 for i in range(maxnum)]
  '''
  print("otunum : {}".format(outnum))
  hparams = final_model.get_model_hparams_withOutNum(outnum)
  hparams.discparam_size=args.ParamSize
  print(hparams)
  hparams.num_sources_for_summaries = [i+1 for i in range(outnum)]
  hparams.sr = 16000.0
  #hparams.num_sources_for_summaries = [1, 2, 3, 4] #default
  roomsim_params = {
      'num_sources': len(hparams.signal_names),
      'num_receivers': 1,
      'num_samples': int(hparams.sr * 10.0),
  }
  #tf.logging.info('Params: %s', roomsim_params.values())

  feature_spec = data_io.get_roomsim_spec(**roomsim_params)
  inference_spec = data_io.get_inference_spec()
  hparams.classnum=data_io.SoundConfig(sourceroot=args.data_dir,split="train",path=args.configfile).MaxNum
  hparams.unit = args.unit
  

  params = {
      'feature_spec': feature_spec,
      'inference_spec': inference_spec,
      'hparams': hparams,
      'io_params': {'parallel_readers': 512,
                    'num_samples': int(hparams.sr * 10.0)},
      'source_root': args.data_dir,
      'model_dir': args.model_dir,
      'configpath':args.configfile,
      'sepdir':args.SepModelDir,
      'discdir':args.DiscModelDir,
      'train_batch_size': 4,#変更した、元は下も含め3#train_exampleを割り切れるようにしてくれ
      'eval_batch_size': 4,
      #'train_steps': 20000000,
      #'train_steps': trainstepnum,
      #'train_steps': 5,
      'classnum':hparams.classnum,
      'eval_suffix': 'validation',
      'eval_examples': 400,
      #'train_examples': 20000,
      'train_examples': 10000,#batchで割り切れるように
      'save_checkpoints_secs': 600,
      "train_epoch": 70,
      'save_summary_steps': 1000,
      'keep_checkpoint_every_n_hours': 4,
      'write_inference_graph': True,
      'randomize_training': True,
      "SepModelDir": args.SepModelDir,
      "DiscModelDir": args.DiscModelDir,
      'processed_data_dir': args.p_data_dir,
      "omega": args.omega,
      'outnum': outnum,
      'nograd_disc': args.nograd,
      'part_grad':args.part_grad,#部分的な勾配の更新をするか
      'discper':100,
      'isstop' : args.isstop,
      'omega2' : args.omega2,
      #'phase' : phase,
      'rlp' : args.rawlabeltoprob,
  }
  #tf.logging.info(params)
  #train_with_estimator_final.execute(final_model.model_fn, data_io.input_fn, **params)
  final_model.model_fn(params)


if __name__ == '__main__':
    main()
